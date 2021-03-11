import time
import argparse
import io
import random
from collections import Counter
from typing import Tuple
import math
from tqdm import tqdm
import pickle
import os
from sys import stdout

import numpy as np

import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive

from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable

from gumar_dataset_temp import GumarDataset


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float,
                 rnn_layers: bool):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.rnn_layers = rnn_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, num_layers=2 if rnn_layers else 1)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src_w: Tensor,
                src_c: Tensor,
                lengths_w) -> Tuple[Tensor]:
        """- src: [src_len, batch_size]
           - 2 is the number of directions (forward/backward)"""
        # embedded = [src_len, batch_size, we_dim]
        embedded_w = self.dropout(self.embedding(src_w))
        if lengths_w:
            embedded_w = pack_padded_sequence(embedded_w, lengths_w, enforce_sorted=False)
        # outputs: [src_len, batch_size, 2*rnn_dim] final-layer hidden states
        # hidden: [2*rnn_layers, batch_size, rnn_dim] is the final hidden state of each layer-direction
        # hidden: [forward_1, backward_1, forward_2, backward_2, ...]
        outputs_w, hidden_w = self.rnn(embedded_w)
        if lengths_w:
            outputs_w, _ = pad_packed_sequence(outputs_w)
        
        forward_w = (hidden_w[0, :, :] + hidden_w[2, :, :]) if self.rnn_layers else hidden_w[0, :, :]
        backward_w = (hidden_w[1, :, :] + hidden_w[3, :, :]) if self.rnn_layers else hidden_w[1, :, :]

        # hidden[-2, :, :]: [1, batch_size, rnn_dim]
        # backward_forward: [batch_size, rnn_dim * 2]
        backward_forward_w = torch.cat((forward_w, backward_w), dim=1)
        hidden_w = torch.tanh(self.fc(backward_forward_w))
        #outputs = [src_len, batch_size, rnn_dim * 2] because 2 directions
        #hidden = [batch_size, rnn_dim]
        return outputs_w, hidden_w


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:
        """decoder_hidden: [batch_size, rnn_dim]
           encoder_outputs: [src_len, batch_size, 2*rnn_dim]"""
        src_len = encoder_outputs.shape[0]
        # decoder_hidden: [batch_size, 1, dec rnn_dim]
        decoder_hidden = decoder_hidden.unsqueeze(1)
        # repeated_hidden: [batch_size, src_len, rnn_dim]
        repeated_decoder_hidden = decoder_hidden.repeat(1, src_len, 1)
        # encoder_outputs: [batch_size, src_len, 2*rnn_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # decoder_encoder: [batch_size, src_len, 3*rnn_dim]
        decoder_encoder = torch.cat(
            (repeated_decoder_hidden, encoder_outputs), dim=2)
        # energy: [batch_size, src_len, attn_dim]
        energy = torch.tanh(self.attn(decoder_encoder))
        # attention: [batch_size, src_len]
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:
        """decoder_hidden: [batch_size, rnn_dim]
           encoder_outputs: [src_len, batch_size, 2*rnn_dim]"""
        # a: [batch_size, src_len]
        a = self.attention(decoder_hidden, encoder_outputs)
        # a: [batch_size, 1, src_len]
        a = a.unsqueeze(1)
        # encoder_outputs: [batch_size, src_len, 2*rnn_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # weighted_encoder_rep: [batch_size, 1, 2*rnn_dim] batch-matrix-matrix product
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        # weighted_encoder_rep: [1, batch_size, 2*rnn_dim]
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:
        """ input: [batch_size] Note: "one character at a time"
            decoder_hidden: [batch_size, dec rnn_dim]
            encoder_outputs: [src_len, batch_size, 2*rnn_dim]"""
        # input: [1, batch_size]
        input = input.unsqueeze(0)
        # embedded: [1, batch_size, we_dim]
        embedded = self.dropout(self.embedding(input))
        # weighted_encoder_rep: [1, batch_size, 2*rnn_dim]
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)
        # rnn_input: [1, batch_size, 2*rnn_dim + we_dim]
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        # output: [1, batch_size, rnn_dim] tgt_len = 1 (decoding steps)
        # decoder_hidden: [1, batch_size, rnn_dim]
        # output == decoder_hidden
        output, decoder_hidden = self.rnn(
            rnn_input, decoder_hidden.unsqueeze(0))
        # embedded: [batch_size, we_dim]
        # output: [batch_size, rnn_dim]
        embedded, output = embedded.squeeze(0), output.squeeze(0)
        # weighted_encoder_rep: [batch_size, 2*rnn_dim]
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        # output = [batch_size, tgt_vocab]
        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim=1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src_w: Tensor,
                tgt_w: Tensor,
                src_c: Tensor,
                tgt_c: Tensor,
                lengths=None,
                teacher_forcing_ratio: float = 0.5) -> Tensor:
        """- src = [src_len, batch_size]
           - tgt = [tgt_len, batch_size]
           - teacher_forcing_ratio is probability to use teacher forcing
             e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time"""
        batch_size = src_w.shape[1]
        max_len = tgt_w.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size,
                              tgt_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src_w, src_c, lengths[0] if lengths else lengths)
        # first input to the decoder is the <sos> token
        output = tgt_w[0, :]
        for t in range(1, max_len):
            # output = [batch_size, tgt_vocab]
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (tgt_w[t] if teacher_force else top1)

        return outputs


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float,
          device,
          src_map,
          tgt_map):

    model.train()
    epoch_loss = 0
    for iteration, (src_w, tgt_w, src_c, tgt_c, lengths) in enumerate(iterator):
        src_w, tgt_w = src_w.to(device), tgt_w.to(device)
        src_c, tgt_c = src_c.to(device), tgt_c.to(device)
        optimizer.zero_grad()
        output = model(src_w, tgt_w, src_c, tgt_c, lengths)
        output = output[1:].view(-1, output.shape[-1])
        tgt_lst = tgt_w[1:].contiguous().view(-1)
        loss = criterion(output, tgt_lst)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        if iteration % 10 == 0 or iteration + 1 == len(iterator):
            visualize_training(iteration, iterator, loss, src_w, tgt_w, src_map, tgt_map)
    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module,
             device):
    model.eval()
    metrics = {}
    with torch.no_grad():
        correct_coda_forms, total_coda_forms = 0, 0
        epoch_loss = 0
        for src_w, tgt_w, src_c, tgt_c, lengths in iterator:
            # Loss
            src_w, tgt_w = src_w.to(device), tgt_w.to(device)
            output = predict_batch(src_w, tgt_w)
            tgt_lst = tgt_w[1:].view(-1)
            loss = criterion(output, tgt_lst)
            epoch_loss += loss.item()

            # Accuracy
            predictions = model(src_w, tgt_w, ..., ..., teacher_forcing_ratio=0).permute(
                1, 0, 2).argmax(2)[:, 1:]
            tgt_w = tgt_w.permute(1, 0)[:, 1:]
            tgt_w[tgt_w == GumarDataset.Factor.EOW] = 0
            resized_predictions = torch.cat(
                [predictions, torch.zeros_like(tgt_w)], dim=1)[:, :tgt_w.shape[1]]
            resized_predictions[resized_predictions == 3] = 0
            total_coda_forms += tgt_w.shape[0]
            correct_coda_forms += torch.sum(torch.all(tgt_w == resized_predictions * (
                tgt_w != GumarDataset.Factor.PAD), dim=1))
        metrics['val_word_acc'] = correct_coda_forms / total_coda_forms
        metrics['val_loss'] = epoch_loss / len(iterator)
    return metrics


def predict_batch(src, tgt):
    output = model(src, tgt, ..., ..., teacher_forcing_ratio=0)  # turn off teacher forcing
    output = output[1:].view(-1, output.shape[-1])
    return output


def visualize_training(iteration, iterator, loss, src_w, tgt_w, src_map, tgt_map):
    rand_word = random.randint(0, src_w.shape[1] - 1)
    prediction = predict_batch(src=src_w[:, rand_word:rand_word+1],
                               tgt=tgt_w[:, rand_word:rand_word+1])
    prediction = prediction.argmax(1)
    raw = "".join(src_map[s.item()]
                  for s in src_w[:, rand_word] if s)[5:-5]
    gold_coda = "".join(tgt_map[t.item()]
                        for t in tgt_w[:, rand_word] if t)[5:-5]
    system_coda = "".join(
        tgt_map[p.item()] for p in prediction if p != GumarDataset.Factor.EOW)
    status = "{}{}{}".format(f'<r>{raw}<r>'.rjust(
        25), f'<g>{gold_coda}<g>'.rjust(25), f'<s>{system_coda}<s>'.rjust(25))
    print(
        f'Batch {iteration}/{len(iterator)-1}\t| Loss {loss.item():.4f} {status}')


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Data:
    def __init__(self,
                 src_vocab_word, tgt_vocab_word,
                 src_vocab_char, tgt_vocab_char,
                 src_map_word, tgt_map_word,
                 src_map_char, tgt_map_char,
                 train_iter, valid_iter, test_iter,
                 device) -> None:
        self.src_vocab_word = src_vocab_word
        self.tgt_vocab_word = tgt_vocab_word
        self.src_vocab_char = src_vocab_char
        self.tgt_vocab_char = tgt_vocab_char
        self.src_map_word = src_map_word
        self.tgt_map_word = tgt_map_word
        self.src_map_char = src_map_char
        self.tgt_map_char = tgt_map_char
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.device = device


def load_gumar_data():
    # if os.path.exists('data/gumar_wle_cle'):
    #     with open('data/gumar_wle_cle', 'rb') as g:
    #         gumar = pickle.load(g)
    # else:
    #     gumar = GumarDataset('annotated-gumar-corpus')
    #     with open('data/gumar_wle_cle', 'wb') as g:
    #         pickle.dump(gumar, g)
    gumar = GumarDataset('annotated-gumar-corpus', max_sentence_len=30, add_bow_eow=True)
    # print("Length of dataset before capping sentence length:", len(
    #     gumar.train.data[gumar.Dataset.SOURCE].sentences_chars_ids))
    # temp = [[], []]
    # for f in range(gumar.Dataset.FACTORS):
    #     for s in range(len(gumar.train.data[f].sentences_chars_ids)):
    #         temp[f].append([])
    #         for w in range(len(gumar.train.data[f].sentences_chars_ids[s])):
    #             if len(gumar.train.data[f].sentences_chars_ids[s][w]) < 30 and \
    #                     len(gumar.train.data[f^1].sentences_chars_ids[s][w]) < 30:
    #                 temp[f][-1].append(True)
    #             else:
    #                 temp[f][-1].append(False)
    # for f in range(gumar.Dataset.FACTORS):
    #     dataset = gumar.train.data[f]
    #     for structure in ['sentences_chars_ids', 'sentences_chars', 'sentences_words_ids', 'sentences_words']:
    #         setattr(dataset, structure, [[word for w, word in enumerate(
    #             sent) if temp[f][s][w]] for s, sent in enumerate(getattr(dataset, structure))])
    # print("Number of words discarded:", len(
    #     gumar.train.data[gumar.Dataset.SOURCE].sentences_chars_ids))
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_vocab_char = gumar.train.data[GumarDataset.Dataset.SOURCE].chars_vocab
    tgt_vocab_char = gumar.train.data[GumarDataset.Dataset.TARGET].chars_vocab
    src_vocab_word = gumar.train.data[GumarDataset.Dataset.SOURCE].words_vocab
    tgt_vocab_word = gumar.train.data[GumarDataset.Dataset.TARGET].words_vocab
    src_map_char = gumar.train.data[GumarDataset.Dataset.SOURCE].chars_map
    tgt_map_char = gumar.train.data[GumarDataset.Dataset.TARGET].chars_map
    src_map_word = gumar.train.data[GumarDataset.Dataset.SOURCE].words_map
    tgt_map_word = gumar.train.data[GumarDataset.Dataset.TARGET].words_map
    
    train_data = gumar.train.data_words_chars
    valid_data = gumar.dev.data_words_chars
    test_data = gumar.test.data_words_chars
    
    generate_batch = GumarDataset.Dataset.generate_batch
    
    train_iter = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(valid_data, batch_size=args.batch_size,
                            shuffle=True, collate_fn=generate_batch)
    test_iter = DataLoader(test_data, batch_size=args.batch_size,
                           shuffle=True, collate_fn=generate_batch)

    data = Data(src_vocab_word, tgt_vocab_word, src_vocab_char, tgt_vocab_char,
                src_map_word, tgt_map_word, src_map_char, tgt_map_char,
                train_iter, valid_iter, test_iter,
                device)
    return data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32,
                        type=int, help="batch_size.")
    parser.add_argument("--cle_dim", default=64, type=int,
                        help="Character-level embeddings dimension.")
    parser.add_argument("--we_dim", default=64, type=int,
                        help="Word-level embeddings dimension.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs.")
    parser.add_argument("--max_sentences", default=5000,
                        type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--rnn_dim", default=64, type=int,
                        help="RNN cell dimension.")
    parser.add_argument("--cell", default='gru',
                        type=str, help="RNN cell type.")
    parser.add_argument("--rnn_layers", default=False, action='store_true',
                        help="Whether to include one or two RNN layers for the encoder.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_gumar_data()
    
    INPUT_DIM = len(data.src_vocab_word)
    OUTPUT_DIM = len(data.tgt_vocab_word)
    ATTN_DIM = 8
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, args.we_dim, args.rnn_dim, args.rnn_dim, ENC_DROPOUT, args.rnn_layers)
    attn = Attention(args.rnn_dim, args.rnn_dim, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, args.we_dim, args.rnn_dim,
                args.rnn_dim, DEC_DROPOUT, attn)
    
    model = Seq2Seq(enc, dec, data.device).to(data.device)
    
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())

    PAD_IDX = data.tgt_vocab_word['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    CLIP = 1
    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        start_time = time.time()
        train_loss = train(model, data.train_iter, optimizer, criterion,
                           CLIP, data.device, data.src_map_word, data.tgt_map_word)
        metrics = evaluate(model, data.valid_iter, criterion, data.device)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch {epoch+1}/{args.epochs} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f"\tTrain Loss: {train_loss:.4f} | Val. Eq. Acc.: {metrics['e_word_acc']:.1%} | Val. NotEq. Acc.: {metrics['ne_word_acc']:.1%}")
        print()
    test_loss = evaluate(model, data.test_iter, criterion, data.device)
    print(f'| Test Loss: {test_loss:.4f}')
