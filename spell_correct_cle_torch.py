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
from torch.nn.utils.rnn import pad_sequence

from gumar_dataset_char import GumarDataset


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:
        """- src: [src_len, batch_size]
           - 2 is the number of directions (forward/backward)"""
        # embedded = [src_len, batch_size, we_dim]
        embedded = self.dropout(self.embedding(src))
        # outputs: [src_len, batch_size, 2*rnn_dim] final-layer hidden states
        # hidden: [2*rnn_layers, batch_size, rnn_dim] is the final hidden state of each layer-direction
        # hidden: [forward_1, backward_1, forward_2, backward_2, ...]
        outputs, hidden = self.rnn(embedded)
        # hidden[-2, :, :]: [1, batch_size, rnn_dim]
        # backward_forward: [batch_size, rnn_dim * 2]
        backward_forward = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = torch.tanh(self.fc(backward_forward))
        #outputs = [src_len, batch_size, rnn_dim * 2] because 2 directions
        #hidden = [batch_size, rnn_dim]
        return outputs, hidden


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
                src: Tensor,
                tgt: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:
        """- src = [src_len, batch_size]
           - tgt = [tgt_len, batch_size]
           - teacher_forcing_ratio is probability to use teacher forcing
             e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time"""
        batch_size = src.shape[1]
        max_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size,
                              tgt_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        # first input to the decoder is the <sos> token
        output = tgt[0, :]
        for t in range(1, max_len):
            # output = [batch_size, tgt_vocab]
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (tgt[t] if teacher_force else top1)

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
    for iteration, (src, tgt) in enumerate(iterator):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        output = output[1:].view(-1, output.shape[-1])
        tgt_lst = tgt[1:].view(-1)
        loss = criterion(output, tgt_lst)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        if iteration % 200 == 0 or iteration + 1 == len(iterator):
            rand_word = random.randint(0, src.shape[1] - 1)
            prediction = predict_batch(
                src[:, rand_word:rand_word+1], tgt[:, rand_word:rand_word+1])
            prediction = prediction.argmax(1)
            raw = "".join(src_map[s.item()] for s in src[:, rand_word] if s)[5:-5]
            gold_coda = "".join(tgt_map[t.item()] for t in tgt[:, rand_word] if t)[5:-5]
            system_coda = "".join(tgt_map[p.item()] for p in prediction if p != GumarDataset.Factor.EOW)
            status = "{}{}{}".format(f'<r>{raw}<r>'.rjust(
                25), f'<g>{gold_coda}<g>'.rjust(25), f'<s>{system_coda}<s>'.rjust(25))
            print(
                f'Batch {iteration}/{len(iterator)-1}\t| Loss {loss.item():.4f} {status}')
    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             ne_iterator: torch.utils.data.DataLoader,
             e_iterator: torch.utils.data.DataLoader,
             criterion: nn.Module,
             device):
    model.eval()
    metrics = {}
    with torch.no_grad():
        for i, iterator in enumerate([ne_iterator, e_iterator]):
            correct_coda_forms, total_coda_forms = 0, 0
            epoch_loss = 0
            for _, (src, tgt) in enumerate(iterator):
                # Loss
                src, tgt = src.to(device), tgt.to(device)
                output = predict_batch(src, tgt)
                tgt_lst = tgt[1:].view(-1)
                loss = criterion(output, tgt_lst)
                epoch_loss += loss.item()

                # Accuracy
                predictions = model(src, tgt, 0).permute(1, 0, 2).argmax(2)[:, 1:]
                tgt = tgt.permute(1, 0)[:, 1:]
                tgt[tgt == GumarDataset.Factor.EOW] = 0
                resized_predictions = torch.cat(
                    [predictions, torch.zeros_like(tgt)], dim=1)[:, :tgt.shape[1]]
                resized_predictions[resized_predictions == 3] = 0
                total_coda_forms += tgt.shape[0]
                correct_coda_forms += torch.sum(torch.all(tgt == resized_predictions * (
                    tgt != GumarDataset.Factor.PAD), dim=1))
            metrics['ne_word_acc' if i == 0 else 'e_word_acc'] = correct_coda_forms / total_coda_forms
            metrics['ne_loss' if i == 0 else 'e_loss'] = epoch_loss / len(iterator)
    return metrics

def predict_batch(src, tgt):
    output = model(src, tgt, 0)  # turn off teacher forcing
    output = output[1:].view(-1, output.shape[-1])
    return output


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Data:
    def __init__(self,
                 src_vocab, tgt_vocab,
                 src_map, tgt_map,
                 train_iter, ne_valid_iter, e_valid_iter, test_iter,
                 device) -> None:
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_map = src_map
        self.tgt_map = tgt_map
        self.train_iter = train_iter
        self.ne_valid_iter = ne_valid_iter
        self.e_valid_iter = e_valid_iter
        self.test_iter = test_iter
        self.device = device


def load_data():
    def build_vocab(filepath, tokenizer):
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    def data_process(filepaths):
        raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
            de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)],
                                    dtype=torch.long)
            en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                                    dtype=torch.long)
            data.append((de_tensor_, en_tensor_))
        return data

    def generate_batch(data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(
                torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
            en_batch.append(
                torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
        return de_batch, en_batch
    
    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.de.gz', 'train.en.gz')
    val_urls = ('val.de.gz', 'val.en.gz')
    test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

    train_filepaths = [extract_archive(
        download_from_url(url_base + url))[0] for url in train_urls]
    val_filepaths = [extract_archive(
        download_from_url(url_base + url))[0] for url in val_urls]
    test_filepaths = [extract_archive(
        download_from_url(url_base + url))[0] for url in test_urls]

    de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

    train_data = data_process(train_filepaths)
    val_data = data_process(val_filepaths)
    test_data = data_process(test_filepaths)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PAD_IDX = de_vocab['<pad>']
    BOS_IDX = de_vocab['<bos>']
    EOS_IDX = de_vocab['<eos>']

    train_iter = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(val_data, batch_size=args.batch_size,
                            shuffle=True, collate_fn=generate_batch)
    test_iter = DataLoader(test_data, batch_size=args.batch_size,
                        shuffle=True, collate_fn=generate_batch)

    data = Data(de_vocab, en_vocab, train_iter, valid_iter, test_iter, device)
    
    return data

def load_gumar_data():
    if os.path.exists('data/gumar_char'):
        with open('data/gumar_char', 'rb') as g:
            gumar = pickle.load(g)
            print("Length of dataset before capping sentence length:", len(
                gumar.train.data[gumar.Dataset.SOURCE].sentences_chars_ids))
            temp = []
            for i in range(len(gumar.train.data[0].sentences_chars_ids)):
                if len(gumar.train.data[0].sentences_chars_ids[i]) < 30 and \
                        len(gumar.train.data[1].sentences_chars_ids[i]) < 30:
                    temp.append(
                        (gumar.train.data[0].sentences_chars_ids[i], gumar.train.data[1].sentences_chars_ids[i]))
            gumar.train.data[0].sentences_chars_ids = [c[0] for c in temp]
            gumar.train.data[1].sentences_chars_ids = [c[1] for c in temp]
            print("Length of dataset after capping sentence length to 30:", len(
                gumar.train.data[gumar.Dataset.SOURCE].sentences_chars_ids))
            token_pairs_fl = Counter()
            words_src, words_tgt = [], []
            src_sentences_chars = [''.join(map(lambda x: gumar.train.data[gumar.train.SOURCE].chars_map[x], word))
                                   for word in gumar.train.data[0].sentences_chars_ids]
            tgt_sentences_chars = [''.join(map(lambda x: gumar.train.data[gumar.train.TARGET].chars_map[x], word))
                                   for word in gumar.train.data[1].sentences_chars_ids]
            for i, token_pair in enumerate(zip(src_sentences_chars, tgt_sentences_chars)):
                if token_pairs_fl[token_pair] < 50:
                    words_src.append(
                        gumar.train.data[0].sentences_chars_ids[i])
                    words_tgt.append(
                        gumar.train.data[1].sentences_chars_ids[i])
                token_pairs_fl.update([token_pair])
            gumar.train._data[0].sentences_chars_ids = words_src
            gumar.train._data[1].sentences_chars_ids = words_tgt
            print("Length of dataset after keeping only examples which coontain errors:", len(
                gumar.train.data[gumar.Dataset.SOURCE].sentences_chars_ids))

    else:
        gumar = GumarDataset('annotated-gumar-corpus')
        with open('data/gumar_char', 'wb') as g:
            pickle.dump(gumar, g)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_vocab = gumar.train.data[GumarDataset.Dataset.SOURCE].chars_vocab
    tgt_vocab = gumar.train.data[GumarDataset.Dataset.TARGET].chars_vocab
    src_map = gumar.train.data[GumarDataset.Dataset.SOURCE].chars_map
    tgt_map = gumar.train.data[GumarDataset.Dataset.TARGET].chars_map
    PAD_IDX = gumar.train.data[GumarDataset.Dataset.SOURCE].chars_vocab['<pad>']
    BOS_IDX = gumar.train.data[GumarDataset.Dataset.SOURCE].chars_vocab['<bow>']
    EOS_IDX = gumar.train.data[GumarDataset.Dataset.SOURCE].chars_vocab['<eow>']

    def data_process(gumar, split, src_map=None, tgt_map=None):
        data = list(zip(getattr(gumar, split).data[GumarDataset.Dataset.SOURCE].sentences_chars_ids,
                        getattr(gumar, split).data[GumarDataset.Dataset.TARGET].sentences_chars_ids))
        data_temp = [] if split != 'dev' else [[] ,[]]
        for src, tgt in data:
            src_tensor_ = torch.tensor(src, dtype=torch.long)
            tgt_tensor_ = torch.tensor(tgt, dtype=torch.long)
            (data_temp[0 if list(map(lambda x: src_map[x], src)) != list(map(lambda x: tgt_map[x], tgt)) else 1] if split ==
             'dev' else data_temp).append((src_tensor_, tgt_tensor_))
        return data_temp
    
    def generate_batch(data_batch):
        src_batch, tgt_batch = [], []
        for (src_item, tgt_item) in data_batch:
            src_batch.append(
                torch.cat([torch.tensor([BOS_IDX]), src_item, torch.tensor([EOS_IDX])], dim=0))
            tgt_batch.append(
                torch.cat([torch.tensor([BOS_IDX]), tgt_item, torch.tensor([EOS_IDX])], dim=0))
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    train_data = data_process(gumar, 'train')
    ne_valid_data, e_valid_data = data_process(gumar, 'dev', src_map, tgt_map)
    test_data = data_process(gumar, 'test')

    train_iter = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, collate_fn=generate_batch)
    ne_valid_iter = DataLoader(ne_valid_data, batch_size=args.batch_size,
                            shuffle=True, collate_fn=generate_batch)
    e_valid_iter = DataLoader(e_valid_data, batch_size=args.batch_size,
                               shuffle=True, collate_fn=generate_batch)
    test_iter = DataLoader(test_data, batch_size=args.batch_size,
                           shuffle=True, collate_fn=generate_batch)

    data = Data(src_vocab, tgt_vocab, src_map, tgt_map, train_iter,
                ne_valid_iter, e_valid_iter, test_iter, device)
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
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_gumar_data()
    
    INPUT_DIM = len(data.src_vocab)
    OUTPUT_DIM = len(data.tgt_vocab)
    ATTN_DIM = 8
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, args.we_dim, args.rnn_dim, args.rnn_dim, ENC_DROPOUT)
    attn = Attention(args.rnn_dim, args.rnn_dim, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, args.we_dim, args.rnn_dim,
                args.rnn_dim, DEC_DROPOUT, attn)
    
    model = Seq2Seq(enc, dec, data.device).to(data.device)
    
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())

    # PAD_IDX = data.tgt_vocab.stoi['<pad>']
    PAD_IDX = data.tgt_vocab['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    CLIP = 1
    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        start_time = time.time()
        train_loss = train(model, data.train_iter, optimizer, criterion,
                           CLIP, data.device, data.src_map, data.tgt_map)
        metrics = evaluate(model, data.ne_valid_iter,
                              data.e_valid_iter, criterion, data.device)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch {epoch+1}/{args.epochs} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f"\tTrain Loss: {train_loss:.4f} | Val. Eq. Acc.: {metrics['e_word_acc']:.1%} | Val. NotEq. Acc.: {metrics['ne_word_acc']:.1%}")
        print()
    test_loss = evaluate(model, data.test_iter, criterion, data.device)
    print(f'| Test Loss: {test_loss:.4f}')


