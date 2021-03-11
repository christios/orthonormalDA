import time
import argparse
import io
import random
from collections import Counter
from typing import Tuple
import pickle
import os

# Required non-native dependencies
from nltk import bigrams
import numpy as np

import torch
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
        # embedded = [src_len, batch_size, cle_dim]
        embedded = self.dropout(self.embedding(src))
        # outputs: [src_len, batch_size, 2*rnn_dim] final-layer hidden states
        # hidden: [2*rnn_layers, batch_size, rnn_dim] is the final hidden state of each layer-direction
        # hidden: [forward_1, backward_1, forward_2, backward_2, ...]
        outputs, hidden = self.rnn(embedded)
        # hidden[-2, :, :]: [1, batch_size, rnn_dim]
        # backward_forward: [batch_size, rnn_dim * 2]
        backward_forward = torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]), dim=1)
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
        # embedded: [1, batch_size, cle_dim]
        embedded = self.dropout(self.embedding(input))
        # weighted_encoder_rep: [1, batch_size, 2*rnn_dim]
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)
        # rnn_input: [1, batch_size, 2*rnn_dim + cle_dim]
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        # output: [1, batch_size, rnn_dim] tgt_len = 1 (decoding steps)
        # decoder_hidden: [1, batch_size, rnn_dim]
        # output == decoder_hidden
        output, decoder_hidden = self.rnn(
            rnn_input, decoder_hidden.unsqueeze(0))
        # embedded: [batch_size, cle_dim]
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
                teacher_forcing_ratio: float = 1) -> Tensor:
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


def train_epoch(model: nn.Module,
                iterator: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
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
        optimizer.step()
        epoch_loss += loss.item()
        if iteration % 50 == 0 or iteration + 1 == len(iterator):
            rand_word = random.randint(0, src.shape[1] - 1)
            prediction = model(
                src[:, rand_word:rand_word+1], tgt[:, rand_word:rand_word+1])
            prediction = prediction[1:].view(-1, prediction.shape[-1])
            prediction = prediction.argmax(1)
            visualization = visualize_prediction(
                rand_word, prediction, src, tgt, src_map, tgt_map)
            print(
                f'Batch {iteration}/{len(iterator)-1}\t| Loss {loss.item():.4f} {visualization}')
    return epoch_loss / len(iterator)


def train(args, data, model):
    best_valid_loss = float('inf')
    results = []
    for epoch in range(args.epochs):
        print(
            f"Epoch {epoch+1}/{args.epochs} | lr: {optimizer.param_groups[0]['lr']}")
        start_time = time.time()
        train_loss = train_epoch(model, data.train_iter, optimizer, criterion,
                                 data.device, data.src_map, data.tgt_map)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        iterators = [data.ne_valid_iter, data.e_valid_iter] if not args.bigram else [
            data.ne_valid_iter]
        metrics = evaluate(model, iterators, criterion, data.device)
        scheduler.step()
        r = (train_loss, metrics['ne_loss'], metrics['e_loss'], metrics['ne_word_acc'], metrics['e_word_acc']) if not args.bigram \
            else (train_loss, metrics['loss'], metrics['word_acc'])

        print(
            f'Epoch {epoch+1}/{args.epochs} | Time: {epoch_mins}m {epoch_secs}s')
        if not args.bigram:
            print(
                f"\tTrain Loss: {r[0]:.4f} | Val. N.eq. Loss: {r[1]:.4f} | Val. Eq. Loss: {r[2]:.4f}")
            print(f"\tVal. N.eq. Acc.: {r[3]:.1%} | Val. Eq. Acc.: {r[4]:.1%}")
        else:
            print(f"\tTrain Loss: {r[0]:.4f} | Val. N.eq. Loss: {r[1]:.4f}")
            print(f"\tVal. N.eq. Acc.: {r[2]:.1%}")
        print()
        results.append(r)
    return results


def evaluate(model: nn.Module,
             iterators: torch.utils.data.DataLoader,
             criterion: nn.Module,
             device):
    model.eval()
    metrics = {}
    with torch.no_grad():
        for i, iterator in enumerate(iterators):
            correct_coda_forms, total_coda_forms = 0, 0
            epoch_loss = 0
            for _, (src, tgt) in enumerate(iterator):
                # Loss
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt, teacher_forcing_ratio=0)
                output = output[1:].view(-1, output.shape[-1])
                tgt_lst = tgt[1:].view(-1)
                loss = criterion(output, tgt_lst)
                epoch_loss += loss.item()

                # Accuracy
                predictions = model(src, tgt, 0).permute(
                    1, 0, 2).argmax(2)[:, 1:]
                tgt = tgt.permute(1, 0)[:, 1:]
                tgt[tgt == GumarDataset.Factor.EOW] = 0
                resized_predictions = torch.cat(
                    [predictions, torch.zeros_like(tgt)], dim=1)[:, :tgt.shape[1]]
                resized_predictions[resized_predictions == 3] = 0
                total_coda_forms += tgt.shape[0]
                correct_coda_forms += torch.sum(torch.all(tgt == resized_predictions * (
                    tgt != GumarDataset.Factor.PAD), dim=1))
            acc_name = ('ne_word_acc' if i == 0 else 'e_word_acc') if len(
                iterators) == 2 else 'word_acc'
            loss_name = ('ne_loss' if i == 0 else 'e_loss') if len(
                iterators) == 2 else 'loss'
            metrics[acc_name] = (correct_coda_forms / total_coda_forms).item()
            metrics[loss_name] = epoch_loss / len(iterator)
    return metrics


def visualize_prediction(i, prediction, src, tgt, src_map, tgt_map):
    raw = "".join(src_map[s.item()] for s in src[:, i] if s)[5:-5]
    gold_coda = "".join(tgt_map[t.item()] for t in tgt[:, i] if t)[5:-5]
    system_coda = "".join(
        tgt_map[p.item()] for p in prediction if p != GumarDataset.Factor.EOW)
    visualization = "{}{}{}".format(f'<r>{raw}<r>'.rjust(
        25), f'<g>{gold_coda}<g>'.rjust(25), f'<s>{system_coda}<s>'.rjust(25))
    return visualization


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
                 train_iter,
                 ne_valid_iter, e_valid_iter,
                 ne_test_iter, e_test_iter,
                 device) -> None:
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_map = src_map
        self.tgt_map = tgt_map
        self.train_iter = train_iter
        self.ne_valid_iter = ne_valid_iter
        self.e_valid_iter = e_valid_iter
        self.ne_test_iter = ne_test_iter
        self.e_test_iter = e_test_iter
        self.device = device


def load_gumar_data(args):
    """In addition to gumar_dataset_char.py, this function does some more preprocessing.
    The gumar data would usually be loaded using the gumar_dataset_char.py script,
    but since this would require the acutal dataset which I am not allowed
    to share with anyone due to copyright issues, I pickled the essential data in
    the gumar_char file. """
    # gumar = GumarDataset('annotated-gumar-corpus')

    with open('gumar_char', 'rb') as g:
        gumar = pickle.load(g)
        SOURCE = GumarDataset.Dataset.SOURCE  # 0
        TARGET = GumarDataset.Dataset.TARGET  # 1
        sentences_chars_ids_src = gumar.train.data[SOURCE].sentences_chars_ids
        sentences_chars_ids_tgt = gumar.train.data[TARGET].sentences_chars_ids
        chars_map_src = gumar.train.data[SOURCE].chars_map
        chars_map_tgt = gumar.train.data[TARGET].chars_map

    print("Length of dataset before capping word length to 30:", len(
        sentences_chars_ids_src))
    temp = []
    for i in range(len(sentences_chars_ids_src)):
        if len(sentences_chars_ids_src[i]) < args.max_word_len and \
                len(sentences_chars_ids_src[i]) < args.max_word_len:
            temp.append(
                (sentences_chars_ids_src[i], sentences_chars_ids_tgt[i]))
    sentences_chars_ids_src = [c[0] for c in temp]
    sentences_chars_ids_tgt = [c[1] for c in temp]
    print("Length of dataset after capping word length to 30:", len(
        sentences_chars_ids_src))

    token_pairs_fl = Counter()
    words_src, words_tgt = [], []
    src_sentences_chars = [''.join(map(lambda x: chars_map_src[x], word))
                           for word in sentences_chars_ids_src]
    tgt_sentences_chars = [''.join(map(lambda x: chars_map_tgt[x], word))
                           for word in sentences_chars_ids_tgt]
    for i, token_pair in enumerate(zip(src_sentences_chars, tgt_sentences_chars)):
        if token_pairs_fl[token_pair] < args.max_num_pair_types:
            words_src.append(sentences_chars_ids_src[i])
            words_tgt.append(sentences_chars_ids_tgt[i])
        token_pairs_fl.update([token_pair])
    gumar.train.data[SOURCE].sentences_chars_ids = words_src
    gumar.train.data[TARGET].sentences_chars_ids = words_tgt
    print("Length of dataset after limiting number of types to 50:", len(
        gumar.train.data[SOURCE].sentences_chars_ids))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_vocab = gumar.train.data[SOURCE].chars_vocab
    tgt_vocab = gumar.train.data[TARGET].chars_vocab
    src_map = chars_map_src
    tgt_map = chars_map_tgt
    PAD_IDX = src_vocab['<pad>']
    BOW_IDX = src_vocab['<bow>']
    EOW_IDX = src_vocab['<eow>']

    def data_process(gumar, split):
        """Splits the validation and test dataset into two parts, one in which
        source-target pairs have equal source and target (e), and another in which
        source and target are not equal in the pair (ne). This is done for better
        monitoring of accuracy because of the class imbalance (70% of examples
        are e and 30% are ne."""
        data = list(zip(getattr(gumar, split).data[SOURCE].sentences_chars_ids,
                        getattr(gumar, split).data[TARGET].sentences_chars_ids))
        data_temp = [[], []] if split != 'train' else []
        for src, tgt in data:
            src_tensor_ = torch.tensor(src, dtype=torch.long)
            tgt_tensor_ = torch.tensor(tgt, dtype=torch.long)
            pair_ne = list(map(lambda x: src_map[x], src)) != list(
                map(lambda x: tgt_map[x], tgt))
            dataset_temp_split = data_temp[0 if pair_ne else 1] if split != 'train' else data_temp
            dataset_temp_split.append((src_tensor_, tgt_tensor_))
        return data_temp

    def generate_batch(data_batch):
        src_batch, tgt_batch = [], []
        for (src_item, tgt_item) in data_batch:
            src_batch.append(
                torch.cat([torch.tensor([BOW_IDX]), src_item, torch.tensor([EOW_IDX])], dim=0))
            tgt_batch.append(
                torch.cat([torch.tensor([BOW_IDX]), tgt_item, torch.tensor([EOW_IDX])], dim=0))
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    train_data = data_process(gumar, 'train')
    ne_valid_data, e_valid_data = data_process(gumar, 'dev')
    ne_test_data, e_test_data = data_process(gumar, 'test')

    # If bigram mode is on, dataset is augmented into bigrams (see documentation)
    if args.bigram:
        data_bi = []
        for dataset in [('train', train_data), ('val', ne_valid_data)]:
            merges = [t for t in dataset[1] if list(
                t[0]).count(src_vocab[' '])]
            no_merges = [t for t in dataset[1] if list(
                t[0]).count(src_vocab[' ']) == 0]

            no_merges_src_bi = bigrams([ex[0] for ex in no_merges])
            no_merges_src_bi = [torch.cat(
                [bi[0], torch.tensor([src_vocab[' ']]), bi[1]], dim=0) for bi in no_merges_src_bi]
            no_merges_tgt_bi = bigrams([ex[1] for ex in no_merges])
            no_merges_tgt_bi = [torch.cat(
                [bi[0], torch.tensor([tgt_vocab[' ']]), bi[1]], dim=0) for bi in no_merges_tgt_bi]
            no_merges = list(zip(no_merges_src_bi, no_merges_tgt_bi))
            data_bi.append(merges + no_merges)

        train_data = data_bi[0]
        ne_valid_data = data_bi[1]

    train_iter = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, collate_fn=generate_batch)
    ne_valid_iter = DataLoader(ne_valid_data, batch_size=args.batch_size,
                               shuffle=True, collate_fn=generate_batch)
    e_valid_iter = DataLoader(e_valid_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=generate_batch)
    ne_test_iter = DataLoader(ne_test_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=generate_batch)
    e_test_iter = DataLoader(e_test_data, batch_size=args.batch_size,
                             shuffle=True, collate_fn=generate_batch)

    data = Data(src_vocab, tgt_vocab, src_map, tgt_map, train_iter,
                ne_valid_iter, e_valid_iter, ne_test_iter, e_test_iter, device)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256,
                        type=int, help="Number of elements per batch.")
    parser.add_argument("--cle_dim", default=128, type=int,
                        help="Character-level embeddings dimension (in our case a sequence is a word not a sentence).")
    parser.add_argument("--epochs", default=30, type=int,
                        help="Number of epochs.")
    parser.add_argument("--rnn_dim", default=256, type=int,
                        help="RNN hidden layer dimension.")
    parser.add_argument("--attn_dim", default=16, type=int,
                        help="Attention layer output dimension.")
    parser.add_argument("--dropout", default=0.5, type=float,
                        help="Dropout probability (after embedding) in both encoder and decoder.")
    parser.add_argument("--max_word_len", default=30, type=int,
                        help="Maximum sequence (word) length to allow in the data.")
    parser.add_argument("--max_num_pair_types", default=50, type=int,
                        help="Maximum number of occurences per type of source-target pair in the data.")
    parser.add_argument("--load", default='', type=str,
                        help="Load model weights without training.")
    parser.add_argument("--save", default='', type=str,
                        help="Name of file to save model weights to after training.")
    parser.add_argument("--bigram", default=False, action='store_true',
                        help="Augments dataset to bigrams as described in the documentation.")
    parser.add_argument("--predict", default=False, action='store_true',
                        help="After training or loading weights, outputs predictions based on the test set inputs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load preprocessed data and preprocess some more
    data = load_gumar_data(args)

    encoder = Encoder(input_dim=len(data.src_vocab),
                      emb_dim=args.cle_dim,
                      enc_hid_dim=args.rnn_dim,
                      dec_hid_dim=args.rnn_dim,
                      dropout=args.dropout)
    attention = Attention(enc_hid_dim=args.rnn_dim,
                          dec_hid_dim=args.rnn_dim,
                          attn_dim=args.attn_dim)
    decoder = Decoder(output_dim=len(data.tgt_vocab),
                      emb_dim=args.cle_dim,
                      enc_hid_dim=args.rnn_dim,
                      dec_hid_dim=args.rnn_dim,
                      dropout=args.dropout,
                      attention=attention)

    model = Seq2Seq(encoder=encoder,
                    decoder=decoder,
                    device=data.device).to(data.device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=17, gamma=0.3)
    criterion = nn.CrossEntropyLoss(ignore_index=data.tgt_vocab['<pad>'])

    if args.load:
        model.load_state_dict(torch.load(args.load))
    else:
        results = train(args, data, model)
        with open('results.txt', 'w') as f:
            for epoch in range(args.epochs):
                print(results[epoch], file=f)
        if args.save:
            torch.save(model.state_dict(), args.save)

    iterators = [data.ne_test_iter, data.e_test_iter]
    metrics = evaluate(model, iterators, criterion, data.device)
    print(
        f"Test N.eq. Acc.: {metrics['ne_word_acc']:.1%} | Test Eq. Acc.: {metrics['e_word_acc']:.1%}")

    if args.predict:
        model.eval()
        with open('predictions.txt', 'w') as f, torch.no_grad():
            for _, (src, tgt) in enumerate(data.ne_test_iter):
                src, tgt = src.to(data.device), tgt.to(data.device)
                predictions = model(src, tgt, teacher_forcing_ratio=0).permute(
                    1, 0, 2).argmax(2)[:, 1:]
                for i, prediction in enumerate(predictions):
                    visualization = visualize_prediction(
                        i, prediction, src, tgt, data.src_map, data.tgt_map)
                    print(visualization, file=f)
