import argparse
import string
import os
import time
from collections import Counter
import linecache
from tqdm import tqdm
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer

from gumar_dataset_char import GumarDataset


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


class Network:
    def __init__(self, args) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.load_gumar_data(args)
        self.model = Network.RNN(args,
                                 output_vocab_size=len(self.tgt_vocab),
                                 input_vocab=self.src_vocab).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())


    class RNN(nn.Module):
        def __init__(self,
                     args,
                     output_vocab_size,
                     input_vocab):
            super(Network.RNN, self).__init__()
            self.input_vocab = input_vocab
            self.hidden_dim = args.rnn_dim
            self.word_embeddings = nn.Embedding(len(input_vocab), args.we_dim)
            self.lstm = nn.GRU(args.we_dim,
                                args.rnn_dim,
                                num_layers=args.rnn_layers,
                                bidirectional=True)
            self.fc0 = nn.Linear(args.rnn_dim * 2, 100)
            self.fc1 = nn.Linear(100, output_vocab_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, sentences, lengths):
            embeds = self.dropout(self.word_embeddings(sentences))
            packed_embeds = pack_padded_sequence(
                embeds, lengths, enforce_sorted=False)
            lstm_out, hidden = self.lstm(packed_embeds)
            lstm_out, _ = pad_packed_sequence(lstm_out)
            hidden = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)
            tag_space = self.relu(self.fc0(hidden))
            tag_space = self.fc1(tag_space)
            return tag_space

    def load_gumar_data(self, args):
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
                    if token_pairs_fl[token_pair] < 1:
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

        self.src_vocab = gumar.train.data[GumarDataset.Dataset.SOURCE].chars_vocab
        self.tgt_vocab = gumar.train.data[GumarDataset.Dataset.TARGET].chars_vocab
        src_map = gumar.train.data[GumarDataset.Dataset.SOURCE].chars_map
        tgt_map = gumar.train.data[GumarDataset.Dataset.TARGET].chars_map
        PAD_IDX = gumar.train.data[GumarDataset.Dataset.SOURCE].chars_vocab['<pad>']

        def data_process(gumar, split):
            data = list(zip(getattr(gumar, split).data[GumarDataset.Dataset.SOURCE].sentences_chars_ids,
                            getattr(gumar, split).data[GumarDataset.Dataset.TARGET].sentences_chars_ids))
            data_temp = [] if split != 'dev' else [[], []]
            for src, tgt in data:
                src_tensor_ = torch.tensor(src, dtype=torch.long)
                if src.count(self.src_vocab[' ']) > 0:
                    action = 1
                elif tgt.count(self.tgt_vocab[' ']) > 0:
                    action = 2
                else:
                    action = 0
                tgt_tensor_ = torch.tensor([action], dtype=torch.long)
                (data_temp[0 if list(map(lambda x: src_map[x], src)) != list(map(lambda x: tgt_map[x], tgt)) else 1] if split ==
                'dev' else data_temp).append((src_tensor_, tgt_tensor_))
            return data_temp

        def generate_batch(data_batch):
            src_batch, tgt_batch = [], []
            lengths = []
            for (src_item, tgt_item) in data_batch:
                lengths.append(len(src_item))
                src_batch.append(src_item)
                tgt_batch.append(tgt_item)
            src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
            tgt_batch = torch.cat(tgt_batch)
            return (src_batch, tgt_batch), lengths

        train_data = data_process(gumar, 'train')
        ne_valid_data, e_valid_data = data_process(gumar, 'dev')
        test_data = data_process(gumar, 'test')

        self.train_iter = DataLoader(train_data, batch_size=args.batch_size,
                                shuffle=True, collate_fn=generate_batch)
        self.ne_valid_iter = DataLoader(ne_valid_data, batch_size=args.batch_size,
                                shuffle=True, collate_fn=generate_batch)
        self.e_valid_iter = DataLoader(e_valid_data, batch_size=args.batch_size,
                                shuffle=True, collate_fn=generate_batch)
        self.test_iter = DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=True, collate_fn=generate_batch)
        self.tgt_vocab = {'N': 0, 'M': 1, 'S': 2}

        data = Data(self.src_vocab, {'N': 1, 'M': 2, 'S': 2}, src_map, ['N', 'M', 'S'], self.train_iter,
                    self.ne_valid_iter, self.e_valid_iter, self.test_iter, device)
        return data

    @staticmethod
    def epoch_time(start_time: int,
                   end_time: int):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train(self, args):
        metrics = {'train_loss': [], 'ne_loss': [], 'e_loss': [],
                   'ne_word_acc': [], 'e_word_acc': []
                   }
        for epoch in range(args.epochs):
            self.model.train()
            print(f'Epoch {epoch+1}/{args.epochs}')
            epoch_loss = 0
            start_time = time.time()
            for iteration, train_batch in enumerate(self.train_iter):
                (docs, tags), lengths = train_batch
                self.model.zero_grad()
                docs, tags = docs.to(self.device), tags.to(self.device)
                tag_scores = self.model(docs, lengths)
                tag_scores = tag_scores.view(-1, tag_scores.shape[-1])
                tags = tags.view(-1)
                loss = self.loss_function(tag_scores, tags)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if iteration % 10 == 0 and len(self.train_iter) - iteration > 10 \
                        or iteration + 1 == len(self.train_iter):
                    print(
                        f'Batch {iteration}/{len(self.train_iter)-1}\t| Loss {loss.item():.7f}')
                    metrics['train_loss'].append(epoch_loss / len(self.train_iter))
            end_time = time.time()
            epoch_mins, epoch_secs = Network.epoch_time(start_time, end_time)
            val_metrics = self.evaluate()
            for m in metrics:
                if m != 'train_loss':
                    metrics[m].append(val_metrics[m])
            print(
                f'Epoch {epoch+1}/{args.epochs} | Time: {epoch_mins}m {epoch_secs}s')
            print(
                f"\tTrain Loss: {metrics['train_loss'][-1]:.7f} | Val. Eq. Loss: {metrics['e_loss'][-1]:.7f} | Val. N.Eq. Loss: {metrics['ne_loss'][-1]:.7f}")
            print(
                f"\tVal. Eq. Acc.: {metrics['e_word_acc'][-1]:.1%} | Val. N.Eq. Acc.: {metrics['ne_word_acc'][-1]:.1%}")
            print()
        torch.save(self.model.state_dict(), f"{args.cpt}/rnn_text.pt")
        return metrics

    def evaluate(self):
        self.model.eval()
        metrics = {}
        with torch.no_grad():
            for i, iterator in enumerate([self.ne_valid_iter, self.e_valid_iter]):
                correct_coda_forms, total_coda_forms = 0, 0
                epoch_loss = 0
                for (src, tgt), lengths in iterator:
                    # Loss
                    src, tgt = src.to(self.device), tgt.to(self.device)
                    output = self.model(src, lengths)
                    loss = self.loss_function(output, tgt)
                    epoch_loss += loss.item()

                    # Accuracy
                    output = output.argmax(1)
                    correct_coda_forms += torch.sum(output == tgt)
                    total_coda_forms += tgt.shape[0]

                metrics['ne_word_acc' if i ==
                        0 else 'e_word_acc'] = correct_coda_forms / total_coda_forms
                metrics['ne_loss' if i == 0 else 'e_loss'] = epoch_loss / \
                    len(iterator)
        return metrics

    def test(self, args, data):
        self.load_data(args, x_test=data['test'][0], y_test=data['test'][1])
        return self.evaluate(args, test=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64,
                        type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int,
                        help="Number of epochs.")
    parser.add_argument("--we_dim", default=128, type=int,
                        help="Word embedding dimension.")
    parser.add_argument("--max_sent_len", default=2000, type=int,
                        help="Maximum length of a sentence.")
    parser.add_argument("--data_size", default=3000, type=int,
                        help="Maximum number of examples to load.")
    parser.add_argument("--cpt", default='cpt', type=str,
                        help="Directory to save the model checkpoints to.")
    parser.add_argument("--rnn_dim", default=512,
                        type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=1,
                        type=int, help="Number of RNN layers.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    network = Network(args)
    metrics = network.train(args)

    with open(f'outputs/results.txt', 'w') as f:
        print('Train and Val:', file=f)
        for e in range(args.epochs):
            print(f"{metrics['train_loss'][e]}\t{metrics['val_loss'][e]}\t{metrics['val_precision'][e]}\t{metrics['val_recall'][e]}\t{metrics['val_fscore'][e]}", file=f)
        for input_type in ['input_unpunctuated', 'input_asr']:
            data = preprocess_data(args, how2, input_type, test=True)
            metrics = network.test(args, data)
            print(f"Test {input_type}:", file=f)
            print(
                f"{metrics['test_precision']}\t{metrics['test_recall']}\t{metrics['test_fscore']}", file=f)


if __name__ == '__main__':
    main()
