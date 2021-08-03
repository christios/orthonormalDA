import argparse
import time
import re
import os
import datetime
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from segmentation.vocab import Vocab
from segmentation.segmenter import Segmenter
from segmentation.dialect_data import load_data, process_raw_inputs
from segmentation.evalutation import process_indices_batch


class SegmentationTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(
            f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
        self.vocab = Vocab.load(args.vocab_path)

        if args.train_split:
            self.train_iter, self.dev_iter =  load_data(args, self.vocab, self.device)
            bert_tokenizer = self.train_iter.dataset.bert_tokenizer
        else:
            self.test_iter = load_data(args, self.vocab, self.device)
            bert_tokenizer = self.test_iter.dataset.bert_tokenizer
        
            
        self.model = Segmenter(args,
                                       vocab=self.vocab,
                                       bert_tokenizer=bert_tokenizer,
                                       device=self.device).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.tgt.char2id['<pad>'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.writer = SummaryWriter(os.path.join(args.logs, 'tensorboard'))

    
    @staticmethod
    def epoch_time(start_time: int,
                   end_time: int):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    def _compute_loss(self, outputs, tgt):
        max_word_len = self.args.max_word_len
        tgt = tgt.reshape(-1, max_word_len + 1)
        tgt = tgt[torch.any(tgt != self.vocab.src.char2id['<pad>'], dim=1)]
        outputs = outputs.view(-1, outputs.shape[-1])
        tgt = tgt.permute(1, 0).reshape(outputs.shape[0])
        return self.criterion(outputs, tgt)


    def _compute_word_accuracy_word_level(self, outputs_char, src, tgt):
        predictions = outputs_char
        tgt_perm = tgt.reshape(-1, self.args.max_word_len + 1)
        tgt_valid = torch.any(tgt_perm != self.vocab.tgt.char2id['<pad>'], dim=1)
        tgt_perm = tgt_perm[tgt_valid]
        src_perm = src.reshape(-1, self.args.max_word_len + 1)
        src_valid = torch.any(src_perm != self.vocab.tgt.char2id['<pad>'], dim=1)
        src_perm = src_perm[src_valid]
        sensitivity_specificity = process_indices_batch((predictions, tgt_perm), positive_label=1, pad_label=2)
        return sensitivity_specificity
    

    def train(self):
        metrics_train, metrics_val = {}, {}
        for epoch in range(self.args.epochs):
            self.model.train()
            print(f'Epoch {epoch+1}/{self.args.epochs}')
            epoch_loss  = 0
            start_time = time.time()
            for iteration, batch in enumerate(self.train_iter):
                self.model.zero_grad()
                output = self.model(batch, use_crf=self.args.use_crf)
                if self.args.use_crf:
                    loss = output['loss']
                else:
                    loss = self._compute_loss(output['lstm_feats'], batch['tgt_char'])

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if iteration and iteration % 10 == 0 and len(self.train_iter) - iteration > 10 \
                        or iteration + 1 == len(self.train_iter):
                    for param_group in self.optimizer.param_groups:
                        lr = param_group['lr']
                    print(
                        f'Batch {iteration}/{len(self.train_iter)-1}\t| train_loss {loss.item():.7f} | lr {lr}')
            metrics_train.setdefault('train_loss', []).append(epoch_loss / iteration)
            end_time = time.time()
            epoch_mins, epoch_secs = SegmentationTrainer.epoch_time(
                start_time, end_time)
            log_output = 'Evaluation...\n'
            metrics_val_step = self.evaluate()
            for m in metrics_val_step:
                metrics_val.setdefault(m, []).append(metrics_val_step[m])
            metrics = {**metrics_train, **metrics_val}
            print(
                f'Epoch {epoch+1}/{self.args.epochs} | Time: {epoch_mins}m {epoch_secs}s')
            for m, v in metrics.items():
                log_output += (f"\t{m.ljust(25)}: {metrics[m][-1]:.7f}\n" if 'loss' in m
                    else f"\t{m.ljust(25)}: {metrics[m][-1]:.1%}\n")
            print(log_output)
            self.scheduler.step(metrics['dev_loss'][-1])

        self.save_model()
        return metrics

    def evaluate(self):

        self.model.eval()
        with torch.no_grad():
            sensitivity_specificity = [0, 0, 0]
            epoch_loss = 0
            for batch in self.dev_iter:

                output = self.model(batch, use_crf=self.args.use_crf, decode=True)
                if self.args.use_crf:
                    loss = output['loss']
                    sensitivity_specificity_batch = self._compute_word_accuracy_word_level(
                        output['outputs'], batch['src_char'], batch['tgt_char'])
                else:
                    loss = self._compute_loss(output['lstm_feats'], batch['tgt_char'])
                    outputs = output['lstm_feats'].argmax(-1).permute(1, 0)
                    sensitivity_specificity_batch = self._compute_word_accuracy_word_level(
                        outputs, batch['src_char'], batch['tgt_char'])
                epoch_loss += loss.item()
                sensitivity_specificity_batch = [sensitivity_specificity_batch['tp'], sensitivity_specificity_batch['fn'], sensitivity_specificity_batch['fp']]
                sensitivity_specificity = [sum(x) for x in zip(
                    sensitivity_specificity, sensitivity_specificity_batch)]

        metrics = {}
        metrics['dev_recall'] = sensitivity_specificity[0] / \
            (sensitivity_specificity[0] + sensitivity_specificity[1]
             ) if sensitivity_specificity[0] + sensitivity_specificity[1] != 0 else 1e-10
        metrics['dev_precision'] = sensitivity_specificity[0] / \
            (sensitivity_specificity[0] + sensitivity_specificity[2]) if sensitivity_specificity[0] + sensitivity_specificity[2] != 0 else 1e-10
        metrics['dev_f1'] = 2 * (metrics['dev_recall'] * metrics['dev_precision']) / (metrics['dev_recall'] + metrics['dev_precision'])
        metrics['dev_loss'] = epoch_loss / len(self.dev_iter)
        return metrics

    def predict(self, data=None):
        if data:
            iterator = process_raw_inputs(data)
        else:
            iterator = self.test_iter
        
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(self.test_iter))
            output = self.model(
                batch, use_crf=self.args.use_crf, decode=True)
            
            src = batch['src_char']
            
            src_perm = src.reshape(-1, self.args.max_word_len + 1)
            src_valid = torch.any(
                src_perm != self.vocab.tgt.char2id['<pad>'], dim=1)
            src_perm = src_perm[src_valid]

            counter = Counter()
            for i, o in enumerate(output['outputs']):
                pred = ''.join([self.vocab.src.id2char[char_s] + ('+' if label else '')
                    for char_s, label in zip(src_perm[i].tolist(), o)])
                counter.update(pred.split('+'))
            
            pass



    def label_predictions(self, predictions, raw_inputs, raw_golds):
        train_corpus = [
            token for sent in self.train_iter.dataset.src_raw for token in sent.split()]
        labels = []
        for pred, raw_input, raw_gold in zip(predictions, raw_inputs, raw_golds):
            labels.append([])
            for word_pred, word_raw_input, word_raw_gold in zip(pred, raw_input, raw_gold):
                labels[-1].append(
                    ('EQUAL' if word_raw_input == word_raw_gold else 'NOT EQUAL', 
                     'CORRECT' if word_pred == word_raw_gold else 'INCORRECT',
                     'IN_TRAIN' if word_raw_input in train_corpus else ''))
        return labels

    
    @staticmethod
    def find_mask_index(seq, s_type):
        mask_index = torch.where(seq == s_type)[0]
        if mask_index.size:
            mask_index = mask_index[0].item()
        else:
            mask_index = len(seq)
        return mask_index

    @staticmethod
    def load_model(model_path: str, data_path: str):
        params = torch.load(model_path)
        args = params['args']
        args.train_split = 0
        args.data = data_path
        network = SegmentationTrainer(args)
        network.model.load_state_dict(params['state_dict'])
        return network

    def save_model(self):
        save_path = os.path.join(self.args.cpt, self.args.logdir) + '.pt'
        print('Saving model parameters to [%s]\n' % save_path)
        params = {
            'args': self.args,
            'state_dict': self.model.state_dict()
        }
        torch.save(params, save_path)

    def visualize_model(self):
        batch = iter(self.train_iter).next()
        self.writer.add_graph(self.model, {k: v for k, v in batch.items()
                                           if k not in ['src_bert', 'src_bert_mask', 'src_raw', 'tgt_raw']})
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16,
                        type=int, help="Batch size.")
    parser.add_argument("--epochs", default=25, type=int,
                        help="Number of epochs.")
    parser.add_argument("--ce_dim", default=128, type=int,
                        help="Word embedding dimension.")
    parser.add_argument("--rnn_dim_char", default=256,
                        type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=2,
                        type=int, help="Number of RNN layers.")
    parser.add_argument("--data_size", default=10000, type=int,
                        help="Maximum number of examples to load.")
    parser.add_argument("--train_split", default=0.9, type=float,
                        help="Proportion with which to split the train and dev data.")
    parser.add_argument("--max_sent_len", default=35, type=int,
                        help="Maximum length of BERT input sequence.")
    parser.add_argument("--max_word_len", default=25, type=int,
                        help="Maximum length of BERT input sequence.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Probablility of dropout for encoder and decoder.")
    parser.add_argument("--use_bert_enc", default='',
                        help="How to use use BERT embeddings (either as initialization or as concatenated embeddings). Leave empty to exlcude embeddings",
                        choices=['init', 'concat', ''])
    parser.add_argument("--use_crf", default=True, action='store_true',
                        help="Whether or not we should add the CRF layer on top of the LSTM output.")
    parser.add_argument("--gpu_index", default=6, type=int,
                        help="Index of GPU to be used.")
    parser.add_argument("--vocab", dest='vocab_path',
                        default="/local/ccayral/orthonormalDA1/data/coda-corpus/beirut_vocab.json", type=str,
                        help="Path to vocab JSON file.")
    parser.add_argument("--data",
                        default="/local/ccayral/orthonormalDA1/data/asc", type=str,
                        help="Path to file with src dataset.")
    parser.add_argument("--bert_cache_dir",
                        default="/local/ccayral/.transformer_models/MARBERT_pytorch_verison", type=str,
                        help="Path to dir with the cached tokenizer and encoder BERT models.")
    parser.add_argument("--bert_model",
                        default="UBC-NLP/MARBERT", type=str,
                        help="BERT model name.")
    parser.add_argument("--cpt", default='/local/ccayral/orthonormalDA1/model_weights', type=str,
                        help="Directory to save the model checkpoints to.")
    parser.add_argument("--logs", default='/local/ccayral/orthonormalDA1/logs', type=str,
                        help="Directory to save the model checkpoints to.")
    parser.add_argument("--load", default='', type=str,
                        help="Directory to save the model checkpoints to.")
    parser.add_argument("--config_save", default='/local/ccayral/orthonormalDA1/train_configs', type=str,
                        help="Directory to save configuration to.")

    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    args.load = '/local/ccayral/orthonormalDA1/model_weights/train-2021-08-02_22:35:52-bs=16,cd=128,ds=10000,e=25,gi=6,msl=35,mwl=25,rdc=256,rl=2,s=42,uc=True.pt'
    args.use_bert_enc = 'init'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.logdir = "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook").split('.')[0]),
        datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if isinstance(value, int)))
    )
    # error_analysis(args)
    if not args.load:
        trainer = SegmentationTrainer(args)
        metrics = trainer.train()
        with open(os.path.join(args.config_save, args.logdir + '.json'), 'w') as f:
            json.dump(vars(args), f)
        with open(os.path.join(args.logs, 'metrics-' + args.logdir + '.json'), 'w') as f:
            json.dump(metrics, f)
        print(metrics)
    else:
        trainer = SegmentationTrainer.load_model(
            args.load, data_path="/local/ccayral/orthonormalDA1/data/coda-corpus")
        trainer.predict()

        with open(os.path.join(args.logs, args.logdir), 'w') as f:
            pass


def error_analysis(args):
    trainer = SegmentationTrainer(args)
    with open('/local/ccayral/orthonormalDA1/logs/train-2021-05-22_10:34:44-bs=16,cd=128,ds=10000,e=20,gi=6,mdl=25,msl=35,rd=512,rdc=256,rl=1,s=42,ube=False,usl=False,wd=256') as f:
        e, ne = [], []
        for line in f:
            line = line.split('\t')
            if line[3] == 'EQUAL':
                e.append(tuple(line[:3]))
            elif line[3] == 'NOT EQUAL':
                ne.append(tuple(line[:3]))
    train_corpus = [
        token for sent in trainer.train_iter.dataset.src_raw for token in sent.split()]
    total, in_train = 0, 0
    for token in e:
        total += 1
        if token[0] in train_corpus:
            in_train += 1
    ratio = in_train / total
    pass




if __name__ == '__main__':
    main()
