import argparse
import time
import re
import os
import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from spell_correct.vocab import Vocab
from spell_correct.spelling_corrector import SpellingCorrector
from spell_correct.eval_metric import Evaluation
from spell_correct.dialect_data import load_data, process_raw_inputs


class SpellCorrectTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(
            f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
        self.vocab = Vocab.load(args.vocab_path)

        self.train_iter, self.dev_iter =  load_data(args, self.vocab, self.device)
        self.model = SpellingCorrector(args,
                                       vocab=self.vocab,
                                       bert_tokenizer=self.train_iter.dataset.bert_tokenizer,
                                       device=self.device).to(self.device)

        self.criterion_char = nn.CrossEntropyLoss(
            ignore_index=self.vocab.tgt.char2id['<pad>'])
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.writer = SummaryWriter(os.path.join(args.logs, 'tensorboard'))

        self.src_to_tgt = np.vectorize(
            lambda x: self.vocab.tgt.char2id[self.vocab.src.id2char[x]])

    
    @staticmethod
    def epoch_time(start_time: int,
                   end_time: int):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    def _compute_loss(self, outputs, tgt):
        max_word_len = self.args.max_decode_len
        tgt = tgt[:, :, 1:].reshape(-1, max_word_len)
        tgt = tgt[torch.any(tgt.bool(), dim=1)]
        outputs = outputs.view(-1, outputs.shape[-1])
        tgt = tgt.permute(1, 0).reshape(outputs.shape[0])
        return self.criterion_char(outputs, tgt)


    def _compute_word_accuracy_word_level(self, outputs_char, src, tgt):
        predictions = outputs_char.argmax(-1).permute(1, 0)
        tgt_perm = tgt[:, :, 1:]
        tgt_perm = tgt_perm.reshape(-1, self.args.max_decode_len)
        tgt_perm = tgt_perm[torch.any(tgt_perm.bool(), dim=1)]
        src_perm = src[:, :, 1:]
        src_perm = src_perm.reshape(-1, self.args.max_decode_len)
        src_perm = src_perm[torch.any(src_perm.bool(), dim=1)]

        resized_predictions = torch.cat(
            [predictions, torch.zeros_like(tgt_perm)], dim=1)[:, :tgt_perm.shape[1]]
        src_mapped_to_tgt = self.src_to_tgt(src_perm.detach().cpu().numpy())
        src_mapped_to_tgt = torch.from_numpy(src_mapped_to_tgt).to(self.device)
        
        e_mask = torch.all(src_mapped_to_tgt == tgt_perm, dim=1)
        ne_mask = torch.bitwise_not(e_mask)

        total = 0
        correct_total = [0, 0, 0, 0]
        for i, mask in enumerate([e_mask, ne_mask]):
            tgt_mask = tgt_perm[mask] != self.vocab.tgt['<pad>']
            pred_valid = resized_predictions[mask] * tgt_mask
            correct_forms = torch.all(tgt_perm[mask] == pred_valid, dim=1)
            if i == 1:
                ne_changed = torch.all(src_mapped_to_tgt[mask] == pred_valid, dim=1)
            total_len = tgt_perm[mask].shape[0]
            correct_len = torch.sum(correct_forms).item()
            correct_total[i * 2] += correct_len
            correct_total[i * 2 + 1] += total_len
            total += total_len
        correct_total.append(correct_total[-1] - torch.sum(ne_changed).item())
        assert total == src_perm.shape[0]
        return correct_total
    

    def train(self):
        metrics_train, metrics_val = {}, {}
        for epoch in range(self.args.epochs):
            self.model.train()
            print(f'Epoch {epoch+1}/{self.args.epochs}')
            epoch_loss, epoch_loss_char  = 0, 0
            start_time = time.time()
            for iteration, batch in enumerate(self.train_iter):
                tgt_char = batch['tgt_char']

                self.model.zero_grad()
                outputs_char = self.model(batch)
                loss_char = self._compute_loss(outputs_char, tgt_char)
                loss_char.backward()
                self.optimizer.step()
                epoch_loss += loss_char.item()
                if iteration and iteration % 10 == 0 and len(self.train_iter) - iteration > 10 \
                        or iteration + 1 == len(self.train_iter):
                    for param_group in self.optimizer.param_groups:
                        lr = param_group['lr']
                    print(
                        f'Batch {iteration}/{len(self.train_iter)-1}\t| train_loss {loss_char.item():.7f} | lr {lr}')
            metrics_train.setdefault('train_loss', []).append(epoch_loss / iteration)
            end_time = time.time()
            epoch_mins, epoch_secs = SpellCorrectTrainer.epoch_time(
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
            # correct_e, total_e, correct_ne, total_ne
            correct_total = [0, 0, 0, 0, 0]
            epoch_loss = 0
            for batch in self.dev_iter:
                # Loss
                outputs_char = self.model(batch, teacher_force=False)
                loss_char = self._compute_loss(outputs_char, batch['tgt_char'])
                epoch_loss += loss_char.item()
                # Accuracy - Word
                correct_total_batch = self._compute_word_accuracy_word_level(
                    outputs_char, batch['src_char'], batch['tgt_char'])

                correct_total = [sum(x) for x in zip(correct_total, correct_total_batch)]

        metrics = {}
        metrics['dev_recall'] = (correct_total[0] + correct_total[4]) / (correct_total[1] + correct_total[3])
        metrics['dev_precision'] = correct_total[2] / correct_total[3]
        metrics['dev_f1'] = 2 * (metrics['dev_recall'] * metrics['dev_precision']) / (metrics['dev_recall'] + metrics['dev_precision'])
        metrics['dev_loss'] = epoch_loss / len(self.dev_iter)
        return metrics

    def predict(self, data=None):
        if data:
            iterator = process_raw_inputs(data)
        else:
            iterator = self.dev_iter
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                src = batch['src_char']
                tgt = batch['tgt_char']
                outputs, valid_indexes = self.model(
                    batch, teacher_force=False, return_valid_indexes=True)
                predictions = outputs.argmax(-1).permute(1, 0)
                predictions = self.model._scatter(
                    predictions, valid_indexes, tgt.shape)

                matrices = [[], [], [], []]
                for i, matrix in enumerate([src.permute(1, 0, 2), tgt.permute(1, 0, 2), predictions.permute(1, 0, 2)]):
                    for sent in matrix:
                        matrices[i].append([])
                        for word in sent:
                            if word[0].item() == self.vocab.src.char2id['<pad>']:
                                break
                            matrices[i][-1].append([])
                            for char in word:
                                if char.item() == self.vocab.src.char2id['</w>']:
                                    break
                                elif char.item() == self.vocab.src.char2id['<w>']:
                                    continue
                                matrices[i][-1][-1].append(
                                    self.vocab.src.id2char[char.item()])
                            matrices[i][-1][-1] = ''.join(matrices[i][-1][-1])
                        if i == 0:
                            matrices[3].append(' '.join(matrices[i][-1]))
        return matrices

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
    def load_model(model_path: str):
        params = torch.load(model_path)
        args = params['args']
        network = SpellCorrectTrainer(args)
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
    parser.add_argument("--batch_size", default=8,
                        type=int, help="Batch size.")
    parser.add_argument("--epochs", default=25, type=int,
                        help="Number of epochs.")
    parser.add_argument("--ce_dim", default=128, type=int,
                        help="Word embedding dimension.")
    parser.add_argument("--we_dim", default=256, type=int,
                        help="Word embedding dimension.")
    parser.add_argument("--rnn_dim_char", default=256,
                        type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_dim", default=512,
                        type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=2,
                        type=int, help="Number of RNN layers.")
    parser.add_argument("--data_size", default=10000, type=int,
                        help="Maximum number of examples to load.")
    parser.add_argument("--train_split", default=0.9, type=float,
                        help="Proportion with which to split the train and dev data.")
    parser.add_argument("--max_sent_len", default=35, type=int,
                        help="Maximum length of BERT input sequence.")
    parser.add_argument("--max_decode_len", default=25, type=int,
                        help="Maximum length of BERT input sequence.")
    parser.add_argument("--dropout", default=0.2, type=float,
                        help="Probablility of dropout for encoder and decoder.")
    parser.add_argument("--use_bert_enc", default='',
                        help="How to use use BERT embeddings (either as initialization or as concatenated embeddings). Leave empty to exlcude embeddings",
                        choices=['init', 'concat', ''])
    parser.add_argument("--use_sent_level", default=False, action='store_true',
                        help="Whether or not we should translate sentences (with hybrid embeddings) instead of words (character embeddings).")
    parser.add_argument("--gpu_index", default=6, type=int,
                        help="Index of GPU to be used.")
    parser.add_argument("--vocab", dest='vocab_path',
                        default="/local/ccayral/orthonormalDA1/data/coda-corpus/beirut_vocab.json", type=str,
                        help="Path to vocab JSON file.")
    parser.add_argument("--data_src",
                        default="/local/ccayral/orthonormalDA1/data/coda-corpus/beirut_src.txt", type=str,
                        help="Path to file with src dataset.")
    parser.add_argument("--data_tgt",
                        default="/local/ccayral/orthonormalDA1/data/coda-corpus/beirut_tgt.txt", type=str,
                        help="Path to file with tgt dataset.")
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

    # args.load = '/local/ccayral/orthonormalDA1/model_weights/train-2021-05-25_18:18:40-bs=8,cd=128,ds=10000,e=23,gi=6,mdl=25,msl=35,rd=512,rdc=256,rl=2,s=42,usl=False,wd=256.pt'
    # args.use_bert_enc = 'init'

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
        trainer = SpellCorrectTrainer(args)
        metrics = trainer.train()
        with open(os.path.join(args.config_save, args.logdir + '.json'), 'w') as f:
            json.dump(vars(args), f)
        with open(os.path.join(args.logs, 'metrics-' + args.logdir + '.json'), 'w') as f:
            json.dump(metrics, f)
        print(metrics)
    else:
        trainer = SpellCorrectTrainer.load_model(args.load)
        inputs, golds, predictions, sentences = trainer.predict()
        labels = trainer.label_predictions(predictions, inputs, golds)
        with open(os.path.join(args.logs, args.logdir), 'w') as f:
            for p, i, g, s, l in zip(predictions, inputs, golds, sentences, labels):
                for p_word, i_word, g_word, r_word in zip(p, i, g, l):
                    # print(r_word[2], file=f, end='\t')
                    # print(i_word, file=f, end='\t')
                    # print(g_word, file=f, end='\t')
                    print(p_word, file=f, end='\t')
                    # print(f"{r_word[0]}\t{r_word[1]}", file=f, end='\t')
                    # print(s, file=f)


def error_analysis(args):
    trainer = SpellCorrectTrainer(args)
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
