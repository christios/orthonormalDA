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
from sklearn.metrics import precision_score, recall_score, f1_score

from spell_correct.vocab import Vocab
from spell_correct.joint_learner import JointLearner
from spell_correct.dialect_data import load_data, preprocess
from spell_correct.models import TaxonomyTagger1

class Trainer:
    def __init__(self, args, vocab) -> None:
        self.args = args
        self.features = args.features.split()
        self.device = torch.device(
            f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.vocab = vocab
        self.train_iter, self.dev_iter, self.annotations = load_data(
            args, self.vocab, self.device, args.load)
        self.model = JointLearner(args,
                                 vocab=self.vocab,
                                 bert_tokenizer=self.dev_iter.dataset.bert_tokenizer,
                                 device=self.device).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.tgt.char2id['<pad>'])
        self.criterion_char = nn.CrossEntropyLoss(
            ignore_index=self.vocab.tgt.char2id['<pad>'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=10, factor=0.5)
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
        max_word_len = self.args.max_word_len
        tgt = tgt.reshape(-1, max_word_len + 1)
        tgt = tgt[torch.any(tgt != self.vocab.src.char2id['<pad>'], dim=1)]
        outputs = outputs.view(-1, outputs.shape[-1])
        tgt = tgt.permute(1, 0).reshape(outputs.shape[0])
        return self.criterion(outputs, tgt)

    def _compute_loss_spell_correct(self, outputs, tgt):
        max_word_len = self.args.max_decode_len
        tgt = tgt[:, :, 1:].reshape(-1, max_word_len)
        tgt = tgt[torch.any(tgt.bool(), dim=1)]
        outputs = outputs.view(-1, outputs.shape[-1])
        tgt = tgt.permute(1, 0).reshape(outputs.shape[0])
        return self.criterion_char(outputs, tgt)

    def _compute_loss_taxonomy(self, outputs, tgt):
        outputs = torch.stack([tokens for tokens in outputs.values()]).permute(1, 0, 2)
        outputs = outputs.reshape(-1, outputs.size(-1))
        tgt = tgt.reshape(-1)
        return self.criterion_char(outputs, tgt)

    def _compute_metrics_taxonomy(self, outputs, tgt):
        categories_metrics = {tag_id: dict() 
            for tag_id in self.model.taxonomy_tagger.taxonomy_categories_layers}
        outputs = {tag_id: tokens.argmax(-1).detach().cpu().numpy() for tag_id, tokens in outputs.items()}
        tgt = {tag_id: labels.detach().cpu().numpy() for tag_id, labels in enumerate(tgt.permute(1, 0))}
        for tag_id in range(len(tgt)):
            categories_metrics[tag_id]['precision'] = precision_score(
                tgt[tag_id], outputs[tag_id], pos_label=self.vocab.taxonomy.word2id['<y>'])
            categories_metrics[tag_id]['recall'] = recall_score(
                tgt[tag_id], outputs[tag_id], pos_label=self.vocab.taxonomy.word2id['<y>'])
            categories_metrics[tag_id]['f1'] = f1_score(
                tgt[tag_id], outputs[tag_id], pos_label=self.vocab.taxonomy.word2id['<y>'])
        return categories_metrics

    def _compute_accuracy(self, outputs, features_labels):
        outputs = {f: torch.tensor(outputs[f], device=self.device)[
            :, self.args.window_size] for f in outputs}
        features_labels = {f: features_labels[f].permute(
            1, 0)[:, self.args.window_size] for f in features_labels}
        correct = {f: torch.sum(features_labels[f] == outputs[f]).item()
                   for f in features_labels}
        total = {f: features_labels[f].size(0) for f in features_labels}
        return correct, total

    def _compute_precision_recall(self, outputs, features_labels):
        outputs = {f: torch.tensor(outputs[f], device=self.device)[
            :, self.args.window_size] for f in outputs}
        features_labels = {f: features_labels[f].permute(
            1, 0)[:, self.args.window_size] for f in features_labels}
        not_na_mask = {f: features_labels[f] != getattr(
            self.vocab, f).word2id['NA'] for f in features_labels}
        not_na_outputs = {f: outputs[f] != getattr(self.vocab, f).word2id['NA']
                          for f in outputs}
        recall = {f: (torch.sum(not_na_mask[f] & not_na_outputs[f]) / torch.sum(not_na_mask[f])).item()
                  for f in features_labels}
        precision = {f: (torch.sum((features_labels[f] == outputs[f]) * not_na_mask[f]) / torch.sum(not_na_mask[f])).item()
                     for f in features_labels}
        return precision, recall

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
                ne_changed = torch.all(
                    src_mapped_to_tgt[mask] == pred_valid, dim=1)
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
            epoch_loss = 0
            start_time = time.time()
            for iteration, batch in enumerate(self.train_iter):
                self.model.zero_grad()
                loss = 0
                # Tagger
                output = self.model(batch, use_crf=self.args.use_crf)
                if output['tagger'] is not None and self.args.mode != 'taxonomy':
                    loss += output['tagger']['loss']
                # Standardizer
                if output['standardizer'] is not None and self.args.mode != 'taxonomy':
                    loss_standardizer = self._compute_loss_spell_correct(
                        output['standardizer'], batch['tgt_char'])
                    loss += loss_standardizer
                # Taxonomy
                if output['taxonomy'] is not None:
                    loss += self._compute_loss_taxonomy(
                        output['taxonomy'], batch['taxonomy'])
                if self.args.mode == 'tagger':
                    loss /= len(self.features)
                elif self.args.mode == 'joint':
                    loss /= len(self.features) + 1
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()
                epoch_loss += loss.item()
                if iteration and iteration % 50 == 0 and len(self.train_iter) - iteration > 10 \
                        or iteration + 1 == len(self.train_iter):
                    for param_group in self.optimizer.param_groups:
                        lr = param_group['lr']
                    print(
                        f'Batch {iteration}/{len(self.train_iter)-1}\t| train_loss {loss.item():.7f} | lr {lr}')
            metrics_train.setdefault('train_loss', []).append(
                epoch_loss / iteration)
            end_time = time.time()
            epoch_mins, epoch_secs = Trainer.epoch_time(
                start_time, end_time)
            log_output = 'Evaluation...\n'
            metrics_val_step = self.evaluate()
            for m in metrics_val_step:
                metrics_val.setdefault(m, []).append(metrics_val_step[m])
            metrics = {**metrics_train, **metrics_val}
            print(
                f'Epoch {epoch+1}/{self.args.epochs} | Time: {epoch_mins}m {epoch_secs}s')
            for m, v in metrics.items():
                if 'precision' in m or 'recall' in m:
                    continue
                elif 'loss' in m:
                    log_output += f"\t{m.ljust(25)}: {metrics[m][-1]:.7f}\n" 
                elif 'f1' in m and 'taxonomy' not in m:
                    f = m[4:-3]
                    precision = metrics[f'dev_{f}_precision'][-1]
                    recall = metrics[f'dev_{f}_recall'][-1]
                    log_output += f"\t{m.ljust(25)}: {metrics[m][-1]:.1%}  {precision:.1%}  {recall:.1%}\n"
                elif 'f1' in m and 'taxonomy' in m:
                    tag_id = m[13:-3]
                    tag = self.vocab.taxonomy.id2taxonomy[self.args.new_taxonomy]
                    tag = tag[tag.index('-') + 2:]
                    precision = metrics[f'dev_taxonomy_{tag_id}_precision'][-1]
                    recall = metrics[f'dev_taxonomy_{tag_id}_recall'][-1]
                    log_output += f"\t{tag[:23].ljust(25)}: {metrics[m][-1]:.1%}  {precision:.1%}  {recall:.1%}\n"
                else:
                    log_output += f"\t{m.ljust(25)}: {metrics[m][-1]:.1%}\n"
            print(log_output)
            self.scheduler.step(metrics['dev_loss'][-1])

        self.save_model()
        return metrics

    def evaluate(self):
        constant_features = ['pos']
        grammatical_features = [f for f in self.features if f not in constant_features]
        self.model.eval()
        with torch.no_grad():
            correct_total = [0, 0, 0, 0, 0]
            correct, total = {f: 0 for f in constant_features}, {
                f: 0 for f in constant_features}
            precision, recall = {f: 0 for f in grammatical_features}, {
                f: 0 for f in grammatical_features}
            epoch_loss = 0
            for batch in self.dev_iter:
                loss = 0
                output = self.model(
                    batch, use_crf=self.args.use_crf, decode=True, teacher_force=False)
                # Standardizer
                if output['standardizer'] is not None and self.args.mode != 'taxonomy':
                    loss += self._compute_loss_spell_correct(output['standardizer'],
                                           batch['tgt_char'])
                    correct_total_batch = self._compute_word_accuracy_word_level(
                        output['standardizer'], batch['src_char'], batch['tgt_char'])
                    correct_total = [sum(x) for x in zip(
                        correct_total, correct_total_batch)]
                if output['taxonomy'] is not None:
                    loss +=  self._compute_loss_taxonomy(
                        output['taxonomy'], batch['taxonomy'])
                    taxonomy_metrics = self._compute_metrics_taxonomy(
                        output['taxonomy'], batch['taxonomy'])
                    
                # Tagger
                if output['tagger'] is not None and self.args.mode != 'taxonomy':
                    if self.args.use_crf:
                        loss += output['tagger']['loss']
                        c, t = self._compute_accuracy(
                            {k: v for k, v in output['tagger']['features_outputs'].items() if k in constant_features},
                            {k: v for k, v in output['tagger']['features_labels'].items() if k in constant_features})
                        p, r = self._compute_precision_recall(
                            {k: v for k, v in output['tagger']['features_outputs'].items() if k in grammatical_features},
                            {k: v for k, v in output['tagger']['features_labels'].items() if k in grammatical_features})
                        for f in constant_features:
                            correct[f] += c[f]
                            total[f] += t[f]
                        for f in grammatical_features:
                            precision[f] += p[f]
                            recall[f] += r[f]
                    else:
                        loss = self._compute_loss(
                            output['tagger']['lstm_feats'], batch['tgt_char'])
                        outputs = output['tagger']['lstm_feats'].argmax(
                            -1).permute(1, 0)
                        sensitivity_specificity_batch = self._compute_accuracy(
                            outputs, batch['src_char'], batch['tgt_char'])
                    
                if self.args.mode == 'tagger':
                    loss /= len(self.features)
                elif self.args.mode == 'joint':
                    loss /= len(self.features) + 1
                
                epoch_loss += loss.item()

        metrics = {}
        if output['standardizer'] is not None and self.args.mode != 'taxonomy':
            metrics['dev_std_recall'] = (
                correct_total[0] + correct_total[4]) / (correct_total[1] + correct_total[3])
            metrics['dev_std_precision'] = correct_total[2] / correct_total[3]
            metrics['dev_std_f1'] = 2 * (metrics['dev_std_recall'] * metrics['dev_std_precision']) / (
            metrics['dev_std_recall'] + metrics['dev_std_precision'])
        if output['taxonomy'] is not None:
            for tag_id, m in taxonomy_metrics.items():
                metrics[f'dev_taxonomy_{tag_id}_precision'] = m['precision']
                metrics[f'dev_taxonomy_{tag_id}_recall'] = m['recall']
                metrics[f'dev_taxonomy_{tag_id}_f1'] = m['f1']
        if output['tagger'] is not None and self.args.mode != 'taxonomy':
            for f in constant_features:
                metrics[f'dev_{f}_accuracy'] = correct[f] / total[f]
            for f in grammatical_features:
                metrics[f'dev_{f}_precision'] = precision[f]
                metrics[f'dev_{f}_recall'] = recall[f] 
                metrics[f'dev_{f}_f1'] = 2 * (precision[f] * recall[f]) / (precision[f] + recall[f]) if precision[f] + recall[f] else 0
        metrics['dev_loss'] = epoch_loss / len(self.dev_iter)
        return metrics

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.dev_iter:
                output = self.model(
                    batch, use_crf=self.args.use_crf, decode=True, output_loss=False)
        features_tags = {}
        for f in self.features:
            features_tags[f] = [seg[self.args.window_size] for seg in output['features_outputs'][f]]
        return features_tags
            

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
        # args.load = True
        # args.train_split = 0
        args.new_taxonomy = 13
        vocab = params['vocab']
        network = Trainer(args, vocab)
        network.model.load_state_dict(params['state_dict'])
        # network.model.taxonomy_tagger = TaxonomyTagger1(input_dim=25,
        #                                       vocab=network.vocab,
        #                                       most_common=args.taxonomy_most_common).to(network.device)
        return network

    def save_model(self):
        save_path = os.path.join(self.args.cpt, self.args.logdir) + '.pt'
        print('Saving model parameters to [%s]\n' % save_path)
        params = {
            'args': self.args,
            'state_dict': self.model.state_dict(),
            'vocab': self.vocab
        }
        torch.save(params, save_path)

    def visualize_model(self):
        batch = iter(self.train_iter).next()
        self.writer.add_graph(self.model, {k: v for k, v in batch.items()
                                           if k not in ['src_bert', 'src_bert_mask', 'src_raw', 'tgt_raw']})
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4,
                        type=int, help="Batch size.")
    parser.add_argument("--epochs", default=25, type=int,
                        help="Number of epochs.")
    parser.add_argument("--ce_dim", default=128, type=int,
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
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate.")
    parser.add_argument("--clip", default=0.5, type=float,
                        help="Learning rate.")
    parser.add_argument("--max_sent_len", default=35, type=int,
                        help="Maximum length of BERT input sequence.")
    parser.add_argument("--max_decode_len", default=25, type=int,
                        help="Maximum length of BERT input sequence.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Probablility of dropout for encoder and decoder.")
    parser.add_argument("--use_bert_enc", default='',
                        help="How to use use BERT embeddings (either as initialization or as concatenated embeddings). Leave empty to exlcude embeddings",
                        choices=['init', 'concat', ''])
    parser.add_argument("--window_size", default=7, type=int,
                        help="How much context to the left and right of segment do we want to take (in number of segments).")
    parser.add_argument("--use_crf", default=True, action='store_true',
                        help="Whether or not we should add the CRF layer on top of the LSTM output.")
    parser.add_argument('--features', nargs='+',
                        default='pos state number gender person voice mood aspect verbForm',
                        # default='pos',
                        help='Grammatical features that we are training for')
    parser.add_argument("--taxonomy_most_common", default=1, type=int,
                        help="Number of taxonomy tags to use (from most to least common).")
    parser.add_argument("--features_layer", default=64, type=int,
                        help="Features layer dimension.")
    parser.add_argument("--mode", default='tagger',
                        help="Training mode.",
                        choices=['tagger', 'standardizer', 'standardizer_taxonomy', 'joint', 'taxonomy'])
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

    # args.load = '/local/ccayral/orthonormalDA1/model_weights/train_pos-2021-07-27_21:19:57-bs=4,cd=128,ds=10000,e=12,gi=6,mdl=25,msps=60,msl=35,rd=512,rdc=256,rl=2,s=42,uc=True,ws=7.pt'
    # args.train_split = 0
    args.use_bert_enc = 'init'
    args.mode = 'taxonomy'

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
        if args.mode != 'taxonomy':
            vocab = Vocab.load(args.vocab_path)
            trainer = Trainer(args, vocab)
        else:
            trainer = Trainer.load_model(
                '/local/ccayral/orthonormalDA1/model_weights/train_pos-2021-08-03_18:16:07-bs=4,cd=128,ds=10000,e=15,fl=64,gi=6,mdl=25,msl=35,rd=512,rdc=256,rl=2,s=42,tmc=1,uc=True,ws=7.pt')
            for param in trainer.model.standardizer.parameters():
                param.requires_grad = False
            for param in trainer.model.tagger.parameters():
                param.requires_grad = False
        metrics = trainer.train()
        with open(os.path.join(args.config_save, args.logdir + '.json'), 'w') as f:
            json.dump(vars(args), f)
        with open(os.path.join(args.logs, 'metrics-' + args.logdir + '.json'), 'w') as f:
            json.dump(metrics, f)
        print(metrics)
    else:
        trainer = Trainer.load_model(args.load)
        features_tags = trainer.predict()
        predictions = []
        for seg_text in [
            seg for sent in trainer.dev_iter.dataset.src_segments_raw for token in sent for seg in token]:
            predictions.append({'text': seg_text})
        for i, pred in enumerate(predictions):
            for f in trainer.features:
                tag = getattr(trainer.vocab, f).id2word[features_tags[f][i]]
                pred[f] = tag
        write_automatic_annotations(trainer, predictions)
        pass


def error_analysis(args):
    trainer = Trainer(args)
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

def write_automatic_annotations(trainer, predictions):
    feature_values = {'aspect': ['P', 'I', 'C', 'NONE'],
                        'voice': ['A', 'P', 'NONE'],
                        'mood': ['S', 'I', 'J', 'NONE'],
                        'person': ['1', '2', '3', 'NONE'],
                        'gender': ['M', 'F', 'NONE'],
                        'number': ['S', 'D', 'P', 'NONE'],
                        'state': ['D', 'I', 'C', 'NONE'],
                        'nounForm': ['NONE'],
                        'verbForm': ["فَعَل", "فَعَّل", "فاعَل", "أفْعَل", "تَفَعَّل", "تَفاعَل", "اِنْفَعَل", "اِفْتَعَل", "اِفْعَل", "اِسْتَفْعَل", "اِفْعَال", "اِفْعَوْعَل", "اِفْعَوَّل", "فَعْلَل", "فَعْفَع", "فَعْوَعل", "فَعْفَل"]
                        }
    pos_features = {'ABBREV': ['aspect', 'voice', 'mood', 'person', 'gender', 'number', 'state', 'nounForm', 'verbForm'],
                    'ADJ': ['gender', 'number'],
                    'NOUN': ['gender', 'number', 'state', 'nounForm'],
                    'PRON': ['person', 'gender', 'number'],
                    'VERB': ['aspect', 'voice', 'mood', 'person', 'gender', 'number', 'verbForm']}

    def set_to_none(pos):
        if f not in pos_features[pos]:
            segment[f] = 'NA'
        else:
            if predictions[i][f] not in feature_values[f]:
                segment[f] = 'NONE'
            else:
                segment[f] = predictions[i][f]
    
    i = 0
    annotations_ = []
    for annotation in trainer.annotations:
        for token in annotation['segments']:
            for segment in token:
                assert preprocess(segment['text']) == preprocess(predictions[i]['text'])
                predictions[i]['nounForm'] = 'NA'
                predictions[i]['verbForm'] = 'NA'
                segment['pos'] = predictions[i]['pos']
                for f in [f for f in trainer.features if f != 'pos'] +['nounForm', 'verbForm']:
                    if predictions[i]['pos'] == 'ABBREV':
                        set_to_none('ABBREV')
                    elif predictions[i]['pos'].startswith('ADJ'):
                        set_to_none('ADJ')
                    elif predictions[i]['pos'].startswith('NOUN'):
                        set_to_none('NOUN')
                    elif predictions[i]['pos'].startswith('PRON'):
                        set_to_none('PRON')
                    elif predictions[i]['pos'].startswith('VERB'):
                        set_to_none('VERB')
                    else:
                        segment[f] = 'NA'
                i += 1
        annotations_.append(annotation)
    with open('/local/ccayral/orthonormalDA1/data/asc/annotations_carine_automatic.json', 'w') as a:
        json.dump(annotations_, a)

if __name__ == '__main__':
    main()
