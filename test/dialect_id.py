import argparse
import time
import re
import os
import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch.optim as optim

from spell_correct.spelling_corrector import SpellingCorrector
from spell_correct.dialect_data import load_data, process_raw_inputs


class DialectIDData(Dataset):
    def __init__(self, args, data, device) -> None:
        self.device = device
        self.args = args

        self.bert_tokenizer = BertTokenizerFast.from_pretrained(
            args.bert_model, cache_dir=args.bert_cache_dir)

        self.src_bert = [f[0] for f in data]
        self.src_bert_mask = [f[1] for f in data]
        self.labels = [f[2] for f in data]

        assert len(self.src_bert) == len(self.src_bert_mask) == len(self.labels), 'Error in data compilation'

    def __getitem__(self, index):
        inputs = dict(src_bert=self.src_bert[index],
                      src_bert_mask=self.src_bert_mask[index],
                      labels=self.labels[index])
        return inputs

    def __len__(self):
        return len(self.src_bert)

    def generate_batch(self, data_batch):
        src_bert_batch, src_bert_mask_batch = [], []
        labels_batch = []
        for inputs in data_batch:
            src_bert_batch.append(inputs['src_bert'])
            src_bert_mask_batch.append(inputs['src_bert_mask'])
            labels_batch.append(inputs['labels'])

        src_bert_batch = torch.tensor(
            src_bert_batch, dtype=torch.long).to(self.device)
        src_bert_mask_batch = torch.tensor(
            src_bert_mask_batch, dtype=torch.long).to(self.device)
        labels_batch = torch.tensor(
            labels_batch, dtype=torch.long).to(self.device)

        batch = dict(src_bert=src_bert_batch,
                     src_bert_mask=src_bert_mask_batch,
                     labels=labels_batch)
        return batch


class DialectIDTrainer:
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(
            f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')

        self.load_data()
        # self.model = DialectIdentifier(args.bert_model).to(self.device)
        bert_model = args.bert_model if not args.load else args.load
        self.model = BertForSequenceClassification.from_pretrained(bert_model).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=1, factor=0.5)

    def load_data(self):
        bert_tokenizer = BertTokenizerFast.from_pretrained(
            self.args.bert_model, cache_dir=self.args.bert_cache_dir)

        sentences, labels = [], []
        with open(self.args.data_path) as f:
            for line in f:
                line = line.split(' ||| ')
                sentences.append(line[0].strip())
                labels.append(0 if line[1].strip() == 'MSA' else 1)

        src_bert = bert_tokenizer(sentences,
                                padding="max_length",
                                truncation=True,
                                max_length=70)
        src_bert, src_bert_mask = src_bert.input_ids, src_bert.attention_mask
        src_bert = src_bert[:self.args.data_size]
        src_bert_mask = src_bert_mask[:self.args.data_size]
        labels = labels[:self.args.data_size]

        data = list(zip(src_bert, src_bert_mask, labels))

        lengths = [int(len(src_bert)*self.args.train_split),
                   int(len(src_bert)*(1-self.args.train_split))]
        if sum(lengths) != len(src_bert):
            lengths[0] += len(src_bert) - sum(lengths)
        train_data, dev_data = random_split(data, lengths)

        train_data = DialectIDData(self.args, train_data, self.device)
        dev_data = DialectIDData(self.args, dev_data, self.device)

        self.train_iter = DataLoader(train_data, batch_size=self.args.batch_size,
                                shuffle=True, collate_fn=train_data.generate_batch)
        self.dev_iter = DataLoader(dev_data, batch_size=len(dev_data),
                            collate_fn=dev_data.generate_batch)

    @staticmethod
    def epoch_time(start_time: int,
                   end_time: int):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _compute_loss_acc(self, outputs, labels):
        labels = labels.view(-1)
        correct = torch.sum(labels == outputs.logits.argmax(-1))
        total = outputs.logits.shape[0]
        return outputs.loss, (correct, total)


    def train(self):
        metrics_train, metrics_val = {}, {}
        for epoch in range(self.args.epochs):
            self.model.train()
            print(f'Epoch {epoch+1}/{self.args.epochs}')
            epoch_loss = 0
            start_time = time.time()
            for iteration, batch in enumerate(self.train_iter):
                labels = batch['labels']

                self.model.zero_grad()
                outputs = self.model(batch['src_bert'], batch['src_bert_mask'], labels=labels)
                loss, acc = self._compute_loss_acc(outputs, labels)
                loss.backward()
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
            epoch_mins, epoch_secs = DialectIDTrainer.epoch_time(
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
            epoch_loss = 0
            correct, total = 0, 0
            for batch in self.dev_iter:
                # Loss
                outputs = self.model(batch['src_bert'], batch['src_bert_mask'], labels=batch['labels'])
                loss, acc = self._compute_loss_acc(outputs, batch['labels'])
                correct += acc[0]
                total += acc[1]
                epoch_loss += loss.item()
                # Accuracy

        metrics = {}
        metrics['dev_acc'] = correct / total
        metrics['dev_loss'] = epoch_loss / len(self.dev_iter)
        return metrics

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(self.dev_iter))
            outputs = self.model(batch['src_bert'], batch['src_bert_mask'])
            outputs = nn.functional.softmax(outputs.logits, dim=-1)
            outputs = [(self.dev_iter.dataset.bert_tokenizer.decode(batch['src_bert'][i], skip_special_tokens=True), score[1].item())
                        for i, score in enumerate(outputs) if score[1] > score[0]]
        return outputs

    def save_model(self):
        save_path = os.path.join(self.args.cpt, self.args.logdir) + '.pt'
        self.model.save_pretrained(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32,
                        type=int, help="Batch size.")
    parser.add_argument("--epochs", default=2, type=int,
                        help="Number of epochs.")
    parser.add_argument("--data_size", default=100000, type=int,
                        help="Maximum number of examples to load.")
    parser.add_argument("--train_split", default=0.9, type=float,
                        help="Proportion with which to split the train and dev data.")
    parser.add_argument("--dropout", default=0.2, type=float,
                        help="Probablility of dropout for encoder and decoder.")
    parser.add_argument("--gpu_index", default=6, type=int,
                        help="Index of GPU to be used.")
    parser.add_argument("--data_path",
                        default="/local/ccayral/orthonormalDA1/data/dialect-id/dialect-id-data.txt", type=str,
                        help="Path to dataset file.")
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.logdir = "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook").split('.')[0]),
        datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if isinstance(value, int)))
    )

    args.load = '/local/ccayral/orthonormalDA1/model_weights/dialect_id-2021-05-28_19:28:12-bs=32,ds=100000,e=2,gi=6,s=42.pt'
    if not args.load:
        trainer = DialectIDTrainer(args)
        metrics = trainer.train()
        with open(os.path.join(args.config_save, args.logdir + '.json'), 'w') as f:
            json.dump(vars(args), f)
        with open(os.path.join(args.logs, 'metrics-' + args.logdir + '.json'), 'w') as f:
            json.dump(metrics, f)
        print(metrics)
    else:
        trainer = DialectIDTrainer(args)
        predictions = trainer.predict()
        pass


if __name__ == '__main__':
    main()
