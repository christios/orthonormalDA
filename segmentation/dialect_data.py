import re
import regex
import os
from collections import Counter
import json

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast
from torch import Tensor

import pyarabic.araby as araby
ALEFAT = araby.ALEFAT[:5] + tuple(araby.ALEFAT[-1])
ALEFAT_PATTERN = re.compile(u"[" + u"".join(ALEFAT) + u"]", re.UNICODE)

from segmentation.utils import pad_sents_char, pad_sents


class DialectData(Dataset):
    def __init__(self, args, data, vocab, device) -> None:
        self.vocab = vocab
        self.device = device
        self.args = args
        self.bert_tokenizer = None
        if args.use_bert_enc:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(
                args.bert_model, cache_dir=args.bert_cache_dir)
        
        self.src_raw = [f[0] for f in data]
        self.tgt_raw = [f[1] for f in data]
        self.src_char = [f[2] for f in data]
        self.src_char = pad_sents_char(self.src_char,
                                       self.vocab.src.char2id['<pad>'],
                                       max_sent_length=args.max_sent_len,
                                       max_word_length=args.max_word_len)
        self.tgt_char = [f[3] for f in data]
        self.tgt_char = pad_sents_char(self.tgt_char,
                                       self.vocab.src.char2id['<pad>'],
                                       max_sent_length=args.max_sent_len,
                                       max_word_length=args.max_word_len)
        if args.use_bert_enc:
            self.src_bert = [f[4] for f in data]
            self.src_bert_mask = [f[5] for f in data]

        assert len(self.src_char) == len(self.tgt_char) \
            == len(self.src_bert) if args.use_bert_enc else len(self.src_char) \
            == len(self.src_bert_mask) if args.use_bert_enc else len(self.src_char), 'Error in data compilation'

    def __getitem__(self, index):
        src_bert = getattr(self, 'src_bert', None)
        src_bert_mask = getattr(self, 'src_bert_mask', None)
        if src_bert:
            src_bert = src_bert[index]
            src_bert_mask = src_bert_mask[index]
        inputs = dict(src_raw=self.src_raw[index],
                      src_char=self.src_char[index],
                      src_bert=src_bert,
                      src_bert_mask=src_bert_mask,
                      tgt_raw=self.tgt_raw[index],
                      tgt_char=self.tgt_char[index],)
        return inputs

    def __len__(self):
        return len(self.src_char)

    def generate_batch(self, data_batch):
        src_raw_batch, tgt_raw_batch = [], []
        src_char_batch = []
        src_bert_batch, src_bert_mask_batch = [], []
        tgt_char_batch = []
        for inputs in data_batch:
            src_raw_batch.append(inputs['src_raw'])
            src_char_batch.append(inputs['src_char'])
            tgt_raw_batch.append(inputs['tgt_raw'])
            tgt_char_batch.append(inputs['tgt_char'])
            if inputs['src_bert']:
                src_bert_batch.append(inputs['src_bert'])
                src_bert_mask_batch.append(inputs['src_bert_mask'])

        src_char_batch = torch.tensor(src_char_batch, dtype=torch.long).permute(
            1, 0, 2).to(self.device)
        tgt_char_batch = torch.tensor(tgt_char_batch, dtype=torch.long).permute(
            1, 0, 2).to(self.device)
        if src_bert_batch:
            src_bert_batch = torch.tensor(
                src_bert_batch, dtype=torch.long).to(self.device)
            src_bert_mask_batch = torch.tensor(
                src_bert_mask_batch, dtype=torch.long).to(self.device)

        batch = dict(src_raw=src_raw_batch,
                    src_char=src_char_batch,
                    src_bert=src_bert_batch if isinstance(src_bert_batch, Tensor) else None,
                    src_bert_mask=src_bert_mask_batch if isinstance(src_bert_mask_batch, Tensor) else None,
                    tgt_raw=tgt_raw_batch,
                    tgt_char=tgt_char_batch)

        return batch


def split_into_src_tgt():
    with open('/local/ccayral/orthonormalDA/data/coda-corpus/dialects/Cairo.tsv') as f, \
            open('/local/ccayral/orthonormalDA/data/coda-corpus/cairo_src.txt', 'w') as f_r, \
            open('/local/ccayral/orthonormalDA/data/coda-corpus/cairo_tgt.txt', 'w') as f_c:
        for line in f:
            line = line.split('\t')
            if not line[0].isnumeric():
                continue
            raw, coda = line[3].strip(), line[4].strip()
            raw = regex.sub(r'[^\P{P}،]+', '', raw)
            coda = regex.sub(r'[^\P{P}،]+', '', coda)
            raw = re.sub(r'(\b)(\w+)(\b)', r'\1 \2 \3', raw)
            raw = re.sub(r'( ){2,}', r'\1', raw)
            coda = re.sub(r'(\b)(\w+)(\b)', r'\1 \2 \3', coda)
            coda = re.sub(r'( ){2,}', r'\1', coda)
            raw = re.split(r'،', raw)
            coda = re.split(r'،', coda)
            assert len(raw) == len(coda), 'Wrong splitting of sentences'
            for r, c in zip(raw, coda):
                if r not in ['', ' '] and c not in ['', ' ']:
                    print(r.strip(), file=f_r)
                    print(c.strip(), file=f_c)


def generate_char_dict():
    char_dict = Counter()
    for file in os.scandir('/local/ccayral/orthonormalDA/data/coda-corpus/dialects'):
        with open(file) as f:
            for line in f:
                line = line.split('\t')
                if not line[0].isnumeric():
                    continue
                raw, coda = line[3].strip(), line[4].strip()
                char_dict.update(raw)
                char_dict.update(coda)

def load_data(args, vocab, device):
    if args.use_bert_enc:
        bert_tokenizer = BertTokenizerFast.from_pretrained(
            args.bert_model, cache_dir=args.bert_cache_dir)
    
    src, src_raw, tgt, tgt_raw = read_asc(path=args.data)

    char_ids_src = vocab.src.words2charindices(src, add_beg_end=False)
    char_ids_tgt = [[[0 if c == '0' else 1 for c in w] for w in s] for s in tgt]
    src_char = char_ids_src[:args.data_size]
    tgt_char = char_ids_tgt[:args.data_size]

    src_bert, src_bert_mask = None, None
    if args.use_bert_enc:
        src_bert = bert_tokenizer([re.sub(r'_', '', ' '.join(sent)) for sent in src],
                                  padding="max_length",
                                  truncation=True,
                                  max_length=args.max_sent_len)
        src_bert, src_bert_mask = src_bert.input_ids, src_bert.attention_mask
        src_bert = src_bert[:args.data_size]
        src_bert_mask = src_bert_mask[:args.data_size]

    data = [x for x in [src_raw, tgt_raw, src_char, tgt_char,
                        src_bert, src_bert_mask] if x]
    data = list(zip(*data))

    lengths = [int(len(src_char)*args.train_split),
                int(len(src_char)*(1-args.train_split))]
    if sum(lengths) != len(src_char):
        lengths[0] += len(src_char) - sum(lengths)
    train_data, dev_data = random_split(data, lengths)

    train_data = DialectData(args, train_data, vocab, device)
    dev_data = DialectData(args, dev_data, vocab, device)

    train_iter = DataLoader(train_data, batch_size=args.batch_size,
                                shuffle=True, collate_fn=train_data.generate_batch)
    dev_iter = DataLoader(dev_data, batch_size=len(dev_data),
                            collate_fn=dev_data.generate_batch)
    return train_iter, dev_iter

def read_asc(path):
    data = []
    for file in os.listdir(path):
        with open(os.path.join(path, file)) as f:
            data += json.load(f)
    # `c` counts the number of sentences discarded and `broke` is to monitor
    # which sentences should be discarded due to malformation
    c, broke = 0, False
    src, src_raw, tgt, tgt_raw = [], [], [], []
    for idx, d in enumerate(data):
        src.append([])
        tgt.append([])
        for token_raw, token_morpho in zip(d['raw'], d['segments']):
            src[-1] += [preprocess(t) for t in token_raw.split()]
            token_morpho = ''.join(
                [segment['text'] + '§' for segment in token_morpho])[:-1]
            token_morpho = [c for c in token_morpho]
            segments = []
            for i in range(len(token_raw)):
                if token_raw[i] == token_morpho[i]:
                    if token_raw[i] == token_morpho[i] == ' ':
                        segments.append('1')
                    segments.append('0')
                elif token_raw[i] == ' ' and token_morpho[i] == '§':
                    segments.append(' ')
                elif token_raw[i] != ' ' and token_morpho[i] == '§':
                    segments[-1] = '1'
                    del token_morpho[i]
                    assert token_raw[i] == token_morpho[i], 'Error'
                    segments.append('0')
                else:
                    c += 1
                    broke = True
                    break
            if broke:
                break
            tgt[-1] += ''.join(segments).split()
        if broke:
            del src[-1]
            del tgt[-1]
            broke = False

        src_raw.append(preprocess(' '.join(d['raw'])))
        tgt_raw.append(' '.join(tgt[-1]))
    # Remove malformed sentences by hand (there are very few)
    indexes_to_remove = [279, 318]
    src = [s for idx, s in enumerate(src) if idx not in indexes_to_remove]
    tgt = [t for idx, t in enumerate(tgt) if idx not in indexes_to_remove]
    src_raw = [s for idx, s in enumerate(src_raw) if idx not in indexes_to_remove]
    tgt_raw = [t for idx, t in enumerate(tgt_raw) if idx not in indexes_to_remove]
    # Sanity check
    assert len(src) == len(tgt), 'Number of sentences is not equal.'
    src_, tgt_, src_raw_, tgt_raw_ = [], [], [], []
    c = len(src)
    for idx, (src_sent, tgt_sent) in enumerate(zip(src, tgt)):
        if len(src_sent) != len(tgt_sent):
            continue
        well_formed = False
        for src_token, tgt_token in zip(src_sent, tgt_sent):
            if len(src_token) != len(tgt_token):
                break
            if ' ' in src_token:
                break
            if tgt_token[-1] == '1':
                break
            if not (len(tgt_token) == 1 and tgt_token[0] != '1' or len(tgt_token) != 1):
                break
            well_formed = True
        if well_formed:
            c -= 1
            src_.append(src_sent)
            tgt_.append(tgt_sent)
            src_raw_.append(src_raw[idx])
            tgt_raw_.append(tgt_raw[idx])
    src, tgt, src_raw, tgt_raw = src_, tgt_, src_raw_, tgt_raw_
    tgt_raw = [' '.join(sent) for sent in tgt]
    return src, src_raw, tgt, tgt_raw


def preprocess(sentence):
    sentence = araby.strip_tatweel(sentence)
    sentence = sentence.replace(
        araby.SMALL_ALEF+araby.ALEF_MAKSURA, araby.ALEF_MAKSURA)
    sentence = sentence.replace(
        araby.ALEF_MAKSURA+araby.SMALL_ALEF, araby.ALEF_MAKSURA)
    sentence = re.sub(ALEFAT_PATTERN, araby.ALEF, sentence)
    sentence = araby.normalize_ligature(sentence)
    sentence = araby.normalize_teh(sentence)
    sentence = araby.strip_tashkeel(sentence)
    sentence = re.sub(r',', r'،', sentence)
    sentence = re.sub(r'\?', r'؟', sentence)
    return sentence


def process_raw_inputs(args, vocab, raw_inputs, device):
        if args.use_bert_enc:
            bert_tokenizer = BertTokenizerFast.from_pretrained(
                args.bert_model, cache_dir=args.bert_cache_dir)
        src = vocab.src.words2charindices(
            [[sent[0]] for sent in raw_inputs])
        tgt = vocab.src.words2charindices(
            [[sent[1]] for sent in raw_inputs])

        src_bert = bert_tokenizer(raw_inputs,
                                  padding="max_length",
                                  truncation=True,
                                  max_length=args.max_sent_len)
        src_bert, src_bert_mask = src_bert.input_ids, src_bert.attention_mask

        data = list(zip(src, src_bert, src_bert_mask, tgt))
        data = DialectData(args, data, vocab, device)

        return DataLoader(data, batch_size=len(data),
                          collate_fn=data.generate_batch)
