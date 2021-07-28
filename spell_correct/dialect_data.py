import re
import regex
import os
from collections import Counter
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast
from torch import Tensor

from spell_correct.utils import pad_sents_char, create_padded_segment_contexts
from spell_correct.utils import AlignmentHandler
from spell_correct.vocab import VocabEntry

import pyarabic.araby as araby
ALEFAT = araby.ALEFAT[:5] + tuple(araby.ALEFAT[-1])
ALEFAT_PATTERN = re.compile(u"[" + u"".join(ALEFAT) + u"]", re.UNICODE)

class DialectData(Dataset):
    def __init__(self, args, data, features, vocab, device) -> None:
        self.vocab = vocab
        self.device = device
        self.args = args
        self.bert_tokenizer = None
        self.features = features
        if args.use_bert_enc:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(
                args.bert_model, cache_dir=args.bert_cache_dir)
        
        i = 0
        self.mode = args.mode
        self.src_raw = [f[i] for f in data]
        i += 1
        self.tgt_raw = [f[i] for f in data]
        i += 1
        self.src_segments_raw = [f[i] for f in data]
        self.segments_per_token = [
            [len(token) for token in sent] for sent in self.src_segments_raw]
        i += 1
        self.src_char = [f[i] for f in data]
        self.src_char = pad_sents_char(self.src_char,
                                       self.vocab.src.char2id['<pad>'],
                                       max_sent_length=args.max_sent_len,
                                       max_word_length=args.max_decode_len)
        i += 1
        self.tgt_char = [f[i] for f in data]
        self.tgt_char = pad_sents_char(self.tgt_char,
                                       self.vocab.tgt.char2id['<pad>'],
                                       max_sent_length=args.max_sent_len,
                                       max_word_length=args.max_decode_len)
        i += 1
        assert len(self.src_raw) == len(self.tgt_raw) == len(self.src_char) == len(self.tgt_char)
        if args.use_bert_enc:
            self.src_bert = [f[i] for f in data]
            i += 1
            self.src_bert_mask = [f[i] for f in data]
            i += 1
            assert len(self.src_raw) == len(self.src_bert) == len(self.src_bert_mask)

        if args.mode == 'pos_tagger':
            self.src_segments = [f[i] for f in data]
            self.src_segments = create_padded_segment_contexts(self.src_segments,
                                                               self.vocab.src.char2id['<pad>'],
                                                               self.vocab.src.char2id['<b>'],
                                                               max_sent_length=args.max_sent_len,
                                                               max_seg_per_token=4,
                                                               max_seg_length=15,
                                                               window_size=args.window_size)
            i += 1
            assert len(self.tgt_char) == len(self.src_segments)
            for feature in features:
                setattr(self, feature, [f[i] for f in data])
                setattr(self, feature, create_padded_segment_contexts(getattr(self, feature),
                                                                          self.vocab.tgt.word2id['<pad>'],
                                                                          self.vocab.src.char2id['<b>'],
                                                                          max_sent_length=args.max_sent_len,
                                                                          max_seg_per_token=4,
                                                                          window_size=args.window_size))
                i += 1
                assert len(self.tgt_char) == len(getattr(self, feature))

    def __getitem__(self, index):
        src_bert = getattr(self, 'src_bert', None)
        src_bert_mask = getattr(self, 'src_bert_mask', None)
        lengths_word = getattr(self, 'lengths_word', None)
        if src_bert:
            src_bert = src_bert[index]
            src_bert_mask = src_bert_mask[index]
        inputs = dict(src_raw=self.src_raw[index],
                      src_char=self.src_char[index],
                      lengths_word=lengths_word,
                      src_bert=src_bert,
                      src_bert_mask=src_bert_mask,
                      tgt_raw=self.tgt_raw[index],
                      tgt_char=self.tgt_char[index],
                      src_segments=self.src_segments[index],
                      segments_per_token=self.segments_per_token[index])
        for feature in self.features:
            inputs[feature] = getattr(self, feature)[index]
        return inputs

    def __len__(self):
        return len(self.src_char)

    def generate_batch(self, data_batch):
        src_raw_batch, tgt_raw_batch = [], []
        src_char_batch, src_segments_batch = [], []
        src_bert_batch, src_bert_mask_batch = [], []
        tgt_char_batch = []
        segments_per_token_batch = []
        features_labels_batch = {feature: [] for feature in self.features}
        for inputs in data_batch:
            src_raw_batch.append(inputs['src_raw'])
            src_char_batch.append(inputs['src_char'])
            tgt_raw_batch.append(inputs['tgt_raw'])
            tgt_char_batch.append(inputs['tgt_char'])
            src_segments_batch.append(inputs['src_segments'])
            segments_per_token_batch.append(inputs['segments_per_token'])
            for feature in self.features:
                features_labels_batch[feature].append(inputs[feature])
            if inputs['src_bert']:
                src_bert_batch.append(inputs['src_bert'])
                src_bert_mask_batch.append(inputs['src_bert_mask'])

        src_char_batch = torch.tensor(src_char_batch, dtype=torch.long).permute(
            1, 0, 2).to(self.device)
        tgt_char_batch = torch.tensor(tgt_char_batch, dtype=torch.long).permute(
            1, 0, 2).to(self.device)
        for feature in self.features:
            features_labels_batch[feature] = torch.tensor(
                features_labels_batch[feature], dtype=torch.long).to(self.device)
        src_segments_batch = torch.tensor(src_segments_batch, dtype=torch.long).to(self.device)
        if src_bert_batch:
            src_bert_batch = torch.tensor(
                src_bert_batch, dtype=torch.long).to(self.device)
            src_bert_mask_batch = torch.tensor(
                src_bert_mask_batch, dtype=torch.long).to(self.device)

        batch = dict(src_raw=src_raw_batch,
                     src_char=src_char_batch,
                     src_segments=src_segments_batch,
                     segments_per_token=segments_per_token_batch,
                     src_bert=src_bert_batch if isinstance(src_bert_batch, Tensor) else None,
                     src_bert_mask=src_bert_mask_batch if isinstance(src_bert_mask_batch, Tensor) else None,
                     tgt_raw=tgt_raw_batch,
                     tgt_char=tgt_char_batch,
                     features_labels=features_labels_batch)

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

def load_data(args, vocab, device, load=False):
    asc, annotations = read_asc(args)

    char_ids_src = vocab.src.words2charindices(asc['src'], add_beg_end=False)
    char_ids_tgt = vocab.tgt.words2charindices(asc['tgt'])
    src_char = char_ids_src[:args.data_size]
    tgt_char = char_ids_tgt[:args.data_size]
    char_ids_src_segments = vocab.src.words2charindices(
        asc['src_segments'], segments=True)
    if not load:
        for feature_name, feature in asc['features'].items():
            feature_vocab = VocabEntry(VocabEntry.build_feature_vocab(feature))
            setattr(vocab, feature_name, feature_vocab)
    features_ids_labels = {}
    for feature_name, feature in asc['features'].items():
        features_ids_labels[feature_name] = getattr(vocab, feature_name).pos2indices(feature)

    src_segments_char = char_ids_src_segments[:args.data_size]
    features_labels = {feature_name: feature[:args.data_size]
                       for feature_name, feature in features_ids_labels.items()}

    src_bert, src_bert_mask = None, None
    if args.use_bert_enc:
        bert_tokenizer = BertTokenizerFast.from_pretrained(
            args.bert_model, cache_dir=args.bert_cache_dir)
        src_bert = bert_tokenizer([' '.join(sent) for sent in asc['src']],
                                       padding="max_length",
                                       truncation=True,
                                       max_length=args.max_sent_len)
        src_bert, src_bert_mask = src_bert.input_ids, src_bert.attention_mask
        src_bert = src_bert[:args.data_size]
        src_bert_mask = src_bert_mask[:args.data_size]

    data = [x for x in [asc['src_raw'], asc['tgt_raw'], asc['src_segments'], src_char, tgt_char,
                        src_bert, src_bert_mask, src_segments_char, *features_labels.values()] if x]
    data = list(zip(*data))

    lengths = [int(len(src_char)*args.train_split),
               int(len(src_char)*(1-args.train_split))]
    if sum(lengths) != len(src_char):
        lengths[0] += len(src_char) - sum(lengths)
    if not load:
        train_data, dev_data = random_split(data, lengths)
    else:
         train_data, dev_data = [], data
    features = list(features_labels.keys())
    train_data = DialectData(args, train_data, features, vocab, device)
    dev_data = DialectData(args, dev_data, features, vocab, device)

    train_iter = None
    if len(train_data) > 0:
        train_iter = DataLoader(train_data, batch_size=args.batch_size,
                                shuffle=True, collate_fn=train_data.generate_batch)
    dev_iter = DataLoader(dev_data, batch_size=len(dev_data),
                          collate_fn=dev_data.generate_batch)
    return train_iter, dev_iter, annotations


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


def read_asc(args, get_untagged=False):
    data = []
    for file in os.listdir(args.data):
        if ('carine' in file or not get_untagged) and 'automatic' not in file:
        # if 'automatic' in file:
            with open(os.path.join(args.data, file)) as f:
                data += json.load(f)

    src, src_raw, tgt, tgt_raw = [], [], [], []
    src_segments = []
    features = {feature: [] for feature in args.features.split()}

    data_ = []
    for idx, d in enumerate(data):
        if [True for token in d['segments'] if token[0]['pos'] == 'NONE']:
            if not get_untagged:
                continue
        elif get_untagged:
            continue
        data_.append(d)
        src_segments.append([])
        for feature in features.values():
            feature.append([])
        for token in d['segments']:
            src_segments[-1].append([])
            for feature in features.values():
                feature[-1].append([])
            for segment in token:
                src_segments[-1][-1].append(preprocess(segment['text']))
                for feature_name, feature in features.items():
                    feature[-1][-1].append(segment[feature_name])
        src.append([preprocess_for_sos(raw_token) for raw_token in d['raw']])
        tgt.append([preprocess_for_sos(coda_token) for coda_token in d['coda']])
        assert len(src[-1]) == len(tgt[-1]) == len(src_segments[-1])
        assert sum(len(feature[-1]) for feature in features.values()
                   ) / len(features) == len(src[-1])
        src_raw.append(preprocess(' '.join(d['raw'])))
        tgt_raw.append(preprocess(' '.join(d['coda'])))

    asc = dict(src=src,
               src_raw=src_raw,
               tgt=tgt,
               tgt_raw=tgt_raw,
               src_segments=src_segments,
               features=features)
    return asc, data_

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
    sentence = re.sub(r'_', '', sentence)
    return sentence


def preprocess_for_sos(sentence):
    sentence = araby.strip_tatweel(sentence)
    sentence = araby.strip_tashkeel(sentence)
    sentence = re.sub(r',', r'،', sentence)
    sentence = re.sub(r'\?', r'؟', sentence)
    sentence = re.sub(r'_', '', sentence)
    sentence = re.sub(r'ﺁ', r'آ', sentence)
    # Merge step
    sentence = re.sub(r' ', r'', sentence)
    return sentence
