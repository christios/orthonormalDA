import os
import zipfile
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re
from typing import List, TextIO, Dict, Optional
from collections import Counter
import argparse

import torch
import numpy as np
import pyarabic.araby as araby
from edit_distance import SequenceMatcher

ALEFAT = araby.ALEFAT[:5] + tuple(araby.ALEFAT[-1])
ALEFAT_PATTERN = re.compile(u"[" + u"".join(ALEFAT) + u"]", re.UNICODE)


class GumarDataset:
    """ * Loads a NMT dataset sentence per sentence.
        * The data consists of three `Dataset`s
            - TRAIN
            - DEV
            - TEST
        *  Each `Dataset` is composed of factors (SOURCE, TARGET)
    """

    train: 'GumarDataset.Dataset'
    dev: 'GumarDataset.Dataset'
    test: 'GumarDataset.Dataset'

    class Factor:
        PAD = 0
        UNK = 1
        BOW, BOS = 2, 2
        EOW, EOS = 3, 3

        """String -> word_id map"""
        words_vocab: Dict[str, int]
        """Word_id -> string list"""
        words_map: List[str]
        """Word ids of the original words (uses <unk> and <pad>)"""
        sentences_words_ids: List[np.ndarray]
        """Character -> char_id map"""
        chars_vocab: Dict[str, int]
        """Char_id -> character list"""
        chars_map: List[str]
        """Sequences of characters of the original words"""
        sentences_chars_ids: List[np.ndarray]

        def __init__(self,
                     characters: bool,
                     train: Optional['GumarDataset.Factor'] = None) -> None:

            self.words_vocab = train.words_vocab if train else {
                "<pad>": self.PAD, "<unk>": self.UNK, "<bos>": self.BOS, "<eos>": self.EOS}
            self.words_map = train.words_map if train else [
                "<pad>", "<unk>", "<bos>", "<eos>"]
            self.sentences_words_ids = []
            self.characters = characters
            # For example, if the factor represents POS, then it makes no sense to keep
            # a characer level representation of them, then this should be `False`.
            if characters:
                self.chars_vocab = train.chars_vocab if train else {
                    "<pad>": self.PAD, "<unk>": self.UNK, "<bow>": self.BOW, "<eow>": self.EOW}
                self.chars_map = train.chars_map if train else [
                    "<pad>", "<unk>", "<bow>", "<eow>"]
                self.sentences_chars_ids = []

        @property
        def sentences_words(self):
            return [[self.words_map[w] for w in s] for s in self.sentences_words_ids]

        @property
        def sentences_chars(self):
            return [[[self.chars_map[c] for c in w] for w in s] for s in self.sentences_words_ids]

    class Dataset(torch.utils.data.Dataset):
        SOURCE = 0
        TARGET = 1
        FACTORS = 2

        """Contains a factor for each column of data"""
        _data: List['GumarDataset.Factor']
        """Number of sentences in the dataset"""
        _size: int
        """Random number generator for generating random batches"""
        _shuffler: np.random.RandomState

        def __init__(self,
                     data_file: TextIO,
                     train: Optional['GumarDataset.Factor'] = None,
                     add_bow_eow: bool = False,
                     max_sentences: int = 0,
                     max_sentence_len: int = 5000,
                     name: str = '') -> None:
            self.max_sentences = max_sentences
            self.max_sentence_len = max_sentence_len
            self.name = name
            self.add_bow_eow = add_bow_eow

            self.raw_data = []
            data_xml = ET.parse(data_file).getroot()
            for idx, sentence in enumerate(data_xml):
                src = GumarDataset.preprocess(sentence[0].text)
                tgt = GumarDataset.preprocess(sentence[1].text)
                self.raw_data.append((src, tgt))

            token_fl = Counter([token for s in [pair[0]
                               for pair in self.raw_data] for token in s.split()])
            token_fl = [token[0] for token in token_fl.most_common()
                        if token[0].startswith('و')]
            delimiters = [0, 1, 3, 4, 5, 6, 9, 10, 12,
                          13, 19, 20, 21, 22, 23, 25, 26, 38, 40]
            self.delimiters = [token for i, token in enumerate(
                token_fl) if i in delimiters]
            self.punct_delim = re.compile(r'[\.:;،,!?!؟]')

            # Create factors
            self._data = []
            for f in range(self.FACTORS):
                self._data.append(GumarDataset.Factor(
                    f in [self.SOURCE, self.TARGET], train._data[f] if train else None))

            for pair in tqdm([pair for pair in self.raw_data]):
                src = [token for token in pair[0].split(' ') if token]
                pairs = [pair]
                if len(src) > max_sentence_len:
                    pairs = self.split_pair(pair)

                for pair in pairs:
                    src = [token for token in pair[0].split(' ') if token]
                    tgt = [token for token in pair[1].split(' ') if token]

                    if (max_sentence_len and len(src) > max_sentence_len) or not (src and tgt):
                        continue

                    for f in range(self.FACTORS):
                        factor = self._data[f]
                        if len(factor.sentences_words_ids):
                            factor.sentences_words_ids[-1] = np.array(
                                factor.sentences_words_ids[-1], np.int32)
                        factor.sentences_words_ids.append([])
                        if add_bow_eow:
                            factor.sentences_words_ids[-1].append(
                                GumarDataset.Factor.BOS)
                        if factor.characters:
                            factor.sentences_chars_ids.append([])

                        sentence = tgt if f else src
                        # Word-level information
                        for word in sentence:
                            if word not in factor.words_vocab:
                                if train:
                                    word = "<unk>"
                                else:
                                    factor.words_vocab[word] = len(
                                        factor.words_map)
                                    factor.words_map.append(word)
                            factor.sentences_words_ids[-1].append(
                                factor.words_vocab[word])

                            # Character-level information
                            if factor.characters:
                                factor.sentences_chars_ids[-1].append([])
                                if add_bow_eow:
                                    factor.sentences_chars_ids[-1][-1].append(
                                        GumarDataset.Factor.BOW)
                                for c in word:
                                    if c not in factor.chars_vocab:
                                        if train:
                                            c = "<unk>"
                                        else:
                                            factor.chars_vocab[c] = len(
                                                factor.chars_map)
                                            factor.chars_map.append(c)
                                    factor.sentences_chars_ids[-1][-1].append(
                                        factor.chars_vocab[c])
                                if add_bow_eow:
                                    factor.sentences_chars_ids[-1][-1].append(
                                        GumarDataset.Factor.EOW)
                        if add_bow_eow:
                            factor.sentences_words_ids[-1].append(
                                GumarDataset.Factor.EOS)
                    if max_sentences and len(self._data[self.SOURCE].sentences_words_ids) >= max_sentences:
                        break

            self._size = len(self._data[0].sentences_words_ids)

        def split_pair(self, pair):
            data_ = []
            split_src = [s.strip() for s in self.punct_delim.split(pair[0])]
            split_tgt = [s.strip() for s in self.punct_delim.split(pair[1])]
            if len(split_src) == len(split_tgt):
                for pair in zip(split_src, split_tgt):
                    if len(pair[0].split()) == 1 and self.punct_delim.search(pair[0]) or not pair[0]:
                        continue
                    if len(pair[0].split()) <= self.max_sentence_len:
                        data_.append(pair)
                    else:
                        src = [[token, 'n'] for token in pair[0].split()]
                        tgt = [[token, 'n'] for token in pair[1].split()]
                        src, tgt = GumarDataset.align(src, tgt)
                        cut = 0
                        split_indexes = [0]
                        for i, token in enumerate(zip(src, tgt)):
                            cut += 1
                            if cut >= 2:
                                if token[0][0].startswith('و') \
                                        and token[0][0] in self.delimiters \
                                        and token[0][1] in 'se':
                                    split_indexes.append(i)
                                    cut = 0
                        split_indexes.append(len(src))
                        src = [src[start:end] for start, end in zip(
                            split_indexes, split_indexes[1:])]
                        tgt = [tgt[start:end] for start, end in zip(
                            split_indexes, split_indexes[1:])]
                        src = [
                            ' '.join([token[0] for token in src_ if token[1] in 'ies']) for src_ in src]
                        tgt = [
                            ' '.join([token[0] for token in tgt_ if token[1] in 'des']) for tgt_ in tgt]
                        assert len(src) == len(tgt), 'Wrong split'
                        for pair_ in zip(src, tgt):
                            data_.append(pair_)
            return data_

        @property
        def size(self):
            return self._size

        def get_token_num(self):
            return sum(len(pair[0]) for pair in self._data), \
                sum(len(pair[0]) for pair in self._data)

    @staticmethod
    def align(src, tgt):
        """Corrects misalignments between the gold and predicted tokens
        which will almost almost always have different lengths due to inserted, 
        deleted, or substituted tookens in the predicted systme output."""

        sm = SequenceMatcher(
            a=list(map(lambda x: x[0], tgt)), b=list(map(lambda x: x[0], src)))
        tgt_temp, src_temp = [], []
        opcodes = sm.get_opcodes()
        for tag, i1, i2, j1, j2 in opcodes:
            # If they are equal, do nothing except lowercase them
            if tag == 'equal':
                for i in range(i1, i2):
                    tgt[i][1] = 'e'
                    tgt_temp.append(tgt[i])
                for i in range(j1, j2):
                    src[i][1] = 'e'
                    src_temp.append(src[i])
            # For insertions and deletions, put a filler of '***' on the other one, and
            # make the other all caps
            elif tag == 'delete':
                for i in range(i1, i2):
                    tgt[i][1] = 'd'
                    tgt_temp.append(tgt[i])
                for i in range(i1, i2):
                    src_temp.append(tgt[i])
            elif tag == 'insert':
                for i in range(j1, j2):
                    src[i][1] = 'i'
                    tgt_temp.append(src[i])
                for i in range(j1, j2):
                    src_temp.append(src[i])
            # More complicated logic for a substitution
            elif tag == 'replace':
                for i in range(i1, i2):
                    tgt[i][1] = 's'
                for i in range(j1, j2):
                    src[i][1] = 's'
                tgt_temp += tgt[i1:i2]
                src_temp += src[j1:j2]

        src, tgt = src_temp, tgt_temp
        return src, tgt

    @staticmethod
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
        sentence = re.sub(r'[^\d\w]', r' ', sentence)
        sentence = re.sub(r'( ){2,}', r'\1', sentence)
        return sentence

    def __init__(self,
                 dataset: str = 'annotated-gumar-corpus',
                 max_sentences: int = 0,
                 max_sentence_len: int = 5000):

        path = "data/{}.zip".format(dataset)
        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["TRAIN", "DEV", "TEST"]:
                with zip_file.open(f"{os.path.splitext(path.split('/')[1])[0]}/{dataset}_annotated_Gumar_corpus.xml", "r") as dataset_file:
                    setattr(self, dataset.lower(), self.Dataset(dataset_file,
                                                                max_sentences=max_sentences,
                                                                max_sentence_len=max_sentence_len))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_src", type=str,
                        help="Path of the training source corpus.")
    parser.add_argument("--train_tgt", type=str,
                        help="Path of the training target corpus.")
    parser.add_argument("--vocab_size", default=50000, type=int,
                        help="Maximum size of the vocabulary.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    gumar = GumarDataset(max_sentence_len=30)
    # save_dir = '/Users/chriscay/thesis/spelling-correction/data/hybrid'
    # with open(f'{save_dir}/gumar_src_train.txt', 'w') as src_train, open(f'{save_dir}/gumar_tgt_train.txt', 'w') as tgt_train, \
    #         open(f'{save_dir}/gumar_src_dev.txt', 'w') as src_dev, open(f'{save_dir}/gumar_tgt_dev.txt', 'w') as tgt_dev, \
    #         open(f'{save_dir}/gumar_src_test.txt', 'w') as src_test, open(f'{save_dir}/gumar_tgt_test.txt', 'w') as tgt_test:
    #     print('\n'.join([pair[0] for pair in gumar.train.data]), file=src_train)
    #     print('\n'.join([pair[1] for pair in gumar.train.data]), file=tgt_train)
    #     print('\n'.join([pair[0] for pair in gumar.dev.data]), file=src_dev)
    #     print('\n'.join([pair[1] for pair in gumar.dev.data]), file=tgt_dev)
    #     print('\n'.join([pair[0] for pair in gumar.test.data]), file=src_test)
    #     print('\n'.join([pair[1] for pair in gumar.test.data]), file=tgt_test)
    print(''.join(sorted(''.join([c[0] for c in Counter([char for pair in gumar.train.data for char in pair[0]
                                                         ] + [char for pair in gumar.train.data for char in pair[1]]).most_common()]))))

    print('read in source sentences: %s' % args.train_src)
    print('read in target sentences: %s' % args.train_tgt)

    src_sents = read_corpus(args.train_src, source='src')
    tgt_sents = read_corpus(args.train_tgt, source='tgt')

    vocab = Vocab.build(src_sents[1], tgt_sents[1], int(
        args['--size']), int(args['--freq-cutoff']))
    print('generated vocabulary, source %d words, target %d words' %
          (len(vocab.src), len(vocab.tgt)))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])
