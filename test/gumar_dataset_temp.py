import os
import zipfile
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re
from typing import Dict, List, Optional, TextIO
from math import inf
from collections import Counter
import pickle

import numpy as np
import pyarabic.araby as araby
from edit_distance import SequenceMatcher
from nltk import edit_distance, masi_distance

from torch import zeros, long, tensor

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
        BOW, BOS = 1, 1
        EOW, EOS = 2, 2

        """String -> word_id map"""
        words_vocab: Dict[str, int]
        """Word_id -> string list"""
        words_map: List[str]
        """Word ids of the original words"""
        sentences_words_ids: List[np.ndarray]
        """Strings of the original words split on the word level"""
        sentences_words: List[List[str]]
        """Character -> char_id map"""
        chars_vocab: Dict[str, int]
        """Char_id -> character list"""
        chars_map: List[str]
        """Sequences of characters of the original words"""
        sentences_chars_ids: List[np.ndarray]
        """Strings of the original words split on the character level"""
        sentences_chars: List[List[List[str]]]
        

        def __init__(self,
                     include_chars: bool) -> None:

            self.words_vocab = {"<pad>": self.PAD, "<bos>": self.BOS, "<eos>": self.EOS}
            self.words_map = ["<pad>", "<bos>", "<eos>"]
            self.sentences_words_ids = []
            self.sentences_words = []
            self.include_chars = include_chars
            # For example, if the factor represents POS, then it makes no sense to keep
            # a characer level representation of them, then this should be `False`.
            if include_chars:
                self.chars_vocab = {"<pad>": self.PAD, "<bow>": self.BOW, "<eow>": self.EOW}
                self.chars_map = ["<pad>", "<bow>", "<eow>"]
                self.sentences_chars_ids = []
                self.sentences_chars = []

    class FactorBatch:
        def __init__(self,
                     sentences_words_ids: tensor,
                     sentences_chars_ids: tensor = None) -> None:

            self.sentences_words_ids = sentences_words_ids
            self.sentences_chars_ids = sentences_chars_ids

    class Dataset:
        SOURCE = 0
        TARGET = 1
        FACTORS = 2

        EXCLUSIONS = {'train': [1231, 2169, 2755, 4165, 5674, 9360, 10970,  # ede[ies]
                                1447, 1717, 4828, 8889, 9049,  # eie[des]
                                5147, 10606,  # missing token
                                2031, 4852, 5280],  # misc
                      'dev': [1109],
                      'test': []}

        """Contains a factor for each column of data"""
        _data: List['GumarDataset.Factor']
        """Number of sentences in the dataset"""
        _size: int
        """Random number generator for generating random batches"""
        _shuffler: np.random.RandomState

        def __init__(self,
                     data_file: TextIO,
                     add_bow_eow: bool = False,
                     max_sentences: int = 0,
                     max_sentence_len: int = 0,
                     name: str = '') -> None:

            # Create factors
            self._data = []
            for f in range(self.FACTORS):
                self._data.append(GumarDataset.Factor(f in [self.SOURCE, self.TARGET]))

            print(f'Loading {name}...')
            data_xml = ET.parse(data_file).getroot()
            for idx, sentence in enumerate(tqdm(data_xml)):
                if idx in GumarDataset.Dataset.EXCLUSIONS[name.lower()]:
                    continue
                src_temp = sentence[0].text
                src_temp = GumarDataset.preprocess(src_temp)
                src_temp = src_temp.split(' ')
                src_temp = [token for token in src_temp if token]

                tgt_temp = sentence[1].text
                tgt_temp = GumarDataset.preprocess(tgt_temp)
                tgt_temp = tgt_temp.split(' ')
                tgt_temp = [token for token in tgt_temp if token]

                # Drop empty sentences
                if not (src_temp and tgt_temp):
                    continue
                # # Align src and tgt or discard example
                # try:
                #     src_temp = [[token, 'n'] for token in src_temp]
                #     tgt_temp = [[token, 'n'] for token in tgt_temp]
                #     src, tgt = GumarDataset.align(src_temp, tgt_temp)
                # except:
                #     GumarDataset.Dataset.EXCLUSIONS[name].append(idx)
                #     continue
                
                # check_src = (' '.join([s[0] for s in src_temp]), ' '.join(src))
                # check_tgt = (' '.join([t[0] for t in tgt_temp]), ' '.join(tgt))
                # if check_src[0] != check_src[1] or check_tgt[0] != check_tgt[1]:
                #     continue
                src, tgt = src_temp, tgt_temp
                if max_sentence_len and (len(src) > max_sentence_len or len(tgt) > max_sentence_len):
                    continue

                for f in range(self.FACTORS):
                    factor = self._data[f]
                    if len(factor.sentences_words_ids):
                        factor.sentences_words_ids[-1] = np.array(
                            factor.sentences_words_ids[-1], np.int32)
                    factor.sentences_words_ids.append([])
                    factor.sentences_words.append([])
                    if add_bow_eow:
                        factor.sentences_words[-1].append(
                            factor.words_map[GumarDataset.Factor.BOS])
                        factor.sentences_words_ids[-1].append(
                            GumarDataset.Factor.BOS)
                    if factor.include_chars:
                        factor.sentences_chars_ids.append([])
                        factor.sentences_chars.append([])

                    sentence = tgt if f else src
                    # Word-level information
                    for word in sentence:
                        if word not in factor.words_vocab:
                            factor.words_vocab[word] = len(factor.words_map)
                            factor.words_map.append(word)
                        factor.sentences_words_ids[-1].append(factor.words_vocab[word])
                        factor.sentences_words[-1].append(word)

                        # Character-level information
                        if factor.include_chars:
                            factor.sentences_chars_ids[-1].append([])
                            factor.sentences_chars[-1].append([])
                            if add_bow_eow:
                                factor.sentences_chars_ids[-1][-1].append(
                                    GumarDataset.Factor.BOW)
                                factor.sentences_chars[-1][-1].append(
                                    factor.chars_map[GumarDataset.Factor.BOW])
                            for c in word:
                                if c not in factor.chars_vocab:
                                    factor.chars_vocab[c] = len(factor.chars_map)
                                    factor.chars_map.append(c)
                                factor.sentences_chars[-1][-1].append(c)
                                factor.sentences_chars_ids[-1][-1].append(
                                    factor.chars_vocab[c])
                            if add_bow_eow:
                                factor.sentences_chars_ids[-1][-1].append(
                                    GumarDataset.Factor.EOW)
                                factor.sentences_chars[-1][-1].append(
                                    factor.chars_map[GumarDataset.Factor.EOW])
                    if add_bow_eow:
                        factor.sentences_words_ids[-1].append(GumarDataset.Factor.EOS)
                        factor.sentences_words[-1].append(factor.words_map[GumarDataset.Factor.EOS])
                if max_sentences and len(self._data[self.SOURCE].sentences_words_ids) >= max_sentences:
                    break

            self._size = len(self._data[self.SOURCE].sentences_words_ids)

        @property
        def data(self):
            return self._data
        
        @property
        def data_words_chars(self):
            return list(zip(self._data[0].sentences_words_ids, \
                            self._data[1].sentences_words_ids, \
                            self._data[0].sentences_chars_ids, \
                            self._data[1].sentences_chars_ids))

        @property
        def size(self):
            return self._size


        def get_token_num(self):
            return sum(len(s) for s in self._data[self.SOURCE].sentences_words_ids), \
                   sum(len(s) for s in self._data[self.TARGET].sentences_words_ids), \
                   sum(len(s) for s in self._data[self.SOURCE].sentences_chars_ids), \
                   sum(len(s) for s in self._data[self.TARGET].sentences_words_ids)

        
        @staticmethod
        def generate_batch(data_batch):
            batch_size = len(data_batch)
            batch_perm = range(batch_size)
            batch: List[GumarDataset.FactorBatch] = []
            lengths = [[], []]

            # Word-level data
            data_batch_words = []
            for f in range(GumarDataset.Dataset.FACTORS):
                data_batch_words.append([example[f] for example in data_batch])
            for f, factor in enumerate(data_batch_words):
                max_sentence_len = max(len(data_batch[i][f]) for i in batch_perm)
                batch.append(GumarDataset.FactorBatch(
                    zeros([batch_size, max_sentence_len], dtype=long)))
                for i in range(batch_size):
                    batch[-1].sentences_words_ids[i, :len(factor[i])] = tensor(factor[i], dtype=long)
                    lengths[f].append(factor[i].size)

            # Character-level data
            data_batch_chars = []
            for f in range(2, GumarDataset.Dataset.FACTORS + 2):
                data_batch_chars.append([example[f] for example in data_batch])
            for f, factor in enumerate(data_batch_chars):
                max_word_len = max(
                    len(word_chars) for i in batch_perm for word_chars in factor[i])

                batch[f].sentences_chars_ids = zeros(
                    [batch_size, max_sentence_len, max_word_len], dtype=long)
                for i in range(batch_size):
                    for j, sentence_chars in enumerate(factor[i]):
                        batch[f].sentences_chars_ids[i, j, :len(
                            sentence_chars)] = tensor(sentence_chars, dtype=long)
                

            return batch[0].sentences_words_ids.permute(1, 0), \
                   batch[1].sentences_words_ids.permute(1, 0), \
                   batch[0].sentences_chars_ids.permute(1, 0, 2), \
                   batch[1].sentences_chars_ids.permute(1, 0, 2), \
                   lengths


    @staticmethod
    def align_subsequences(src_sub, tgt_sub):
        def include_alignment():
            # If there are 'i' and 'd' tokens in addition to 's'
            if [True for t in src_sub[start:end] if t[1] != 's']:
                src_sub_temp, tgt_sub_temp, alignment, flipped = GumarDataset.soft_align(
                    tgt_sub, src_sub, start, end)
                src_align = [a.split('-')[0] for a in alignment]
                if src_align == sorted(src_align):
                    align_dict = {}
                    for a in alignment:
                        a = a.split('-')
                        align_dict.setdefault(int(a[0]), []).append(int(a[1]))
                    align_dict = [(src_sub_temp[s], ' '.join(
                        map(lambda x: tgt_sub_temp[x], t))) for s, t in align_dict.items()]
                    for s, t in align_dict:
                        if flipped:
                            s, t = t, s
                        src_temp.append(s)
                        tgt_temp.append(t)
            # Else they are already aligned
            else:
                for j in range(start, end):
                    src_temp.append(src_sub[j][0])
                    tgt_temp.append(tgt_sub[j][0])

        start, end = -1, -1
        src_temp, tgt_temp = [], []
        for i, token in enumerate(src_sub):
            op = token[1]
            if start == -1 and op == 'e':
                src_temp.append(src_sub[i][0])
                tgt_temp.append(tgt_sub[i][0])
            elif start == -1 and op != 'e':
                start = i
            # RHS of OR is for when the
            elif start != -1 and op == 'e':
                end = i
                include_alignment()
                # Add first token with value 'e'
                src_temp.append(src_sub[i][0])
                tgt_temp.append(tgt_sub[i][0])
                start, end = -1, -1
        else:
            end = i + 1
            # If last operation is not e and we are in the
            # middle of a (possibly) badly aligned subsequence
            if start != -1:
                include_alignment()

        return src_temp, tgt_temp

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

        src, tgt = GumarDataset.align_subsequences(src_temp, tgt_temp)
        return src, tgt


    @staticmethod
    def soft_align(tgt, src, start, end):
        src = [token[0] for token in src[start:end] if token[1] != 'd']
        tgt = [token[0] for token in tgt[start:end] if token[1] != 'i']
        
        flipped = False
        if len(tgt) < len(src):
            src, tgt = tgt, src
            flipped = True
        
        I, J = len(tgt), len(src)
        alignment = []
        start = 0
        for i in range(I):
            best_score = inf
            best_j = 0

            src_assigned = list(Counter([a[0] for a in alignment]).items())
            last_src_assigned_counts_2 = src_assigned[-1][1] == 2 if src_assigned else False
            last_two_src_different = True if len(
                src_assigned) > 1 and src_assigned[-1][1] == 1 else False
            if len(tgt) - len(src) < 5 and last_src_assigned_counts_2:
                start = len(src_assigned) if start < J else J - 1
            elif len(tgt) - len(src) < 5 and last_two_src_different:
                start = len(src_assigned) - 1

            for j in range(start, J):
                context = [tgt[i][0]]
                if 1 <= i < I - 1 and I > 2:
                    context = [tgt[i-1][0] + tgt[i]
                               [0], tgt[i][0] + tgt[i+1][0]]
                for c in context:
                    # Add 1 penalty if source token was already assigned 2 times
                    #TODO: change to masi_distance or add a bigram lookahead feature
                    score = edit_distance(
                        src[j][0], tgt[i][0]) * ((masi_distance(set(c), set(src[j][0])) if 1 <= i < I else 0))
                    if score < best_score:
                        best_score = score
                        best_j = j
            alignment.append(f"{best_j}-{i}")

        return src, tgt, alignment, flipped


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
                 dataset: str,
                 add_bow_eow: bool = False,
                 max_sentences: int = 0,
                 max_sentence_len: int = 0):

        path = "{}.zip".format(dataset)
        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["TRAIN", "DEV", "TEST"]:
                with zip_file.open(f"{os.path.splitext(path)[0]}/{dataset}_annotated_Gumar_corpus.xml", "r") as dataset_file:
                    # setattr(self, dataset.lower(), self.Dataset(dataset_file,
                    #                                             add_bow_eow=add_bow_eow,
                    #                                             max_sentences=max_sentences,
                    #                                             max_sentence_len=max_sentence_len,
                    #                                             name=dataset.lower()))
                    inspect = []
                    data_xml = ET.parse(dataset_file).getroot()
                    for idx, ex in enumerate(data_xml):
                        for token in ex[2]:
                            enclitics = [bool(token.attrib.get(
                                f'prc{i}_form')) for i in range(4)]
                            if enclitics.count(True) > 1:
                                inspect.append(idx)
                                break
                    pass



if __name__ == "__main__":
    # if os.path.exists('alignments/useful_examples'):
    #     with open('alignments/useful_examples', 'rb') as u:
    #         useful_examples = pickle.load(u)
    # else:
    gumar = GumarDataset('annotated-gumar-corpus')
    pass
