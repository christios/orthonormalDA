import os
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pyarabic.araby as araby
import re
from typing import Dict, List, Optional, TextIO

ALEFAT = araby.ALEFAT[:5] + tuple(araby.ALEFAT[-1])
ALEFAT_PATTERN = re.compile(u"[" + u"".join(ALEFAT) + u"]", re.UNICODE)

class GumarDataset:
    """ * Loads a NMT dataset sentence per sentence.
        * The data consists of three Datasets
            - TRAIN
            - DEV
            - TEST
        *  Each dataset is composed of factors (SOURCE, TARGET)
    """

    train: 'GumarDataset.Dataset'
    dev: 'GumarDataset.Dataset'
    test: 'GumarDataset.Dataset'

    class Factor:
        PAD = 0
        UNK = 1
        BOW = 2
        EOW = 3

        """String -> word_id map"""
        words_map: Dict[str, int]
        """Word_id -> string list"""
        words: List[str]
        """Word ids of the original words (uses <unk> and <pad>)"""
        word_ids: List[np.ndarray]
        """Strings of the original words"""
        word_strings: List[np.ndarray]
        """Character -> char_id map"""
        alphabet_map: Dict[str, int]
        """Char_id -> character list"""
        alphabet: List[str]
        """Sequences of characters of the original words"""
        charseqs: List[np.ndarray]

        def __init__(self,
                     characters: bool,
                     train: Optional['GumarDataset.Factor'] = None) -> None:

            self.words_map = train.words_map if train else {
                "<pad>": self.PAD, "<unk>": self.UNK}
            self.words = train.words if train else ["<pad>", "<unk>"]
            self.word_ids = []
            self.word_strings = []
            self.characters = characters
            # For example, if the factor represents POS, then it makes no sense to keep
            # a characer level representation of them, then this should be `False`.
            if characters:
                self.alphabet_map = train.alphabet_map if train else {
                    "<pad>": self.PAD, "<unk>": self.UNK, "<bow>": self.BOW, "<eow>": self.EOW}
                self.alphabet = train.alphabet if train else [
                    "<pad>", "<unk>", "<bow>", "<eow>"]
                self.charseqs = []

    class FactorBatch:
        def __init__(self,
                     word_ids: List[np.ndarray],
                     charseqs: List[np.ndarray] = None) -> None:

            self.word_ids = word_ids
            self.charseqs = charseqs

    class Dataset:
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
                     shuffle_batches: bool = True,
                     add_bow_eow: bool = False,
                     max_sentences: int = None,
                     seed: int = 42) -> None:

            # Create factors
            self._data = []
            for f in range(self.FACTORS):
                self._data.append(GumarDataset.Factor(
                    f in [self.SOURCE, self.TARGET], train._data[f] if train else None))

            data_xml = ET.parse(data_file).getroot()
            for sentence in data_xml:
                sentence_temp_src = sentence[0].text
                sentence_temp_src = GumarDataset.preprocess(sentence_temp_src)
                sentence_temp_src = sentence_temp_src.split(' ')
                sentence_temp_src = [
                    token for token in sentence_temp_src if token]

                sentence_temp_tgt = sentence[1].text
                sentence_temp_tgt = GumarDataset.preprocess(sentence_temp_tgt)
                sentence_temp_tgt = sentence_temp_tgt.split(' ')
                sentence_temp_tgt = [
                    token for token in sentence_temp_tgt if token]

                # Drop different-length and empty sentences
                if len(sentence_temp_tgt) != len(sentence_temp_src) or \
                        not (sentence_temp_src and sentence_temp_tgt):
                    continue

                for f in range(self.FACTORS):
                    factor = self._data[f]
                    if len(factor.word_ids):
                        factor.word_ids[-1] = np.array(
                            factor.word_ids[-1], np.int32)
                    factor.word_ids.append([])
                    factor.word_strings.append([])
                    if factor.characters:
                        factor.charseqs.append([])

                    sentence_temp = sentence_temp_tgt if f else sentence_temp_src
                    # Word-level information
                    for word in sentence_temp:
                        if word not in factor.words_map:
                            if train:
                                word = "<unk>"
                            else:
                                factor.words_map[word] = len(factor.words)
                                factor.words.append(word)
                        factor.word_ids[-1].append(factor.words_map[word])
                        factor.word_strings[-1].append(word)

                        # Character-level information
                        if factor.characters:
                            factor.charseqs[-1].append([])
                            if add_bow_eow:
                                factor.charseqs[-1][-1].append(
                                    GumarDataset.Factor.BOW)
                            for c in word:
                                if c not in factor.alphabet_map:
                                    if train:
                                        c = "<unk>"
                                    else:
                                        factor.alphabet_map[c] = len(
                                            factor.alphabet)
                                        factor.alphabet.append(c)
                                factor.charseqs[-1][-1].append(
                                    factor.alphabet_map[c])
                            if add_bow_eow:
                                factor.charseqs[-1][-1].append(
                                    GumarDataset.Factor.EOW)

                if max_sentences is not None and len(self._data[self.SOURCE].word_ids) >= max_sentences:
                    break

            self._size = len(self._data[self.SOURCE].word_ids)
            self._shuffler = np.random.RandomState(
                seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None):
            permutation = self._shuffler.permutation(
                self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = []
                # Current implementation has same length for source and target
                max_sentence_len = max(
                    len(self._data[self.SOURCE].word_ids[i]) for i in batch_perm)

                # Word-level data
                for factor in self._data:
                    batch.append(GumarDataset.FactorBatch(
                        np.zeros([batch_size, max_sentence_len], np.int32)))
                    for i in range(batch_size):
                        batch[-1].word_ids[i, :len(factor.word_ids[batch_perm[i]])
                                           ] = factor.word_ids[batch_perm[i]]

                # Character-level data
                for f, factor in enumerate(self._data):
                    if not factor.characters:
                        continue
                    max_charseq_len = max(
                        len(charseq) for i in batch_perm for charseq in factor.charseqs[i])

                    batch[f].charseqs = np.zeros(
                        [batch_size, max_sentence_len, max_charseq_len], np.int32)
                    for i in range(batch_size):
                        for j, charseq in enumerate(factor.charseqs[batch_perm[i]]):
                            batch[f].charseqs[i, j, :len(charseq)] = charseq

                yield batch

    @staticmethod
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
                 max_sentences: int = None):

        path = "{}.zip".format(dataset)
        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["TRAIN", "DEV", "TEST"]:
                with zip_file.open(f"{os.path.splitext(path)[0]}/{dataset}_annotated_Gumar_corpus.xml", "r") as dataset_file:
                    setattr(self, dataset.lower(), self.Dataset(dataset_file,
                                                                train=self.train if dataset != "TRAIN" else None,
                                                                shuffle_batches=dataset == "TRAIN",
                                                                add_bow_eow=add_bow_eow,
                                                                max_sentences=max_sentences))


if __name__ == "__main__":
    gumar = GumarDataset('annotated-gumar-corpus')
