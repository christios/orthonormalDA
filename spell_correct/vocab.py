#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import re

import torch
from typing import List
from spell_correct.utils import read_corpus, pad_sents, pad_sents_char


class VocabEntry(object):
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """

    def __init__(self, word2id=None, *args):
        """ Init VocabEntry Instance.
        @param word2id (dict): dictionary mapping words 2 indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<s>'] = 1  # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}
        if args:
            x2y = args[0][0].split('2')
            setattr(self, args[0][0], args[0][1])
            setattr(self, x2y[1] + '2' + x2y[0],
                    {v: k for k, v in getattr(self, args[0][0]).items()})

        ## Additions to the A4 code:
        # self.char_list = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]""")
        self.char_list = list(
            """"0123456789#()+-.aeghilmnorst،؛؟ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيٱچڤݣ _,:""")
        self.char2id = dict()  # Converts characters to integers
        self.char2id['<pad>'] = 0
        self.char2id['<w>'] = 1
        self.char2id['</w>'] = 2
        self.char2id['<unk>'] = 3
        self.char2id['<b>'] = 4   # Segment Boundary
        for i, c in enumerate(self.char_list):
            self.char2id[c] = len(self.char2id)
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id['<w>']
        self.end_of_word = self.char2id['</w>']
        assert self.start_of_word+1 == self.end_of_word

        # Converts integers to characters
        self.id2char = {v: k for k, v in self.char2id.items()}
        ## End additions to the A4 code

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        @param wid (int): word index
        @returns word (str): word corresponding to index
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2charindices(self, sents, add_beg_end=True, segments=False):
        """ Convert list of sentences of words into list of list of list of character indices.
        @param sents (list[list[str]]): sentence(s) in words
        @return word_ids (list[list[list[int]]]): sentence(s) in indices
        If `segments` is True:
        @param sents (list[list[list[str]]]): sentence(s) in tokens/segments
        @return word_ids (list[list[list[list[int]]]]): sentence(s) in char indices
        """
        if segments:
            return [[[[self.char2id[c] for c in s] for s in t] for t in s] for s in sents]
        if add_beg_end:
            return [[[self.start_of_word]+[self.char2id[c] for c in w]+[self.end_of_word] for w in s] for s in sents]
        else:
            return [[[self.char2id[c] for c in w] for w in s] for s in sents]

    def pos2indices(self, sents):
        """ Convert list of sentences of words into list of list of indices.
        @param sents (list[list[str]]): sentence(s) in words
        @return word_ids (list[list[int]]): sentence(s) in indices
        """
        return [[[self[pos] for pos in t] for t in s] for s in sents]

    def taxonomy2indices(self, sents, most_common=None):
        """ Convert list of sentences of words into list of list of indices.
        @param sents (list[list[str]]): sentence(s) in words
        @return word_ids (list[list[int]]): sentence(s) in indices
        """
        most_common = most_common if most_common is not None else len(self.taxonomy2id)
        used_tags = {k: v for k, v in self.taxonomy2id.items() if v == most_common}
        tag_vectors = []
        for sent in sents:
            tag_vectors.append([])
            for token in sent:
                tag_vector = [self.word2id['<n>']]
                if isinstance(token, list):
                    for tag in token:
                        if tag in used_tags:
                            tag_id = 0 # used_tags[tag]
                            tag_vector[tag_id] = self.word2id['<y>']
                tag_vectors[-1].append(tag_vector)
        return tag_vectors

    def words2indices(self, sents, add_beg_end=True):
        """ Convert list of sentences of words into list of list of indices.
        @param sents (list[list[str]]): sentence(s) in words
        @return word_ids (list[list[int]]): sentence(s) in indices
        """
        if add_beg_end:
            return [[self.word2id['<s>']] + [self[w] for w in s] + [self.word2id['</s>']] for s in sents]
        else:
            return [[self[w] for w in s] for s in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        @param word_ids (list[int]): list of word ids
        @return sents (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor_char(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tensor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size, max_word_length)
        """
        ### YOUR CODE HERE for part 1g
        ### TODO:
        ###     Connect `words2charindices()` and `pad_sents_char()` which you've defined in
        ###     previous parts
        char_ids = self.words2charindices(sents)
        sents_t = pad_sents_char(char_ids, self.char2id['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return sents_var.permute(1, 0, 2)
        ### END YOUR CODE

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tesnor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def build_feature_vocab(sents: List[List[List[str]]]):
        pos_vocab = Counter()
        for sent in sents:
            for token in sent:
                pos_vocab.update(token)
        pos2id = dict()
        pos2id['<pad>'] = 0
        pos2id['<b>'] = 1
        pos2id['<unk>'] = 2
        for i, pos in enumerate(pos_vocab, start=3):
            pos2id[pos] = i
        return pos2id

    @staticmethod
    def build_taxonomy_map(sents: List[List[List[str]]]):
        taxonomy_tags = Counter()
        for sent in sents:
            for token in sent:
                if isinstance(token, list):
                    taxonomy_tags.update(token)
        taxonomy2id = dict()
        taxonomy2id['<pad>'] = 0
        taxonomy2id['<n>'] = 1
        taxonomy2id['<y>'] = 2
        taxonomy2id['<unk>'] = 3
        taxonomy_vocab = {}
        for i, tag in enumerate(taxonomy_tags.most_common()):
            taxonomy_vocab[tag[0]] = i
        return taxonomy2id, ('taxonomy2id', taxonomy_vocab)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[str]): corpus of text produced by read_corpus function
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(
            valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


class Vocab(object):
    """ Vocab encapsulating src and target langauges.
    """

    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """ Init Vocab.
        @param src_vocab (VocabEntry): VocabEntry for source language
        @param tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, freq_cutoff) -> 'Vocab':
        """ Build Vocabulary.
        @param src_sents (list[str]): Source sentences provided by read_corpus() function
        @param tgt_sents (list[str]): Target sentences provided by read_corpus() function
        @param vocab_size (int): Size of vocabulary for both source and target languages
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word.
        """
        assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)

        return Vocab(src, tgt)

    def save(self, file_path):
        """ Save Vocab to file as JSON dump.
        @param file_path (str): file path to vocab file
        """
        json.dump(dict(src_word2id=self.src.word2id,
                  tgt_word2id=self.tgt.word2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    src_sents = read_corpus(args['--train-src'], source='src')
    tgt_sents = read_corpus(args['--train-tgt'], source='tgt')

    vocab = Vocab.build(src_sents, tgt_sents, int(
        args['--size']), int(args['--freq-cutoff']))
    print('generated vocabulary, source %d words, target %d words' %
          (len(vocab.src), len(vocab.tgt)))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])
