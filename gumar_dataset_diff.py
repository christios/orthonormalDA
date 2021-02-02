import os
import zipfile
import xml.etree.ElementTree as ET
from edit_distance import SequenceMatcher
import pyarabic.araby as araby
import re
from collections import Counter

ALEFAT = araby.ALEFAT[:5] + tuple(araby.ALEFAT[-1])
ALEFAT_PATTERN = re.compile(u"[" + u"".join(ALEFAT) + u"]", re.UNICODE)

# Loads a NMT dataset sentence per sentence.
# - The data consists of three Datasets
#   - TRAIN
#   - DEV
#   - TEST
# - Each dataset is composed of factors (SOURCE, TARGET), each an
#   object containing the following fields:
#   - word_strings: Strings of the original words.
#   - word_ids: Word ids of the original words (uses <unk> and <pad>).
#   - words_map: String -> word_id map.
#   - words: Word_id -> string list.
#   - alphabet_map: Character -> char_id map.
#   - alphabet: Char_id -> character list.
#   - charseqs: Sequences of characters of the original words.
class GumarDataset:

    class Dataset:
        SOURCE = 0
        TARGET = 1
        FACTORS = 2

        def __init__(self, data_file, train=None, shuffle_batches=True, add_bow_eow=False, max_sentences=None, seed=42, name=None):
            self.four_grams = []
            with open(f'gumar_{name.lower()}.src', 'w') as src, \
                 open(f'gumar_{name.lower()}.tgt', 'w') as tgt, \
                 open(f'gumar_{name.lower()}.a', 'w') as a:
                data_xml = ET.parse(data_file).getroot()
                for l, sentence in enumerate(data_xml):
                    # if l == 444:
                    #     c = 0
                    sentence_temp_src = sentence[0].text
                    sentence_temp_src = GumarDataset.preprocess(sentence_temp_src)
                    sentence_temp_src = sentence_temp_src.split(' ')
                    sentence_temp_src = [[token, 'n'] for token in sentence_temp_src if token]

                    sentence_temp_tgt = sentence[1].text
                    sentence_temp_tgt = GumarDataset.preprocess(sentence_temp_tgt)
                    sentence_temp_tgt = sentence_temp_tgt.split(' ')
                    sentence_temp_tgt = [[token, 'n'] for token in sentence_temp_tgt if token]
                    if not (sentence_temp_src and sentence_temp_tgt):
                        continue
                        
                    tgt, src = GumarDataset.align(sentence_temp_tgt, sentence_temp_src)

                    for i in range(len(src) - 3):
                        four_gram = src[i:i+4]
                        four_gram = ''.join(list(map(lambda x: x[1], four_gram)))
                        if four_gram == 'ddss':
                            c = 0
                        self.four_grams.append(four_gram)
                    

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

    @staticmethod
    def align(seq1, seq2):
        """Corrects misalignments between the gold and predicted tokens
        which will almost almost always have different lengths due to inserted, 
        deleted, or substituted tookens in the predicted systme output."""

        sm = SequenceMatcher(a=list(map(lambda x: x[0], seq1)), b=list(map(lambda x: x[0], seq2)))
        ref_tokens, hyp_tokens = [], []
        opcodes = sm.get_opcodes()
        for tag, i1, i2, j1, j2 in opcodes:
        # If they are equal, do nothing except lowercase them
            if tag == 'equal':
                for i in range(i1, i2):
                    seq1[i][1] = 'e'
                    ref_tokens.append(seq1[i])
                for i in range(j1, j2):
                    seq2[i][1] = 'e'
                    hyp_tokens.append(seq2[i])
            # For insertions and deletions, put a filler of '***' on the other one, and
            # make the other all caps
            elif tag == 'delete':
                for i in range(i1, i2):
                    seq1[i][1] = 'd'
                    ref_tokens.append(seq1[i])
                for i in range(i1, i2):
                    hyp_tokens.append(seq1[i])
            elif tag == 'insert':
                for i in range(j1, j2):
                    seq2[i][1] = 'i'
                    ref_tokens.append(seq2[i])
                for i in range(j1, j2):
                    hyp_tokens.append(seq2[i])
            # More complicated logic for a substitution
            elif tag == 'replace':
                for i in range(i1, i2):
                    seq1[i][1] = 's'
                for i in range(j1, j2):
                    seq2[i][1] = 's'
                ref_tokens += seq1[i1:i2]
                hyp_tokens += seq2[j1:j2]

        return ref_tokens, hyp_tokens


    def __init__(self, dataset, add_bow_eow=False, max_sentences=None):
        path = "{}.zip".format(dataset)


        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["TRAIN", "DEV", "TEST"]:
                with zip_file.open(f"{os.path.splitext(path)[0]}/{dataset}_annotated_Gumar_corpus.xml", "r") as dataset_file:
                    setattr(self, dataset.lower(), self.Dataset(dataset_file,
                                                        train=self.train if dataset != "TRAIN" else None,
                                                        shuffle_batches=dataset == "TRAIN",
                                                        add_bow_eow=add_bow_eow,
                                                        max_sentences=max_sentences,
                                                        name=dataset))
        four_grams = Counter(self.train.four_grams + self.dev.four_grams + self.test.four_grams)
        pass

if __name__ == "__main__":
    gumar = GumarDataset('annotated-gumar-corpus')
    pass
