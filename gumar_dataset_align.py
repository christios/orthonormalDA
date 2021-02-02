import os
import zipfile
import xml.etree.ElementTree as ET
from edit_distance import SequenceMatcher
import pyarabic.araby as araby
import re

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

            with open(f'alignments/gumar.src', 'a') as src, \
                 open(f'alignments/gumar.tgt', 'a') as tgt, \
                 open(f'alignments/gumar.a', 'a') as a:
                data_xml = ET.parse(data_file).getroot()
                for sentence in data_xml:
                    sentence_temp_src = sentence[0].text
                    sentence_temp_src = GumarDataset.preprocess(sentence_temp_src)
                    sentence_temp_src = sentence_temp_src.split(' ')
                    sentence_temp_src = [token for token in sentence_temp_src if token]

                    sentence_temp_tgt = sentence[1].text
                    sentence_temp_tgt = GumarDataset.preprocess(sentence_temp_tgt)
                    sentence_temp_tgt = sentence_temp_tgt.split(' ')
                    sentence_temp_tgt = [token for token in sentence_temp_tgt if token]
                    if not (sentence_temp_src and sentence_temp_tgt):
                        continue
                    print(' '.join(sentence_temp_src), file=src)
                    print(' '.join(sentence_temp_tgt), file=tgt)
                    print(' '.join(sentence_temp_src), ' '.join(
                        sentence_temp_tgt), sep=' ||| ', file=a)
                    
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


    def __init__(self, dataset, add_bow_eow=False, max_sentences=None):
        path = "{}.zip".format(dataset)

        for file in ['alignments/gumar.src', 'alignments/gumar.tgt', 'alignments/gumar.a']:
            if os.path.exists(file):
                os.remove(file)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["TRAIN", "DEV", "TEST"]:
                with zip_file.open(f"{os.path.splitext(path)[0]}/{dataset}_annotated_Gumar_corpus.xml", "r") as dataset_file:
                    setattr(self, dataset.lower(), self.Dataset(dataset_file,
                                                        train=self.train if dataset != "TRAIN" else None,
                                                        shuffle_batches=dataset == "TRAIN",
                                                        add_bow_eow=add_bow_eow,
                                                        max_sentences=max_sentences,
                                                        name=dataset))
        
        os.system('fast_align -d -o -v -i alignments/gumar.a > alignments/gumar.f')
        os.system('fast_align -d -o -v -r -i alignments/gumar.a > alignments/gumar.r')
        os.system('atools -i alignments/gumar.f -j alignments/gumar.r -c intersect > alignments/gumar.i')
        os.system('atools -i alignments/gumar.f -j alignments/gumar.r -c union > alignments/gumar.union')


if __name__ == "__main__":
    gumar = GumarDataset('annotated-gumar-corpus')
    pass
