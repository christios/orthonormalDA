import os
import zipfile
import xml.etree.ElementTree as ET
from edit_distance import SequenceMatcher
import pyarabic.araby as araby
import re
from collections import Counter
from tqdm import tqdm
from nltk import edit_distance, masi_distance
from math import inf

ALEFAT = araby.ALEFAT[:5] + tuple(araby.ALEFAT[-1])
ALEFAT_PATTERN = re.compile(u"[" + u"".join(ALEFAT) + u"]", re.UNICODE)

c, y = 0, 0

class GumarDataset:

    class Dataset:

        EXCLUSIONS = {'train': [1231, 2169, 2755, 4165, 5674, 9360, 10970, #ede[ies]
                                1447, 1717, 4828, 8889, 9049,  # eie[des]
                                5147, 10606,  # missing token
                                2031, 4852, 5280], # misc
                      'dev': [1109],
                      'test': []}

        def __init__(self, data_file, train=None, shuffle_batches=True, add_bow_eow=False, max_sentences=None, seed=42, name=None):
            
            self.bad_alignment, self.clean_data = {}, {}
            data_xml = ET.parse(data_file).getroot()
            global y
            
            for idx, sentence in tqdm(enumerate(data_xml)):
                try:
                    if idx not in GumarDataset.Dataset.EXCLUSIONS[name.lower()]:
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
                        #TODO: deal with this case
                        if len(sentence_temp_src) > len(sentence_temp_tgt):
                            y +=1
                        
                        src, tgt = GumarDataset.align(sentence_temp_src, sentence_temp_tgt)
                        
                        # GumarDataset.examine_patterns(self, src, idx, sentence_temp_src, sentence_temp_tgt, self.four_grams, re.compile(r'eie[des]'), )
                        assert len(src) == len(tgt), f'Bad alignment'
                        
                        check_src = (' '.join([s[0] for s in sentence_temp_src]), ' '.join(src))
                        check_tgt = (' '.join([t[0] for t in sentence_temp_tgt]), ' '.join(tgt))
                        if check_src[0] != check_src[1] or check_tgt[0] != check_tgt[1]:
                            self.bad_alignment[idx] = (check_src, check_tgt)
                        else:
                            self.clean_data[idx] = (src, tgt)

                            with open(f'alignments/gumar_diff.src', 'a') as src_f, \
                                    open(f'alignments/gumar_diff.tgt', 'a') as tgt_f, \
                                    open(f'alignments/gumar_diff.a', 'a') as a_f:
                                print(' '.join(map(lambda s: re.sub(r' ', r'_', s), src)), file=src_f)
                                print(' '.join(map(lambda t: re.sub(r' ', r'_', t), tgt)), file=tgt_f)
                                print(' '.join([f'{x}-{x}' for x in range(len(src))]), file=a_f)
                except:
                    GumarDataset.Dataset.EXCLUSIONS[name.lower()].append(idx)


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
    def align_subsequences(src, tgt):
        def include_alignment():
            global c
            # If there are 'i' and 'd' tokens in addition to 's'
            if [True for t in src[start:end] if t[1] != 's']:
                s_temp, t_temp, alignment, flipped = GumarDataset.soft_align(
                    tgt, src, start, end)
                src_align = [a.split('-')[0] for a in alignment]
                #TODO: deal with this case
                if src_align != sorted(src_align):
                    c += 1
                else:
                    align_dict = {}
                    for a in alignment:
                        a = a.split('-')
                        align_dict.setdefault(int(a[0]), []).append(int(a[1]))
                    align_dict = [(s_temp[s], ' '.join(
                        map(lambda x: t_temp[x], t))) for s, t in align_dict.items()]
                    for s, t in align_dict:
                        if flipped:
                            s, t = t, s
                        src_temp.append(s)
                        tgt_temp.append(t)
            # Else they are already aligned
            else:
                for j in range(start, end):
                    src_temp.append(src[j][0])
                    tgt_temp.append(tgt[j][0])

        start, end = -1, -1
        src_temp, tgt_temp = [], []
        for i, token in enumerate(src):
            op = token[1]
            if start == -1 and op == 'e':
                src_temp.append(src[i][0])
                tgt_temp.append(tgt[i][0])
            elif start == -1 and op != 'e':
                start = i
            # RHS of OR is for when the
            elif start != -1 and op == 'e':
                end = i
                include_alignment()
                # Add first token with value 'e'
                src_temp.append(src[i][0])
                tgt_temp.append(tgt[i][0])
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

        sm = SequenceMatcher(a=list(map(lambda x: x[0], tgt)), b=list(map(lambda x: x[0], src)))
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
    def LCS(X, Y):
        m = len(X)
        n = len(Y)
        L = [[0 for x in range(n + 1)]
            for y in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if (i == 0 or j == 0):
                    L[i][j] = 0
                elif (X[i - 1] == Y[j - 1]):
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j],
                                L[i][j - 1])
        return L[m][n]


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
        # if len(tgt) > 5 or len(src) > 5:
        #     print('lol')
        for i in range(I):
            best_score = inf
            best_j = 0
            
            src_assigned = list(Counter([a[0] for a in alignment]).items())
            last_src_assigned_counts_2 = src_assigned[-1][1] == 2 if src_assigned else False
            last_two_src_different = True if len(src_assigned) > 1 and src_assigned[-1][1] == 1 else False
            if len(tgt) - len(src) < 5 and last_src_assigned_counts_2:
                start = len(src_assigned) if start < J else J - 1
            elif len(tgt) - len(src) < 5 and last_two_src_different:
                start = len(src_assigned) - 1
            
            #TODO: change J here too
            for j in range(start, J):
                context = [tgt[i][0]]
                if 1 <= i < I - 1 and I > 2:
                    context = [tgt[i-1][0] + tgt[i][0], tgt[i][0] + tgt[i+1][0]]
                for c in context:
                    # Add 1 penalty if source token was already assigned 2 times
                    #TODO: change to masi_distance or add a bigram lookahead feature
                    score = edit_distance(
                        src[j][0], tgt[i][0]) * ((masi_distance(set(c), set(src[j][0])) if 1 <= i < I else 0))
                    if score < best_score:
                        best_score = score
                        best_j = j
            alignment.append(f"{best_j}-{i}")
        # alignment = ' '.join(alignment)
        # if len(tgt) > 5 or len(src) > 5:
        #     print('lol')
        return src, tgt, alignment, flipped


    @staticmethod
    def examine_patterns(dataset, src, idx, s, t, pattern, output, ngram=4):
        for i in range(len(src) - (ngram-1)):
            four_gram = src[i:i+ngram+1]
            four_gram = ''.join(list(map(lambda x: x[1], four_gram)))
            if pattern.search(four_gram):
                dataset.exclusions[idx] = [s, t]
            output.append(four_gram)


    def __init__(self, dataset, add_bow_eow=False, max_sentences=None):
        path = "{}.zip".format(dataset)

        if os.path.exists('alignments/gumar_diff.src'):
            os.remove('alignments/gumar_diff.src')
            os.remove('alignments/gumar_diff.tgt')
            os.remove('alignments/gumar_diff.a')

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["TRAIN", "DEV", "TEST"]:
                with zip_file.open(f"{os.path.splitext(path)[0]}/{dataset}_annotated_Gumar_corpus.xml", "r") as dataset_file:
                    setattr(self, dataset.lower(), self.Dataset(dataset_file,
                                                        train=self.train if dataset != "TRAIN" else None,
                                                        shuffle_batches=dataset == "TRAIN",
                                                        add_bow_eow=add_bow_eow,
                                                        max_sentences=max_sentences,
                                                        name=dataset))
        # four_grams = Counter(self.train.four_grams + self.dev.four_grams + self.test.four_grams)
        # bigrams = Counter(self.train.bigrams +self.dev.bigrams + self.test.bigrams)
        pass

if __name__ == "__main__":
    gumar = GumarDataset('annotated-gumar-corpus')
    useful_examples = []
    total_examples = 0
    for sentence in gumar.train.clean_data.values():
        src, tgt = sentence
        total_examples += len(src)
        for i in range(len(src)):
            if src[i] != tgt[i]:
                useful_examples.append((src[i], tgt[i]))
    pass
