import re
import regex
import os
from collections import Counter


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
        pass


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
    pass


split_into_src_tgt()
# generate_char_dict()
