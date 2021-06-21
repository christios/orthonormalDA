from spell_correct.utils import AlignmentHandler
import json


alignment_handler = AlignmentHandler(already_split=False, n=3)

with open('/local/ccayral/orthonormalDA1/data/coda-corpus/beirut_src.txt') as f_src, open('/local/ccayral/orthonormalDA1/data/coda-corpus/beirut_tgt.txt') as f_tgt, open('/local/ccayral/orthonormalDA1/test/coda_examples.tsv', 'w') as f_w:
    src_raw = [line.strip() for line in f_src.readlines()]
    tgt_raw = [line.strip() for line in f_tgt.readlines()]
    src, tgt = alignment_handler.merge_split_src_tgt(
        src_raw, tgt_raw)
    coda_examples = []
    for sent in zip(src, tgt):
        for i in range(len(sent[0])):
            coda_example = {'raw': sent[0][i],
                            'coda': sent[1][i],
                            'context': ' '.join(sent[1])
            }
            coda_examples.append(coda_example)
    json.dump(coda_examples, f_w, ensure_ascii=False)
