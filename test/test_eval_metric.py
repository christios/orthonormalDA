from edit_distance import SequenceMatcher
from collections import Counter

def align_subsequences(src_sub, tgt_sub):
    def process_ne(src_sub, tgt_sub):
        src_temp, tgt_temp = [], []
        # If there are 'i' and 'd' tokens in addition to 's', then there is splitting
        # We should should align at the character level
        if [True for t in src_sub if t[1] != 's']:
            src_temp_, tgt_temp_ = soft_align(tgt_sub, src_sub)
            src_temp += src_temp_
            tgt_temp += tgt_temp_
        # Else they are already aligned but not equal
        else:
            for j in range(len(src_sub)):
                src_temp.append((src_sub[j][0], 'ne'))
                tgt_temp.append((tgt_sub[j][0], 'ne'))
        return src_temp, tgt_temp
        

    start, end = -1, -1
    src_temp, tgt_temp = [], []
    for i, token in enumerate(src_sub):
        op = token[1]
        if start == -1 and op == 'e':
            src_temp.append(src_sub[i])
            tgt_temp.append(tgt_sub[i])
        elif start == -1 and op != 'e':
            start = i
        elif start != -1 and op == 'e':
            end = i
            src_temp_, tgt_temp_ = process_ne(
                src_sub[start:end], tgt_sub[start:end])
            src_temp += src_temp_
            tgt_temp += tgt_temp_
            # Add first token with value 'e'
            src_temp.append(src_sub[i])
            tgt_temp.append(tgt_sub[i])
            start, end = -1, -1
    end = i + 1
    # If last operation is not e and we are in the
    # middle of a (possibly) badly aligned subsequence
    if start != -1:
        src_temp_, tgt_temp_ = process_ne(
            src_sub[start:end], tgt_sub[start:end])
        src_temp += src_temp_
        tgt_temp += tgt_temp_

    return src_temp, tgt_temp

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
    
    return src_temp, tgt_temp

def soft_align(tgt, src):
    """Alignment at the character level."""
    src = ' '.join([token[0] for token in src if token[1] != 'd'])
    tgt = ' '.join([token[0] for token in tgt if token[1] != 'i'])
    src_temp = [[char, 'n'] for char in src]
    tgt_temp = [[char, 'n'] for char in tgt]
    src_temp, tgt_temp = align(src_temp, tgt_temp)
    space_anchors = [0]
    for i, char in enumerate(src_temp):
        if char[0] == ' ' and char[1] == 'e':
            space_anchors.append(i + 1)
    space_anchors.append(len(src_temp) + 1)
    
    src_temp_, tgt_temp_ = [], []
    for i in range(len(space_anchors) - 1):
        src_sub_temp = src_temp[space_anchors[i]:space_anchors[i+1] - 1]
        tgt_sub_temp = tgt_temp[space_anchors[i]:space_anchors[i+1] - 1]
        src_sub_temp = ''.join([char[0] if char[0] != ' ' else '<space>'
                               for char in src_sub_temp if char[1] != 'd'])
        tgt_sub_temp = ''.join([char[0] if char[0] != ' ' else '<space>'
                               for char in tgt_sub_temp if char[1] != 'i'])
        src_temp_.append((src_sub_temp, 'ne'))
        tgt_temp_.append((tgt_sub_temp, 'ne'))
    return src_temp_, tgt_temp_





if __name__ == "__main__":
    with open('/local/ccayral/orthonormalDA/data/coda-corpus/beirut_src.txt') as f_src, \
        open('/local/ccayral/orthonormalDA/data/coda-corpus/beirut_tgt.txt') as f_tgt:
        src = f_src.readlines()
        tgt = f_tgt.readlines()
    system = ['']*len(src)
    system[0] = ' أنا بعطيك رقمه تلفونه و عنوانو .'
    data = zip(src, tgt, system)
    for i, (s, t, ss) in enumerate(data):
        src_temp = [[token, 'n'] for token in s.strip().split()]
        tgt_temp = [[token, 'n'] for token in t.strip().split()]
        src_temp, tgt_temp = align(src_temp, tgt_temp)
        src, tgt = align_subsequences(src_temp, tgt_temp)
        assert len(src) == len(tgt), f'Bad alignment for example at line {i}.'
    pass