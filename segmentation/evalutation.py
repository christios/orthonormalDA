from edit_distance import SequenceMatcher
import torch


def tokenize(sentences_gold, sentences_pred):
    """Returns an aligned version of the gold and pred tokens which can
    then be used to match boundaries."""
    tokens = []
    for sentences in [sentences_gold, sentences_pred]:
        tokens.append([])
        for idx, sent in enumerate(sentences):
            tokenized_sent = sent.split(' ')
            for i, token in enumerate(tokenized_sent):
                boundary = True if i == len(tokenized_sent) - 1 else False
                tokens[-1].append([idx, token, boundary, 'e'])

    tokens_gold, tokens_pred = correct_alignment(tokens[0], tokens[1])
    return tokens_gold, tokens_pred

def correct_alignment(seq1, seq2):
    """Corrects misalignments between the gold and predicted tokens
    which will almost almost always have different lengths due to inserted, 
    deleted, or substituted tookens in the predicted systme output."""

    sm = SequenceMatcher(
        a=list(map(lambda x: x[1], seq1)), b=list(map(lambda x: x[1], seq2)))
    ref_tokens, hyp_tokens = [], []
    opcodes = sm.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
            # If they are equal, do nothing except lowercase them
        if tag == 'equal':
            for i in range(i1, i2):
                seq1[i][3] = 'e'
                ref_tokens.append(seq1[i])
            for i in range(j1, j2):
                seq2[i][3] = 'e'
                hyp_tokens.append(seq2[i])
        # For insertions and deletions, put a filler of '***' on the other one, and
        # make the other all caps
        elif tag == 'delete':
            for i in range(i1, i2):
                seq1[i][3] = 'd'
                ref_tokens.append(seq1[i])
            for i in range(i1, i2):
                hyp_tokens.append(seq1[i])
        elif tag == 'insert':
            for i in range(j1, j2):
                seq2[i][3] = 'i'
                ref_tokens.append(seq2[i])
            for i in range(j1, j2):
                hyp_tokens.append(seq2[i])
        # More complicated logic for a substitution
        elif tag == 'replace':
            for i in range(i1, i2):
                seq1[i][3] = 's'
            for i in range(j1, j2):
                seq2[i][3] = 's'
            ref_tokens += seq1[i1:i2]
            hyp_tokens += seq2[j1:j2]

    return ref_tokens, hyp_tokens

def process_indices_batch(data, positive_label, pad_label):
    tp, fn, fp = 0, 0, 0
    y_pred_tokens_batch = data[0]
    y_gold_tokens_batch = data[1]
    for doc in zip(y_pred_tokens_batch, y_gold_tokens_batch):
        y_pred_tokens = doc[0]
        y_gold_tokens = doc[1]
        if isinstance(y_gold_tokens, list):
            try:
                mask_index = y_gold_tokens.index(pad_label)
            except:
                mask_index = len(y_gold_tokens)
        else:
            mask_index = torch.where(y_gold_tokens == pad_label)[0]
            if mask_index.size:
                mask_index = mask_index[0]
            else:
                mask_index = len(y_gold_tokens)

        boundaries_pred = [i for i, token in enumerate(
            y_pred_tokens[:mask_index]) if token == positive_label]
        boundaries_gold = [i for i, token in enumerate(
            y_gold_tokens[:mask_index]) if token == positive_label]

        metadata = {'boundaries_pred': boundaries_pred,
                        'boundaries_gold': boundaries_gold}

        metadata.update(sensitivity_specificity(boundaries_pred, boundaries_gold))

        tp += len(metadata['tp'])
        fn += len(metadata['fn'])
        fp += len(metadata['fp'])

    metadata = {'tp': tp,
                'fn': fn,
                'fp': fp}
    return metadata

def sensitivity_specificity(boundaries_pred, boundaries_gold):
    boundaries_pred_set = set(boundaries_pred)
    boundaries_gold_set = set(boundaries_gold)

    tp = boundaries_pred_set.intersection(boundaries_gold_set)
    fp = boundaries_pred_set - tp
    fn = boundaries_gold_set - boundaries_pred_set

    precision = len(tp) / (len(tp) + len(fp)) if tp or fp else 0
    recall = len(tp) / (len(tp) + len(fn)) if tp or fn else 0
    fscore = 2 * precision * recall / \
        (precision + recall) if precision + recall != 0 else 0
    metadata = {'boundaries_pred': boundaries_pred,
                'boundaries_gold': boundaries_gold,
                'tp': tp,
                'fn': fn,
                'fp': fp,
                'precision': precision,
                'recall': recall,
                'fscore': fscore
            }
    return metadata
