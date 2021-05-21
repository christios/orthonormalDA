import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast
from transformers.utils.dummy_pt_objects import DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST

from spell_correct.utils import pad_sents_char, pad_sents
from spell_correct.utils import AlignmentHandler

class DialectData(Dataset):
    def __init__(self, args, data, vocab, device) -> None:
        self.vocab = vocab
        self.device = device
        self.args = args
        self.bert_tokenizer = None
        if args.use_bert_enc:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(
                args.bert_model, cache_dir=args.bert_cache_dir)
        
        self.use_sent_level = args.use_sent_level
        self.src_raw = [f[0] for f in data]
        self.tgt_raw = [f[1] for f in data]
        self.src_char = [f[2] for f in data]
        self.lengths_char_src = [[len(word) for word in sent]
                                 for sent in self.src_char]
        self.src_char = pad_sents_char(self.src_char,
                                       self.vocab.src.char2id['<pad>'],
                                       max_sent_length=args.max_sent_len,
                                       max_word_length=args.max_decode_len)
        self.tgt_char = [f[3] for f in data]
        self.lengths_char_tgt = [[len(word) for word in sent]
                                 for sent in self.tgt_char]
        self.tgt_char = pad_sents_char(self.tgt_char,
                                       self.vocab.tgt.char2id['<pad>'],
                                       max_sent_length=args.max_sent_len,
                                       max_word_length=args.max_decode_len)
        if args.use_bert_enc:
            self.src_bert = [f[4] for f in data]
            self.src_bert_mask = [f[5] for f in data]

        if args.use_sent_level:
            self.src_word = [f[6] for f in data]
            self.lengths_word = [len(sent) for sent in self.src_word]
            self.src_word = pad_sents(self.src_word,
                                      self.vocab.src.word2id['<pad>'],
                                      maxlen=args.max_sent_len)
            self.tgt_word = [f[7] for f in data]
            self.tgt_word = pad_sents(self.tgt_word,
                                      self.vocab.tgt.word2id['<pad>'],
                                      maxlen=args.max_sent_len)

        assert len(self.src_char) == len(self.src_word) if args.use_sent_level else len(self.src_char) \
            == len(self.tgt_char) == len(self.tgt_word) if args.use_sent_level else len(self.tgt_char) \
            == len(self.src_bert) if args.use_bert_enc else len(self.src_char) \
            == len(self.src_bert_mask) if args.use_bert_enc else len(self.src_char) \
            == len(self.lengths_char_src) == len(self.lengths_word) if args.use_sent_level else len(self.lengths_char_src)\
            == len(self.lengths_char_tgt), 'Error in data compilation'

    def __getitem__(self, index):
        src_bert = getattr(self, 'src_bert', None)
        src_bert_mask = getattr(self, 'src_bert_mask', None)
        src_word = getattr(self, 'src_word', None)
        tgt_word = getattr(self, 'tgt_word', None)
        lengths_word = getattr(self, 'lengths_word', None)
        if src_bert:
            src_bert = src_bert[index]
            src_bert_mask = src_bert_mask[index]
        if src_word:
            src_word = src_word[index]
            tgt_word = tgt_word[index]
            lengths_word = lengths_word[index]
        inputs = dict(src_raw=self.src_raw[index],
                      src_char=self.src_char[index],
                      lengths_char_src=self.lengths_char_src[index],
                      src_word=src_word,
                      lengths_word=lengths_word,
                      src_bert=src_bert,
                      src_bert_mask=src_bert_mask,
                      tgt_raw=self.tgt_raw[index],
                      tgt_char=self.tgt_char[index],
                      lengths_char_tgt=self.lengths_char_tgt[index],
                      tgt_word=tgt_word)
        return inputs

    def __len__(self):
        return len(self.src_char)

    def generate_batch(self, data_batch):
        src_raw_batch, tgt_raw_batch = [], []
        src_char_batch, src_word_batch = [], []
        lengths_char_src_batch, lengths_word_batch = [], []
        src_bert_batch, src_bert_mask_batch = [], []
        tgt_char_batch, tgt_word_batch = [], []
        for inputs in data_batch:
            src_raw_batch.append(inputs['src_raw'])
            src_char_batch.append(inputs['src_char'])
            lengths_char_src_batch += inputs['lengths_char_src']
            tgt_raw_batch.append(inputs['tgt_raw'])
            tgt_char_batch.append(inputs['tgt_char'])
            if inputs['src_word']:
                src_word_batch.append(inputs['src_word'])
                lengths_word_batch.append(inputs['lengths_word'])
                tgt_word_batch.append(inputs['tgt_word'])
            if inputs['src_bert']:
                src_bert_batch.append(inputs['src_bert'])
                src_bert_mask_batch.append(inputs['src_bert_mask'])

        src_char_batch = torch.tensor(src_char_batch, dtype=torch.long).permute(
            1, 0, 2).to(self.device)
        lengths_char_src_batch = torch.tensor(
            lengths_char_src_batch, dtype=torch.long)
        tgt_char_batch = torch.tensor(tgt_char_batch, dtype=torch.long).permute(
            1, 0, 2).to(self.device)

        if src_word_batch:
            tgt_word_batch = torch.tensor(tgt_word_batch, dtype=torch.long).permute(
                1, 0).to(self.device)
            lengths_word_batch = torch.tensor(
                lengths_word_batch, dtype=torch.long)
            src_word_batch = torch.tensor(src_word_batch, dtype=torch.long).permute(
                1, 0).to(self.device)
        if src_bert_batch:
            src_bert_batch = torch.tensor(
                src_bert_batch, dtype=torch.long).to(self.device)
            src_bert_mask_batch = torch.tensor(
                src_bert_mask_batch, dtype=torch.long).to(self.device)

        batch = dict(src_raw=src_raw_batch,
                    src_char=src_char_batch,
                    lengths_char_src=lengths_char_src_batch,
                    src_word=src_word_batch if src_word_batch else None,
                    lengths_word=lengths_word_batch if lengths_word_batch else None,
                    src_bert=src_bert_batch if src_bert_batch else None,
                    src_bert_mask=src_bert_mask_batch if src_bert_mask_batch else None,
                    tgt_raw=tgt_raw_batch,
                    tgt_char=tgt_char_batch,
                    tgt_word=tgt_word_batch if tgt_word_batch else None)

        return batch


def load_data(args, vocab, device):
    alignment_handler = AlignmentHandler(already_split=False, n=3)
    if args.use_bert_enc:
        bert_tokenizer = BertTokenizerFast.from_pretrained(
            args.bert_model, cache_dir=args.bert_cache_dir)

    with open(args.data_src) as f_src, open(args.data_tgt) as f_tgt:
        src_raw = [line.strip() for line in f_src.readlines()]
        tgt_raw = [line.strip() for line in f_tgt.readlines()]
    
    src_word, tgt_word = None, None
    if args.use_sent_level:
        src = [line.split() for line in src_raw]
        tgt = [line.split() for line in tgt_raw]
        word_ids_src = vocab.src.words2indices(src, add_beg_end=False)
        word_ids_tgt = vocab.tgt.words2indices(tgt)
        char_ids_src = vocab.src.words2charindices(src, add_beg_end=False)
        char_ids_tgt = vocab.tgt.words2charindices(tgt)
        src_word = word_ids_src[:args.data_size]
        tgt_word = word_ids_tgt[:args.data_size]
    else:
        src, tgt = alignment_handler.merge_split_src_tgt(
            src_raw, tgt_raw)
        char_ids_src = vocab.src.words2charindices(
            [sent.split() for sent in src])
        char_ids_tgt = vocab.tgt.words2charindices(tgt)
        src_char = char_ids_src[:args.data_size]
        tgt_char = char_ids_tgt[:args.data_size]

    src_bert, src_bert_mask = None, None
    if args.use_bert_enc:
        src_bert = bert_tokenizer(src_raw,
                                  padding="max_length",
                                  truncation=True,
                                  max_length=args.max_sent_len)
        src_bert, src_bert_mask = src_bert.input_ids, src_bert.attention_mask
        src_bert = src_bert[:args.data_size]
        src_bert_mask = src_bert_mask[:args.data_size]

    data = [x for x in [src_raw, tgt_raw, src_char, tgt_char,
                        src_bert, src_bert_mask, src_word, tgt_word] if x]
    data = list(zip(*data))

    lengths = [int(len(src_char)*args.train_split),
                int(len(src_char)*(1-args.train_split))]
    if sum(lengths) != len(src_char):
        lengths[0] += len(src_char) - sum(lengths)
    train_data, dev_data = random_split(data, lengths)

    train_data = DialectData(args, train_data, vocab, device)
    dev_data = DialectData(args, dev_data, vocab, device)

    train_iter = DataLoader(train_data, batch_size=args.batch_size,
                                shuffle=True, collate_fn=train_data.generate_batch)
    dev_iter = DataLoader(dev_data, batch_size=len(dev_data),
                            collate_fn=dev_data.generate_batch)
    return train_iter, dev_iter


def process_raw_inputs(args, vocab, raw_inputs, device):
        if args.use_bert_enc:
            bert_tokenizer = BertTokenizerFast.from_pretrained(
                args.bert_model, cache_dir=args.bert_cache_dir)
        src = vocab.src.words2charindices(
            [[sent[0]] for sent in raw_inputs])
        tgt = vocab.src.words2charindices(
            [[sent[1]] for sent in raw_inputs])

        src_bert = bert_tokenizer(raw_inputs,
                                  padding="max_length",
                                  truncation=True,
                                  max_length=args.max_sent_len)
        src_bert, src_bert_mask = src_bert.input_ids, src_bert.attention_mask

        data = list(zip(src, src_bert, src_bert_mask, tgt))
        data = DialectData(args, data, vocab, device)

        return DataLoader(data, batch_size=len(data),
                          collate_fn=data.generate_batch)
