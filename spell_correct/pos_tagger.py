from typing import Optional, List, Union, Dict
import numpy as np
from math import prod

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from transformers import AutoModel

from spell_correct.models import BiLSTM_CRF, Encoder


class POSTagger(nn.Module):
    def __init__(self,
                 args,
                 device: torch.device,
                 vocab,
                 bert_tokenizer=None):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.device = device
        self.bert_tokenizer = bert_tokenizer

        if args.use_bert_enc:
            self.bert_encoder = AutoModel.from_pretrained(
                args.bert_model, cache_dir=args.bert_cache_dir)
            self.bert_encoder.config.eos_token_id = self.bert_tokenizer.sep_token_id
            self.bert_encoder.config.pad_token_id = self.bert_tokenizer.pad_token_id
            self.word_to_char_bert = nn.Linear(
                self.bert_encoder.config.hidden_size, args.rnn_dim_char)
            self.rel = nn.ReLU()
            for param in self.bert_encoder.parameters():
                param.requires_grad = False
        bert_emb_dim = self.bert_encoder.config.hidden_size if args.use_bert_enc == 'concat' else 0

        self.pad_token_idx = vocab.src.char2id['<pad>']

        self.encoder = Encoder(input_dim=len(vocab.src.char2id),
                               emb_dim=args.ce_dim,
                               enc_hid_dim=args.rnn_dim_char,
                               dec_hid_dim=args.rnn_dim_char,
                               num_layers=1,
                               dropout=args.dropout)

        self.bi_lstm_crf = BiLSTM_CRF(input_dim=args.rnn_dim_char + bert_emb_dim,
                                      num_tags=len(vocab.tgt.word2id),
                                      bert_emb_dim=0,
                                      rnn_dim=args.rnn_dim,
                                      num_layers=args.rnn_layers,
                                      dropout=args.dropout,
                                      pad_index=self.pad_token_idx,
                                      device=self.device)

    def forward(self,
                batch: Dict[str, Union[Tensor,
                                       List[List[int]], List[str], List[int], None]],
                use_crf=True,
                output_loss=True,
                decode=False) -> Tensor:
        """- `batch_size_char` is the number of valid words (not padded) in the char src input
           - src_char -> [max_len_src_word + 1, batch_size_word, max_len_src_char + 1]
           - tgt_char -> same as src_char
           - src_bert -> [batch_size_word, max_len_src_word]
           - teacher_forcing_ratio is probability to use teacher forcing
            e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time"""
        src_segments: Tensor = batch['src_segments']
        pos_labels: Tensor = batch['pos_labels']
        if batch.get('src_bert') is not None:
            src_bert: Optional[List[str]] = batch['src_bert']
            src_bert_mask: Optional[List[int]] = batch['src_bert_mask']
        else:
            src_bert, src_bert_mask = None, None

        # processed_inputs = self._process_inputs(
        #     src_char, src_bert, src_bert_mask, tgt_char)
        processed_segments = self._process_char_word_embeddings(
            src_segments, src_bert, src_bert_mask)
        # src_char_valid -> [max_len_src_char + 1, batch_size_char]
        src_char_valid = processed_segments['src_char_valid']
        src_indexes_valid = processed_segments['src_indexes_valid']
        # lengths_char_src -> [(max_len_src_char + 1) * batch_size_char]
        lengths_char_src = processed_segments['lengths_char_src']
        bert_encodings = processed_segments['bert_encodings']

        _, hidden, _ = self.encoder(
            src_char_valid, lengths_char_src)
        if bert_encodings is not None:
            segments = torch.cat([hidden[0], bert_encodings], dim=1)
        else:
            segments = hidden[0]
        word_embeddings = self._scatter(segments, src_indexes_valid, src_segments.shape)
        processed_inputs = self._process_input_bi_lstm_crf(word_embeddings, pos_labels)
        word_embeddings_valid = processed_inputs['word_embeddings_valid']
        pos_labels_valid = processed_inputs['pos_labels_valid']
        output = self.bi_lstm_crf(word_embeddings_valid, pos_labels_valid, decode, output_loss, use_crf)
        return output

    def _process_char_word_embeddings(self,
                                      src_segments,
                                      src_bert=None,
                                      src_bert_mask=None,
                                      use_cache=False):
        # max_word_len = self.args.max_word_len + 1
        bert_encodings = None
        if src_bert is not None and src_bert_mask is not None:
            outputs = self.bert_encoder(src_bert, src_bert_mask,
                                        use_cache=use_cache)
            logits_words = outputs.last_hidden_state
            logits_words = self._map_bpe_tokens_to_tgt(logits_words, src_bert)
            logits_words_debatch = logits_words.reshape(
                -1, logits_words.size(-1))
            bert_indexes_valid = torch.any(logits_words_debatch, dim=1)
            logits_words_valid = logits_words_debatch[bert_indexes_valid]
            if self.args.use_bert_enc == 'init':
                logits_words_valid = self.rel(
                    self.word_to_char_bert(logits_words_valid))
            bert_encodings = logits_words_valid

        valid_segments = self._get_valid_segments(src_segments, bert_encodings)
        src_char_valid = valid_segments['src_char_valid']
        src_indexes_valid = valid_segments['valid_indexes']
        lengths_char_src = valid_segments['lengths_char_src']
        bert_encodings = valid_segments['bert_encodings']
        
        if self.args.use_bert_enc == 'concat':
            assert bert_encodings.size(0) == src_char_valid.size(1)

        processed_inputs = dict(src_char_valid=src_char_valid,
                                src_indexes_valid=src_indexes_valid,
                                lengths_char_src=lengths_char_src,
                                bert_encodings=bert_encodings)

        return processed_inputs

    def _process_input_bi_lstm_crf(self, word_embeddings, pos_labels):
        batch_size = word_embeddings.size(0)
        max_sent_len = word_embeddings.size(1)
        max_token_len = word_embeddings.size(2)
        context_size = word_embeddings.size(3)
        embedd_size = word_embeddings.size(4)
        contexts = batch_size * max_sent_len * max_token_len
        
        pos_labels_batch = pos_labels.reshape(contexts, context_size)
        valid_indexes = torch.any(pos_labels_batch != self.pad_token_idx, dim=1)
        pos_labels_valid = pos_labels_batch[valid_indexes]
        pos_labels_valid = pos_labels_valid.permute(1, 0)
        
        word_embeddings_batch = word_embeddings.reshape(contexts, context_size, embedd_size)
        word_embeddings_valid = word_embeddings_batch[valid_indexes]
        word_embeddings_valid = word_embeddings_valid.permute(1, 0, 2)

        processed_inputs = dict(word_embeddings_valid=word_embeddings_valid,
                                pos_labels_valid=pos_labels_valid,
                                valid_indexes=valid_indexes)

        return processed_inputs

    def _get_valid_segments(self, src_segments, bert_encodings=None):
        bert_encodings_ = None
        if bert_encodings is not None:
            bert_encodings_ = []
            i = 0
            for sent in src_segments:
                for token in sent:
                    if torch.any(token):
                        for seg in token:
                            if torch.any(seg):
                                for context_seg in seg:
                                    if torch.any(context_seg):
                                        bert_encodings_.append(bert_encodings[i])
                                i += 1
            bert_encodings_ = torch.stack(bert_encodings_).to(self.device)

        src_char_debatch = src_segments.reshape(-1, src_segments.size(-1))
        valid_indexes = torch.any(
            src_char_debatch!= self.pad_token_idx, dim=1)
        src_char_valid = src_char_debatch[valid_indexes]
        
        lengths_char_src = self._lengths(src_char_valid)
        src_char_valid = src_char_valid.permute(1, 0)

        valid_segments = dict(src_char_valid=src_char_valid,
                              valid_indexes=valid_indexes,
                              lengths_char_src=lengths_char_src,
                              bert_encodings=bert_encodings_)
        return valid_segments

    def _lengths(self, src):
        mask = torch.where(src != self.pad_token_idx)[1]
        zero_indexes = torch.where(mask == 0)[0]
        zero_indexes = torch.cat(
            [zero_indexes, torch.tensor([mask.shape[0]], device=self.device)])
        lengths = []
        for i in range(1, zero_indexes.shape[0]):
            lengths.append((zero_indexes[i] - zero_indexes[i-1]).item())
        return lengths

    def _scatter(self, h, indexes, batch_shape):
        indexes = indexes.nonzero().expand_as(h)
        hidden_char_batch = torch.full((prod(batch_shape[:-1]), h.shape[1]),
                                       self.pad_token_idx, dtype=h.dtype, device=self.device)
        hidden_char_batch.scatter_(0, indexes, h)
        hidden_char_batch = hidden_char_batch.reshape(
            *batch_shape[:-1], h.shape[1])
        return hidden_char_batch

    def _create_join_array(self, src_bert):
        bpe_join = []
        for sentence in src_bert:
            bpe_join.append([])
            for bpe_token in self.bert_tokenizer.convert_ids_to_tokens(sentence):
                # When you remove [CLS] you are effectiely accounting for the fact that
                # that input and output of decoder are shifted by 1
                if bpe_token in ['[CLS]', '[SEP]', '[PAD]']:
                    bpe_join[-1].append(2)
                elif bpe_token.startswith('##'):
                    bpe_join[-1].append(1)
                else:
                    bpe_join[-1].append(0)
        return bpe_join

    def _map_bpe_tokens_to_tgt(self, logits_words, src_bert):
        bpe_join = self._create_join_array(src_bert)
        logits_words_ = []
        for i in range(src_bert.shape[0]):
            logits_words_.append([])
            j = src_bert.shape[1] - 1
            while j >= 0:
                if bpe_join[i][j] == 2:
                    j -= 1
                elif not bpe_join[i][j]:
                    logits_words_[-1].append(logits_words[i][j])
                    j -= 1
                else:
                    incomplete_word = logits_words[i][j]
                    while bpe_join[i][j] and j >= 0:
                        j -= 1
                        incomplete_word = incomplete_word + logits_words[i][j]
                    logits_words_[-1].append(incomplete_word)
                    j -= 1
            logits_words_[-1].reverse()
            logits_words_[-1] = torch.stack(logits_words_[-1])
            logits_words_[-1] = torch.cat(
                [logits_words_[-1], torch.zeros(logits_words.shape[1:], device=self.device)], dim=0)[:logits_words.size(1) + 1, :]
        logits_words = torch.stack(logits_words_, dim=0).permute(1, 0, 2)
        return logits_words
