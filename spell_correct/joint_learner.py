from typing import Optional, List, Union, Dict

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from spell_correct.pos_tagger import POSTagger
from spell_correct.spelling_corrector import SpellingCorrector
from spell_correct.models import TaxonomyTagger


class JointLearner(nn.Module):
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

        self.tagger = POSTagger(args,
                               vocab=self.vocab,
                               bert_tokenizer=self.bert_tokenizer,
                               device=self.device).to(self.device)
        self.standardizer = SpellingCorrector(args,
                                              vocab=self.vocab,
                                              bert_tokenizer=self.bert_tokenizer,
                                              device=self.device).to(self.device)
        self.taxonomy_tagger = TaxonomyTagger(input_dim=args.rnn_dim_char * 2,
                                              vocab=self.vocab,
                                              most_common=args.taxonomy_most_common).to(self.device)
        self.grammatical_features_layer = nn.Linear(
            len(args.features.split()), args.features_layer)
        self.args.mode = 'taxonomy'

    def forward(self,
                batch: Dict[str, Union[Tensor,
                                       List[List[int]], List[str], List[int], None]],
                teacher_force: bool = True,
                use_crf=True,
                output_loss=True,
                decode=False):
        output_tagger, output_standardizer, output_taxonomy_tagger = None, None, None
        valid_indexes = None
        if self.args.mode == 'tagger':
            output_tagger = self.tagger(batch, use_crf=use_crf, output_loss=output_loss, decode=decode)
        elif self.args.mode == 'standardizer':
            output_standardizer = self.standardizer(batch, teacher_force=teacher_force)
            output_standardizer = output_standardizer['decoder_outputs']
        elif self.args.mode == 'taxonomy':
            output_tagger = self.tagger(
                batch, use_crf=use_crf, output_loss=False, decode=decode)
            features_lstm_feats = output_tagger['features_lstm_feats']
            features_argmax = {}
            for name, feature in features_lstm_feats.items():
                features_argmax[name] = torch.argmax(
                    F.softmax(feature[self.args.window_size], dim=-1), dim=-1)
                features_argmax[name] = torch.take(
                    feature[self.args.window_size], features_argmax[name]).detach()

            tagger_context = self._map_segment_to_token(features_argmax, batch['segments_per_token'])
            tagger_context = self.grammatical_features_layer(tagger_context)
            output_standardizer = self.standardizer(batch, tagger_context=tagger_context,
                teacher_force=teacher_force, return_encoder_outputs=True, return_valid_indexes=True)
            encoder_outputs = output_standardizer['encoder_outputs']
            decoder_outputs = output_standardizer['decoder_outputs']
            valid_indexes = output_standardizer['valid_indexes']
            output_taxonomy_tagger = self.taxonomy_tagger(
                encoder_outputs=encoder_outputs.detach(), decoder_outputs=decoder_outputs.detach())
            output_standardizer = output_standardizer['decoder_outputs']
        elif self.args.mode == 'joint':
            output_tagger = self.tagger(batch, use_crf=use_crf, output_loss=output_loss, decode=decode)
            features_lstm_feats = output_tagger['features_lstm_feats']
            features_argmax = {}
            for name, feature in features_lstm_feats.items():
                features_argmax[name] = torch.argmax(F.softmax(feature[self.args.window_size], dim=-1), dim=-1)
                features_argmax[name] = torch.take(feature[self.args.window_size], features_argmax[name]).detach()

            tagger_context = self._map_segment_to_token(features_argmax, batch['segments_per_token'])
            tagger_context = self.grammatical_features_layer(tagger_context)
            output_standardizer = self.standardizer(
                batch, tagger_context=tagger_context, teacher_force=teacher_force)
            output_standardizer = output_standardizer['decoder_outputs']
            
        output = dict(tagger=output_tagger,
                      standardizer=output_standardizer,
                      taxonomy=output_taxonomy_tagger,
                      valid_indexes=valid_indexes)
        return output

    def _map_segment_to_token(self, features_argmax, segments_per_token):
        tagger_context = torch.stack(
            list(features_argmax.values())).permute(1, 0)
        k = 0
        tagger_context_ = []
        for sent in segments_per_token:
            tagger_context_.append([])
            for num_segments in sent:
                token_ = 0
                for _ in range(num_segments):
                    token_ += tagger_context[k]
                    k += 1
                tagger_context_[-1].append(token_)
            tagger_context_[-1] = torch.stack(tagger_context_[-1])
            # Pad - add epsilon so that feature values don't get confused with padding in case they are all 0
            tagger_context_[-1] = torch.cat([tagger_context_[-1] + 1e-10, torch.zeros((self.args.max_sent_len + 1 -
                                            tagger_context_[-1].size(0), tagger_context.size(1)), device=self.device)], dim=0)
        tagger_context = torch.stack(tagger_context_)
        # Manipulate matrix to get the same ordering of tokens as in the standardizer
        tagger_context = tagger_context.permute(1, 0, 2)
        tagger_context_debatch = tagger_context.reshape(-1, tagger_context.size(-1))
        tagger_context_indexes_valid = torch.any(tagger_context_debatch.bool(), dim=1)
        tagger_context_valid = tagger_context_debatch[tagger_context_indexes_valid]
        return tagger_context_valid


