from torchcrf import CRF
from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 num_layers: int = 1,
                 bert_emb_dim: int = 0,
                 tagger_context_dim: int = 0,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.bert_emb_dim = bert_emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim + bert_emb_dim + tagger_context_dim,
                           hidden_size=enc_hid_dim,
                           bidirectional=True,
                           num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.enc_to_dec = nn.Linear(self.enc_hid_dim*2, self.dec_hid_dim)

    def forward(self,
                src: Tensor,
                lengths,
                bert_encodings=None,
                hidden_char=None,
                tagger_context: Optional[Tensor] = None,
                integrated_gradients: bool = False) -> Tuple[Tensor]:
        """- src: [src_len, batch_size]
        - integrated_gradients: if True, then src is a batch of already embedded inputs
        - 2 is the number of directions (forward/backward)"""
        src = self.embedding(src) if not integrated_gradients else src
        if hidden_char is not None:
            src = torch.cat([src, hidden_char], dim=-1)
        if bert_encodings is not None and self.bert_emb_dim:
            src = torch.cat([src, bert_encodings.repeat(src.size(0), 1, 1)], dim=-1)
            bert_encodings = None
        if tagger_context is not None:
            src = torch.cat([src, tagger_context.repeat(src.size(0), 1, 1)], dim=-1)
        # embedded = [src_len, batch_size, cle_dim]
        embedded = self.dropout(src)
        # outputs: [src_len, batch_size, 2*rnn_dim] final-layer hidden states
        # hidden: [2*rnn_layers, batch_size, rnn_dim] is the final hidden state of each layer-direction
        # hidden: [forward_1, backward_1, forward_2, backward_2, ...]
        embedded_packed = pack_padded_sequence(
            embedded, lengths, enforce_sorted=False)

        if bert_encodings is not None:
            bert_encodings = torch.cat(2*[bert_encodings])
            bert_encodings = (torch.cat([bert_encodings.clone()] + [torch.zeros_like(bert_encodings) for i in range(self.num_layers - 1)]),
                              torch.cat([bert_encodings.clone()] + [torch.zeros_like(bert_encodings) for i in range(self.num_layers - 1)]))

        outputs, hidden = self.rnn(embedded_packed, bert_encodings)
        outputs, _ = pad_packed_sequence(outputs)
        # hidden[-2, :, :]: [1, batch_size, rnn_dim]
        h = hidden[0][0, :, :] + hidden[0][1, :, :]
        c = hidden[1][0, :, :] + hidden[1][1, :, :]
        for i in range(self.num_layers - 1):
            h = torch.stack([h, hidden[0][i + 2, :, :] + hidden[0][i + 3, :, :]])
            c = torch.stack([c, hidden[1][i + 2, :, :] + hidden[1][i + 3, :, :]])
        hidden = (h, c)
        #outputs = [src_len, batch_size, rnn_dim * 2] because 2 directions
        #hidden = [batch_size, rnn_dim]
        return outputs, hidden, embedded


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:
        """decoder_hidden: [batch_size, rnn_dim]
        encoder_outputs: [src_len, batch_size, 2*rnn_dim]"""
        src_len = encoder_outputs.shape[0]
        # decoder_hidden: [batch_size, 1, dec rnn_dim]
        decoder_hidden = torch.sum(decoder_hidden[0], dim=0).unsqueeze(1)
        # repeated_hidden: [batch_size, src_len, rnn_dim]
        repeated_decoder_hidden = decoder_hidden.repeat(1, src_len, 1)
        # encoder_outputs: [batch_size, src_len, 2*rnn_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # decoder_encoder: [batch_size, src_len, 3*rnn_dim]
        decoder_encoder = torch.cat(
            (repeated_decoder_hidden, encoder_outputs), dim=2)
        # energy: [batch_size, src_len, attn_dim]
        energy = torch.tanh(self.attn(decoder_encoder))
        # attention: [batch_size, src_len]
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attention: nn.Module,
                 num_layers: int = 1,
                 char_emb_dim: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(input_size=2*enc_hid_dim + emb_dim + char_emb_dim,
                           hidden_size=dec_hid_dim,
                           num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:
        """decoder_hidden: [batch_size, rnn_dim]
        encoder_outputs: [src_len, batch_size, 2*rnn_dim]"""
        # a: [batch_size, src_len]
        a = self.attention(decoder_hidden, encoder_outputs)
        # a: [batch_size, 1, src_len]
        a = a.unsqueeze(1)
        # encoder_outputs: [batch_size, src_len, 2*rnn_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # weighted_encoder_rep: [batch_size, 1, 2*rnn_dim] batch-matrix-matrix product
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        # weighted_encoder_rep: [1, batch_size, 2*rnn_dim]
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep, a

    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor,
                hidden_tgt_char: Tensor = None) -> Tuple[Tensor]:
        """ input: [batch_size] Note: "one character at a time"
            decoder_hidden: [batch_size, dec rnn_dim]
            encoder_outputs: [src_len, batch_size, 2*rnn_dim]"""
        # input: [1, batch_size]
        input = input.unsqueeze(0)
        # embedded: [1, batch_size, cle_dim]
        embedded = self.embedding(input)
        if hidden_tgt_char is not None:
            embedded = torch.cat([embedded, hidden_tgt_char], dim=-1)
        embedded = self.dropout(embedded)
        # weighted_encoder_rep: [1, batch_size, 2*rnn_dim]
        weighted_encoder_rep, attn_weights = self._weighted_encoder_rep(decoder_hidden,
                                                                        encoder_outputs)
        # rnn_input: [1, batch_size, 2*rnn_dim + cle_dim]
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        # output: [1, batch_size, rnn_dim] tgt_len = 1 (decoding steps)
        # decoder_hidden: [1, batch_size, rnn_dim]
        # output == decoder_hidden
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
        # embedded: [batch_size, cle_dim]
        # output: [batch_size, rnn_dim]
        embedded, output = embedded.squeeze(0), output.squeeze(0)
        # weighted_encoder_rep: [batch_size, 2*rnn_dim]
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        # output = [batch_size, tgt_vocab]
        output = torch.cat((output,
                            weighted_encoder_rep,
                            embedded), dim=1)
        return output, decoder_hidden, attn_weights.squeeze(1)


class Generator(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # |x| = (batch_size, length, hidden_size)
        y = self.softmax(self.output(x))
        # |y| = (batch_size, length, output_size)
        # Return log-probability instead of just probability.
        return y


class BiLSTM_CRF(nn.Module):

    def __init__(self, input_dim, num_tags, bert_emb_dim, rnn_dim, num_layers, dropout, pad_index, device):
        super(BiLSTM_CRF, self).__init__()
        self.input_dim = input_dim
        self.num_tags = num_tags
        self.bert_emb_dim = bert_emb_dim
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.lstm = nn.LSTM(input_size=input_dim + bert_emb_dim,
                            hidden_size=rnn_dim,
                            bidirectional=True,
                            num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.features_crf = {}
        for f, num in num_tags.items():
            setattr(self, f'{f}_crf', CRF(num_tags=num))
            self.features_crf[f] = getattr(self, f'{f}_crf')
            
        # Maps the output of the LSTM into tag space.
        self.f1_to_tag = {}
        for f, num in num_tags.items():
            setattr(self, f'{f}_f1_to_tag', nn.Linear(rnn_dim * 2, num))
            self.f1_to_tag[f] = getattr(self, f'{f}_f1_to_tag')

    def forward(self,
                src_segments,
                features_labels,
                decode=False,
                output_loss=True,
                use_crf=True,
                bert_encodings=None):
        """- src: [src_len, batch_size]
        - integrated_gradients: if True, then src is a batch of already embedded inputs
        - 2 is the number of directions (forward/backward)"""
        if bert_encodings is not None and self.bert_emb_dim:
            src_segments = torch.cat(
                [src_segments, bert_encodings.repeat(src_segments.size(0), 1, 1)], dim=-1)
            bert_encodings = None
        # embedded = [src_len, batch_size, cle_dim]
        embedded = self.dropout(src_segments)
        # outputs: [src_len, batch_size, 2*rnn_dim] final-layer hidden states
        # hidden: [2*rnn_layers, batch_size, rnn_dim] is the final hidden state of each layer-direction
        # hidden: [forward_1, backward_1, forward_2, backward_2, ...]

        if bert_encodings is not None:
            bert_encodings = torch.cat(2*[bert_encodings])
            bert_encodings = (torch.cat([bert_encodings.clone()] + [torch.zeros_like(bert_encodings) for i in range(self.num_layers - 1)]),
                              torch.cat([bert_encodings.clone()] + [torch.zeros_like(bert_encodings) for i in range(self.num_layers - 1)]))

        outputs, _ = self.lstm(embedded, bert_encodings)
        #outputs = [src_len, batch_size, rnn_dim * 2] because 2 directions
        features_lstm_feats = {f: self.f1_to_tag[f](outputs) for f in self.num_tags}
        if use_crf:
            outputs, loss = None, None
            if decode:
                outputs = {f: crf.decode(features_lstm_feats[f])
                           for f, crf in self.features_crf.items()}
            if output_loss:
                loss = - sum(crf(features_lstm_feats[f], features_labels[f])
                             for f, crf in self.features_crf.items())
            output = dict(features_lstm_feats=features_lstm_feats, loss=loss,
                          features_outputs=outputs, features_labels=features_labels)

        else:
            output = dict(features_lstm_feats=features_lstm_feats,
                          loss=None, features_outputs=None, features_labels=None)
        return output


class TaxonomyTagger(nn.Module):
    def __init__(self, input_dim, vocab):
        super().__init__()
        self.vocab = vocab
        self.input_dim = input_dim
        self.num_tags = len(vocab.taxonomy.word2id)
        self.hidden2char = nn.Linear(input_dim, len(vocab.tgt.id2char))
        self.taxonomy_categories_layers = {}
        for tag_id in range(len(vocab.taxonomy.taxonomy2id)):
            setattr(self, f'taxonomy_tagger_{tag_id}', nn.Linear(
                input_dim, self.num_tags))
            self.taxonomy_categories_layers[tag_id] = getattr(
                self, f'taxonomy_tagger_{tag_id}')

    def forward(self,
                encoder_outputs: Tensor,
                decoder_outputs: Tensor):
        # encoder_outputs = self.hidden2char(encoder_outputs)
        encoder_outputs = encoder_outputs.sum(0)
        # decoder_outputs = decoder_outputs.sum(0)
        # tagger_input = torch.cat([encoder_outputs, decoder_outputs], dim=-1)
        tagger_input = encoder_outputs
        self.taxonomy_categories_outputs = {
            tag_id: layer(tagger_input) for tag_id, layer in self.taxonomy_categories_layers.items()
        }
        return self.taxonomy_categories_outputs
