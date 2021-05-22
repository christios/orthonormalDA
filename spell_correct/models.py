from typing import Tuple

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
                 char_emb_dim: int = 0,
                 num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + char_emb_dim,
                           enc_hid_dim, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.enc_to_dec = nn.Linear(self.enc_hid_dim*2, self.dec_hid_dim)

    def forward(self,
                src: Tensor,
                lengths,
                bert_encodings=None,
                hidden_char=None,
                integrated_gradients: bool = False) -> Tuple[Tensor]:
        """- src: [src_len, batch_size]
        - integrated_gradients: if True, then src is a batch of already embedded inputs
        - 2 is the number of directions (forward/backward)"""
        src = self.embedding(src) if not integrated_gradients else src
        if hidden_char is not None:
            src = torch.cat([src, hidden_char], dim=-1)
        # embedded = [src_len, batch_size, cle_dim]
        embedded = self.dropout(src)
        # outputs: [src_len, batch_size, 2*rnn_dim] final-layer hidden states
        # hidden: [2*rnn_layers, batch_size, rnn_dim] is the final hidden state of each layer-direction
        # hidden: [forward_1, backward_1, forward_2, backward_2, ...]
        embedded_packed = pack_padded_sequence(
            embedded, lengths, enforce_sorted=False)

        if bert_encodings is not None:
            bert_encodings = torch.cat(2*[bert_encodings])
            bert_encodings = (bert_encodings.clone(),
                              bert_encodings.clone())

        outputs, hidden = self.rnn(embedded_packed, bert_encodings)
        outputs, _ = pad_packed_sequence(outputs)
        # hidden[-2, :, :]: [1, batch_size, rnn_dim]
        # backward_forward: [batch_size, rnn_dim * 2]
        h = hidden[0][0, :, :] + hidden[0][1, :, :]
        c = hidden[1][0, :, :] + hidden[1][1, :, :]
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
        decoder_hidden = decoder_hidden[0].unsqueeze(1)
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
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim +
                           char_emb_dim, dec_hid_dim)
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
        decoder_hidden = (decoder_hidden[0].unsqueeze(
            0), decoder_hidden[1].unsqueeze(0))
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
        decoder_hidden = (decoder_hidden[0].squeeze(
            0), decoder_hidden[1].squeeze(0))
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
