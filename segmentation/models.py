from numpy import uint8
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchcrf import CRF


class BiLSTM_CRF(nn.Module):

    def __init__(self, input_dim, num_tags, emb_dim, bert_emb_dim, rnn_dim, num_layers, dropout, pad_index, device):
        super(BiLSTM_CRF, self).__init__()
        self.input_dim = input_dim
        self.num_tags = num_tags
        self.emb_dim = emb_dim
        self.bert_emb_dim = bert_emb_dim
        self.num_layers = num_layers
        self.pad_index = pad_index
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim + bert_emb_dim,
                           hidden_size=rnn_dim,
                           bidirectional=True,
                           num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.crf_layer = CRF(num_tags=num_tags)

        # Maps the output of the LSTM into tag space.
        self.f0 = nn.Linear(rnn_dim * 2, rnn_dim)
        self.f1 = nn.Linear(rnn_dim, rnn_dim // 2)
        self.f1_to_tag = nn.Linear(rnn_dim // 2, num_tags)

    def forward(self,
                src,
                tgt,
                lengths,
                decode=False,
                use_crf=True,
                bert_encodings=None):
        """- src: [src_len, batch_size]
        - integrated_gradients: if True, then src is a batch of already embedded inputs
        - 2 is the number of directions (forward/backward)"""
        src = self.embedding(src)
        if bert_encodings is not None and self.bert_emb_dim:
            src = torch.cat([src, bert_encodings.repeat(src.size(0), 1, 1)], dim=-1)
            bert_encodings = None
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

        outputs, _ = self.lstm(embedded_packed, bert_encodings)
        outputs, _ = pad_packed_sequence(outputs, total_length=src.size(0), padding_value=self.pad_index)
        #outputs = [src_len, batch_size, rnn_dim * 2] because 2 directions
        lstm_feats = self.f1_to_tag(self.f1(self.f0(outputs)))
        if use_crf:
            mask = [[True if i < l else False for i in range(src.size(0))] for l in lengths]
            mask = torch.tensor(mask, dtype=torch.uint8, device=self.device).permute(1, 0)
            outputs = None
            if decode:
                outputs = self.crf_layer.decode(lstm_feats, mask)
            loss = - self.crf_layer(lstm_feats, tgt, mask=mask)
            output = dict(lstm_feats=lstm_feats, loss=loss, outputs=outputs)

        else:
            output = dict(lstm_feats=lstm_feats, loss=None, outputs=None)
        return output
