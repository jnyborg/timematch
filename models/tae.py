"""
Temporal Attention Encoder module

Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.

paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import math


class TemporalAttentionEncoder(nn.Module):
    def __init__(self,
                 in_channels=128,
                 n_head=4,
                 d_k=32,
                 d_model=None,
                 n_neurons=[512, 128, 128],
                 dropout=0.2,
                 T=1000,
                 max_position=365,
                 max_temporal_shift=100):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            positions (list, optional): List of temporal positions to use instead of position in the sequence
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model

        """

        super(TemporalAttentionEncoder, self).__init__()
        self.in_channels = in_channels
        self.n_neurons = copy.deepcopy(n_neurons)
        self.max_temporal_shift = max_temporal_shift

        self.position_enc = nn.Embedding.from_pretrained(
            get_positional_encoding(max_position + 2*max_temporal_shift, self.in_channels, T=T),
            freeze=True)

        self.inlayernorm = nn.LayerNorm(self.in_channels)

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention(n_head=n_head,
                                                  d_k=d_k,
                                                  d_in=self.d_model)

        assert (self.n_neurons[0] == n_head * self.d_model)
        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend([
                nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                nn.BatchNorm1d(self.n_neurons[i + 1]),
                nn.ReLU()
            ])

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions, return_att=False):
        x = self.inlayernorm(x)
        enc_output = x + self.position_enc(positions + self.max_temporal_shift)

        if self.inconv is not None:
            enc_output = self.inconv(enc_output.permute(0, 2, 1)).permute(0, 2, 1)
        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)
        enc_output = enc_output.permute(1, 0, 2).contiguous().view(x.shape[0], -1)  # Concatenate heads
        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        if return_att:
            return enc_output, attn
        else:
            return enc_output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc2 = nn.Sequential(nn.BatchNorm1d(n_head * d_k),
                                 nn.Linear(n_head * d_k, n_head * d_k))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = self.fc1_q(q).view(sz_b, seq_len, n_head, d_k)
        q = q.mean(dim=1).squeeze()  # MEAN query
        q = self.fc2(q.view(sz_b, n_head * d_k)).view(sz_b, n_head, d_k)
        q = q.permute(1, 0, 2).contiguous().view(n_head * sz_b, d_k)

        k = self.fc1_k(k).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len,
                                                    d_k)  # (n*b) x lk x dk

        v = v.repeat(n_head, 1, 1)  # (n*b) x lv x d_in

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, 1, d_in)
        output = output.squeeze(dim=2)

        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn


def get_positional_encoding(max_len, d_model, T=1000):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(T) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
