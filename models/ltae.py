"""
Lightweight Temporal Attention Encoder module
We modify the original LTAE to support variable time series lengths and domain-specific batch normalization

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

from models.layers import LinearLayer, get_positional_encoding


class LTAE(nn.Module):
    def __init__(self,
                 in_channels=128,
                 n_head=16,
                 d_k=8,
                 n_neurons=[256, 128],
                 dropout=0.2,
                 d_model=256,
                 T=1000,
                 max_temporal_shift=100,
                 max_position=365,
                 ):
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
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)

        """

        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.n_neurons = copy.deepcopy(n_neurons)
        self.max_temporal_shift = max_temporal_shift

        if d_model is not None:
            self.d_model = d_model
            # self.inconv = nn.Conv1d(in_channels, d_model, 1)
            self.inconv = LinearLayer(in_channels, d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.positional_enc = nn.Embedding.from_pretrained(get_positional_encoding(max_position + 2*max_temporal_shift, self.d_model, T=T), freeze=True)
        # not splitting positional encoding seems to adapt better
        # sin_tab = get_positional_encoding(max_position + 2*max_temporal_shift, self.d_model // n_head, T=T)
        # self.positional_enc = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(n_head)], dim=1), freeze=True)

        # self.inlayernorm = nn.LayerNorm(self.in_channels)
        # self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)

        assert (self.n_neurons[0] == self.d_model)

        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.append(LinearLayer(self.n_neurons[i], self.n_neurons[i + 1]))

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions, return_att=False):
        if self.inconv is not None:
            x = self.inconv(x)
        enc_output = x + self.positional_enc(positions + self.max_temporal_shift)

        enc_output, attn = self.attention_heads(enc_output)

        enc_output = self.dropout(self.mlp(enc_output))

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

        self.key = nn.Linear(d_in, n_head * d_k)
        self.query = nn.Parameter(torch.zeros(n_head, d_k)).requires_grad_(True)
        nn.init.normal_(self.query, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.temperature = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        # Slightly more efficient re-implementation of LTAE
        B, T, C = x.size()
        q = self.query.repeat(B, 1, 1, 1).transpose(1, 2)  # (nh, hs) -> (B, nh, 1, d_k)
        k = self.key(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, nh, T, d_k)
        v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # self-attend; (B, nh, 1, d_k) x (B, nh, d_k, T) -> (B, nh, 1, T)
        att = (q @ k.transpose(-2, -1)) / self.temperature
        att = self.softmax(att)
        att = self.dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, C)
        return y, att
