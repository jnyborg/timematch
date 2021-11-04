import math
import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x = (B, C) or (B, S, C)
        x = self.linear(x)  # linear expect channels last
        if x.dim() == 3:  
            # BatchNorm1d expects channels first, move to (B, C, S)
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:  # (B, C)
            x = self.norm(x)
        return self.activation(x)


# inspired by https://github.com/pytorch/examples/blob/master/word_language_model/model.py
def get_positional_encoding(max_len, d_model, T=1000.0):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(T) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
