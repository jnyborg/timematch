"""
Pixel-Set encoder module

author: Vivien Sainte Fare Garnot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.layers import LinearLayer


class PixelSetEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        mlp1=[10, 32, 64],
        pooling="mean_std",
        mlp2=[64, 128],
        with_extra=True,
        extra_size=4,
    ):
        """
        Pixel-set encoder.
        Args:
            input_dim (int): Number of channels of the input tensors
            mlp1 (list):  Dimensions of the successive feature spaces of MLP1
            pooling (str): Pixel-embedding pooling strategy, can be chosen in ('mean','std','max,'min')
                or any underscore-separated combination thereof.
            mlp2 (list): Dimensions of the successive feature spaces of MLP2
            with_extra (bool): Whether additional pre-computed features are passed between the two MLPs
            extra_size (int, optional): Number of channels of the additional features, if any.
        """

        super(PixelSetEncoder, self).__init__()

        self.input_dim = input_dim
        self.mlp1_dim = copy.deepcopy(mlp1)
        self.mlp2_dim = copy.deepcopy(mlp2)
        self.pooling = pooling

        self.with_extra = with_extra
        self.extra_size = extra_size

        self.output_dim = (
            input_dim * len(pooling.split("_"))
            if len(self.mlp2_dim) == 0
            else self.mlp2_dim[-1]
        )

        inter_dim = self.mlp1_dim[-1] * len(pooling.split("_"))
        if self.with_extra:
            inter_dim += self.extra_size

        assert input_dim == mlp1[0]
        assert inter_dim == mlp2[0]
        # Feature extraction
        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(LinearLayer(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        # MLP after pooling
        layers = []
        for i in range(len(self.mlp2_dim) - 1):
            layers.append(LinearLayer(self.mlp2_dim[i], self.mlp2_dim[i + 1]))
        self.mlp2 = nn.Sequential(*layers)

    def forward(self, pixels, mask, extra):
        """
        The input of the PSE is a tuple of tensors as yielded by the PixelSetData class:
          (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
        Pixel-Set : Batch_size x (Sequence length) x Channel x Number of pixels
        Pixel-Mask : Batch_size x (Sequence length) x Number of pixels
        Extra-features : Batch_size x (Sequence length) x Number of features

        If the input tensors have a temporal dimension, it will be combined with the batch dimension so that the
        complete sequences are processed at once. Then the temporal dimension is separated back to produce a tensor of
        shape Batch_size x Sequence length x Embedding dimension
        """
        out = pixels

        batch, temp = out.shape[:2]

        out = out.view(batch * temp, *out.shape[2:]).transpose(1, 2)  # (B*T, S, C)
        mask = mask.view(batch * temp, -1)

        out = self.mlp1(out).transpose(1, 2)
        out = torch.cat(
            [pooling_methods[n](out, mask) for n in self.pooling.split("_")], dim=1
        )

        if self.with_extra:
            extra = extra.unsqueeze(1).repeat(1, temp, 1)
            extra = extra.view(batch * temp, -1)
            out = torch.cat([out, extra], dim=1)
        out = self.mlp2(out)
        out = out.view(batch, temp, -1)
        return out


def masked_mean(x, mask):
    out = x.permute((1, 0, 2))
    out = out * mask
    out = out.sum(dim=-1) / mask.sum(dim=-1)
    out = out.permute((1, 0))
    return out

def masked_std(x, mask):
    m = masked_mean(x, mask)

    out = x.permute((2, 0, 1))
    out = out - m
    out = out.permute((2, 1, 0))

    out = out * mask
    d = mask.sum(dim=-1)
    d[d == 1] = 2

    out = (out ** 2).sum(dim=-1) / (d - 1)
    out = torch.sqrt(out + 10e-32) # To ensure differentiability
    out = out.permute(1, 0)
    return out

def maximum(x, mask):
    return x.max(dim=-1)[0].squeeze()

def minimum(x, mask):
    return x.min(dim=-1)[0].squeeze()


pooling_methods = {
    "mean": masked_mean,
    "std": masked_std,
    "max": maximum,
    "min": minimum,
}
