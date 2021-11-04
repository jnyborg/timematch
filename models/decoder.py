import torch.nn as nn
from models.layers import LinearLayer


def get_decoder(n_neurons, n_classes):
    """Returns an MLP with the layer widths specified in n_neurons.
    Every linear layer but the last one is followed by BatchNorm + ReLu

    args:
        n_neurons (list): List of int that specifies the width and length of the MLP.
        n_classes (int): Output size
    """
    layers = []
    for i in range(len(n_neurons) - 1):
        layers.append(LinearLayer(n_neurons[i], n_neurons[i + 1]))
    layers.append(nn.Linear(n_neurons[-1], n_classes))
    m = nn.Sequential(*layers)
    return m
