import torch.nn as nn
from models.tae import get_positional_encoding
import copy




class GRU(nn.Module):
    """
    Gated Recurrent Unit
    """
    def __init__(self, in_channels=128, hidden_dim=128, max_position=365, max_temporal_shift=100):
        super(GRU, self).__init__()
        self.name = 'GRU_h{}'.format(hidden_dim)
        self.gru_cell = nn.GRU(input_size=in_channels, hidden_size=hidden_dim, batch_first=True)

        self.max_temporal_shift = max_temporal_shift
        self.position_enc = nn.Embedding.from_pretrained(
            get_positional_encoding(max_position + 2*max_temporal_shift, self.in_channels, T=1000),
            freeze=True)

    def forward(self, input, positions):
        enc_output = input + self.position_enc(positions + self.max_temporal_shift)
        out, _ = self.gru_cell(enc_output)
        return out[:,-1,:]



class TempConv(nn.Module):
    """
    Temporal CNN
    """
    def __init__(self, input_size, nker, seq_len, nfc, max_position=365, max_temporal_shift=100):
        super(TempConv, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.name = 'TempCNN_'

        self.nker = copy.deepcopy(nker)
        self.nfc = copy.deepcopy(nfc)
        self.name += '|'.join(list(map(str, self.nker)))

        if self.nfc is not None:
            self.name += 'FC'
            self.name += '|'.join(list(map(str, self.nfc)))

        conv_layers = []
        self.nker.insert(0, input_size)
        for i in range(len(self.nker) - 1):
            conv_layers.extend([
                nn.Conv1d(self.nker[i], self.nker[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm1d(self.nker[i + 1]),
                nn.ReLU()
            ])
        self.conv1d = nn.Sequential(*conv_layers)

        self.nfc.insert(0, self.nker[-1] * seq_len)
        lin_layers = []
        for i in range(len(self.nfc) - 1):
            lin_layers.extend([
                nn.Linear(self.nfc[i], self.nfc[i + 1]),
                nn.BatchNorm1d(self.nfc[i + 1]),
                nn.ReLU()
            ])
        self.linear = nn.Sequential(*lin_layers)

        self.max_temporal_shift = max_temporal_shift
        self.position_enc = nn.Embedding.from_pretrained(
            get_positional_encoding(max_position + 2*max_temporal_shift, self.in_channels, T=1000),
            freeze=True)


    def forward(self, input, positions):
        enc_output = input + self.position_enc(positions + self.max_temporal_shift)
        out = self.conv1d(enc_output.permute(0, 2, 1))
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out
