import math
import torch
import torch.nn as nn

class Mark(nn.Module):
    """
    Simplified version of PSE+LTAE by Garnot et al (https://github.com/VSainteuf/lightweight-temporal-attention-pytorch)
    - only batch norm (no layer norm)
    - no bias before batch norm
    - support for TimeMatch temporal shift
    """
    def __init__(self, input_dim=10, n_classes=20, base_dim=64, temporal_scale=4, return_sequences=False, n_head=16, dropout=0.2, attn_dropout=0.1):
        super(Mark, self).__init__()
        self.predict_at_every_step = return_sequences
        self.n_classes = n_classes

        self.base_dim = base_dim
        self.temporal_dim = temporal_scale * self.base_dim
        self.spatial_encoder = PixelSetEncoder(input_dim,
                hidden_dim=self.base_dim, pooling='mean_std')

        self.temporal_encoder = LightWeightTransformer(d_in=2*base_dim, d_v=self.temporal_dim,
                d_k=self.temporal_dim // 2, d_out=self.temporal_dim // 2,
                n_head=n_head, return_sequences=return_sequences,
                max_seq_length=366, max_temporal_shift=100, out_pdrop=dropout,
                attn_pdrop=attn_dropout)

        self.temporal_feat_dim = self.temporal_dim // 2

        self.classifier = get_decoder(self.temporal_dim // 2, n_classes)

    def forward(self, pixels, valid_pixels, positions, return_feats=False):
        """
         Args:
            pixels: Tensor of shape (batch_size, sequence_length, n_channels, n_pixels)
            valid_pixels: Tensor of shape (batch_size, sequence_length, n_pixels) defining valid pixels
            positions: Tensor of shape (batch_size, sequence_length) defining time step positions
        """

        spatial_feats = self.spatial_encoder(pixels, valid_pixels)
        temporal_feats = self.temporal_encoder(spatial_feats, positions)
        logits = self.classifier(temporal_feats)

        if return_feats:
            return logits, temporal_feats
        else:
            return logits

    def param_ratio(self):
        total = get_num_trainable_params(self)
        s = get_num_trainable_params(self.spatial_encoder)
        t = get_num_trainable_params(self.temporal_encoder)
        c = get_num_trainable_params(self.classifier)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))

    def output_dim(self):
        return self.temporal_feat_dim


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def LinearBatchNormRelu(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=False),
        BatchNorm1d(out_dim),
        nn.ReLU()
    )

class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if x.dim() == 3:  # (B, S, C)
            # When 3D, nn.BatchNorm1d expects channels first
            return super().forward(x.transpose(1, 2)).transpose(1, 2)
        else:  # (B, C)
            return super().forward(x)


class PixelSetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, pooling='mean_std'):
        super().__init__()

        self.in_features = input_dim
        self.mlp_in = nn.Sequential(
                LinearBatchNormRelu(input_dim, hidden_dim // 2),
                LinearBatchNormRelu(hidden_dim // 2, hidden_dim)
        )

        self.pooling_methods = {
            'mean': self.masked_mean,
            'std': self.masked_std,
            'max': self.maximum,
            'min': self.minimum
        }
        self.pooling = [self.pooling_methods[m] for m in pooling.split('_')]

        hidden_dim = len(self.pooling) * hidden_dim
        self.mlp_out = nn.Sequential(
                LinearBatchNormRelu(hidden_dim, hidden_dim),
        )


    def forward(self, x, mask):
        B, T, C, S = x.shape

        # combine time and batch dimension
        mask = mask.view(B * T, -1)
        x = x.view(B * T, C, S).transpose(1, 2)  # (B*T, S, C)

        x = self.mlp_in(x)
        x = torch.cat([m(x, mask) for m in self.pooling], dim=1)
        x = self.mlp_out(x)

        # split batch and time
        x = x.view(B, T, -1)  # (B, T, C)
        return x


    def masked_mean(self, x, mask):
        # x = (B*T, S, C), mask = (B*T, S)
        mask = mask.unsqueeze(-1)
        out = x * mask
        out = out.sum(dim=1) / mask.sum(dim=1)
        return out


    def masked_std(self, x, mask):
        # x = (B*T, S, C)
        mean = self.masked_mean(x, mask)

        mean = mean.unsqueeze(1)       # (B*T, 1, C)
        mask = mask.unsqueeze(-1)  # (B*T, S, 1)

        out = x - mean

        out = out * mask
        d = mask.sum(dim=1)
        d[d == 1] = 2

        out = (out ** 2).sum(dim=1) / (d - 1)
        out = torch.sqrt(out + 10e-32)  # To ensure differentiability

        return out

    def maximum(self, x, mask):
        return x.max(dim=1)

    def minimum(self, x, mask):
        return x.min(dim=1)


class LightWeightTransformer(nn.Module):
    def __init__(self, d_in, d_v, d_k, d_out, n_head, return_sequences=False, max_seq_length=366, out_pdrop=0.2, attn_pdrop=0.1, max_temporal_shift=100, T=1000):
        super().__init__()

        valid_positions = max_seq_length + 2*max_temporal_shift  # 2x for positive and negative shifts
        assert valid_positions <= T
        self.position_enc = nn.Embedding.from_pretrained(self.get_positional_encoding(max_seq_length + 2*max_temporal_shift, d_v, T=T), freeze=True)
        self.positions_offset = max_temporal_shift

        self.in_mlp = LinearBatchNormRelu(d_in, d_v)
        self.attn = LightweightSelfAttention(d_v, d_k, n_head, attn_pdrop, max_seq_length, return_sequences)
        self.out_mlp = LinearBatchNormRelu(d_v, d_out)
        self.dropout = nn.Dropout(out_pdrop)


    def forward(self, inputs, positions=None):
        x = self.in_mlp(inputs)
        x = x + self.position_enc(positions + self.positions_offset)
        x = self.attn(x)
        x = self.out_mlp(x)
        x = self.dropout(x)
        return x

    def get_positional_encoding(self, max_len, d_model, T=1000.0):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(T) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class LightweightSelfAttention(nn.Module):
    """
    Lightweight self-attention layer with query-as-parameter and no value and output projection.
    """

    def __init__(self, d_v, d_k, n_head, attn_pdrop, max_seq_length, return_sequences):
        super().__init__()
        assert d_v % n_head == 0
        assert d_k % n_head == 0

        self.key = nn.Linear(d_v, d_k)
        self.query = nn.Parameter(torch.zeros(n_head, d_k // n_head))
        nn.init.normal_(self.query, mean=0, std=math.sqrt(2.0 / (d_k // n_head)))

        self.temperature = math.sqrt(d_k // n_head)
        self.softmax = nn.Softmax(dim=-1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if return_sequences:
            self.register_buffer("mask", 
                    torch.tril(torch.ones(max_seq_length, max_seq_length)).view(1, 1, max_seq_length, max_seq_length))
        self.n_head = n_head
        self.d_k = d_k
        self.return_sequences = return_sequences

    def forward(self, x, return_att=False):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head dim forward to the batch dim
        if self.return_sequences:
            q = self.query.repeat(B, T, 1, 1).transpose(1, 2)  # (nh, hs) -> (B, nh, T, hs)
        else:
            q = self.query.repeat(B, 1, 1, 1).transpose(1, 2)  # (nh, hs) -> (B, nh, 1, hs)
        k = self.key(x).view(B, T, self.n_head, self.d_k // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) / self.temperature
        if self.return_sequences:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = self.softmax(att)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        if self.return_sequences:
            y = y.transpose(1, 2).contiguous().view(B, T, C)
        else:
            y = y.transpose(1, 2).contiguous().view(B, C)

        if return_att:
            return y, att
        else:
            return y


def get_decoder(in_dim, n_classes):
        return nn.Sequential(
                LinearBatchNormRelu(in_dim, in_dim//2),
                LinearBatchNormRelu(in_dim//2, in_dim//4),
                nn.Linear(in_dim//4, n_classes)
        )

