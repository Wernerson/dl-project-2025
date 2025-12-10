import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_encodings(seq_len, dim, device):
    N = 10000
    i = torch.arange(0, dim // 2, device=device)
    div_term = torch.exp(-np.log(N) * (2 * i / dim))
    position = torch.arange(seq_len, device=device).unsqueeze(1)

    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class OneHot(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, transformer_layers):
        super(OneHot, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.input = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU()
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=transformer_layers
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x, padding=None):
        batch_len, seq_len, dim = x.shape
        x = x.reshape(batch_len, seq_len * self.in_dim, 1)
        x = x.float()
        x = self.input(x)
        x = x + sinusoidal_encodings(seq_len * self.in_dim, self.hidden_dim, x.device)
        if padding is not None:
            padding = padding.repeat_interleave(self.in_dim, dim=-1)
        x = self.transformer(x, src_key_padding_mask=padding)
        x = self.output(x)
        x = x.reshape(batch_len, seq_len * self.in_dim, self.out_dim)
        x = F.softmax(x, dim=-1)
        return x

    @staticmethod
    def loss(x_0, x_0_hat, mask, padding):
        batch_len, seq_len, dim = x_0_hat.shape
        mask = mask & ~padding
        y = x_0.clone()
        y[~padding] -= 1
        y = y.reshape(batch_len, seq_len)
        y = F.one_hot(y, dim).float()
        mask = mask.repeat_interleave(4, dim=-1)
        if ~mask.any():
            return 0.  # useful was masked
        x = x_0_hat[mask]
        y = y[mask]
        loss = 0.
        for xi, yi in zip(x, y):
            loss = loss + F.cross_entropy(xi, yi)
        return loss

    def predict(self, x_t):
        batch_len, seq_len, dim = x_t.shape
        x = self.forward(x_t)
        x = x.argmax(dim=-1) + 1
        x = x.reshape(batch_len, -1, self.in_dim)
        return x
