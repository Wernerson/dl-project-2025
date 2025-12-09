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
        self.out_dim = out_dim

        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
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
            nn.Linear(hidden_dim, in_dim * out_dim),
            nn.ReLU()
        )

    def forward(self, x, padding=None):
        batch_len, seq_len, dim = x.shape
        x = x.float()
        x += sinusoidal_encodings(seq_len, dim, x.device)
        x = self.input(x)
        x = self.transformer(x, src_key_padding_mask=padding)
        x = self.output(x)
        x = x.reshape(batch_len, seq_len, self.in_dim, self.out_dim)
        x = F.softmax(x, dim=-1)
        return x

    @staticmethod
    def loss(x_0, x_0_hat, mask, padding):
        _, _, _, dim = x_0_hat.shape
        mask = mask & ~padding
        y = x_0.clone()
        y[~padding] -= 1
        y = F.one_hot(y, dim)
        loss = (y - x_0_hat) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss category
        return (mask.unsqueeze(-1) * loss).sum() / mask.sum()

    def predict(self, x_t):
        x = self.forward(x_t)
        x = x.argmax(dim=-1) + 1
        return x
