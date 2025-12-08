import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x, mask, padding=None):
        x = x.float()
        x = self.input(x)
        mask = mask.unsqueeze(0) | mask.unsqueeze(1)
        x = self.transformer(x, mask=mask, src_key_padding_mask=padding)
        x = self.output(x)
        x = x.reshape(x.shape[0], x.shape[1], self.in_dim, self.out_dim)
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

    def predict(self, x_t, mask):
        x = self.forward(x_t, mask)
        x = x.argmax(dim=-1) + 1
        return (~mask).unsqueeze(-1) * x
