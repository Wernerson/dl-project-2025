import random

import lightning as L
import torch
from net.nn import Octuple


class OurModel(L.LightningModule):

    def __init__(self, net, optimizer, criterion):
        super(OurModel, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, x):
        return self.net(x)

    def mask(self, x_0, t):
        batch_size, seq_len, dim = x_0.shape
        T = seq_len * dim
        p = torch.randperm(T, device=self.device).reshape(seq_len, dim)
        return (p > t).expand(batch_size, -1, -1)

    def _sample_forward_loss(self, x_0):
        # sample time stamp in denoising process
        T = x_0.shape[1] * x_0.shape[2]
        t = random.randint(0, T)

        # noise / mask the sample
        m = self.mask(x_0, t)
        x_t = m * x_0

        # denoise / unmask the sample
        x_0_hat = self(x_t.float())

        # loss
        padding = x_0 == 0
        m = ~padding & ~m  # only care about masked & non-padded
        m = Octuple.encode_mask(m)  # adjust loss
        y = x_0.clone()
        y[padding] = 1  # don't care about padding, but need a valid value
        y = Octuple.encode(y).float()
        loss = self.criterion(m * x_0_hat, m * y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._sample_forward_loss(batch["input_ids"])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._sample_forward_loss(batch["input_ids"])
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._sample_forward_loss(batch["input_ids"])
        self.log("test/loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # input is currently irrelevant
        n = random.randint(10, 20)
        T = 4 * n
        x_t = torch.zeros((1, n, 4), device=self.device)
        for t in reversed(range(T)):
            m = self.mask(x_t, t)
            x_t = m * Octuple.decode(self(x_t.float()))
        x_t = Octuple.decode(self(x_t.float()))
        return x_t

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())
