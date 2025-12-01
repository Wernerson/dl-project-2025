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
        p = torch.randperm(T, device=x_0.device).reshape(seq_len, dim)
        return (p > t).expand(batch_size, -1, -1)

    def noise(self, x_0, t):
        m = self.mask(x_0, t)
        return m * x_0

    def _sample_forward_loss(self, x_0):
        # sample time stamp in denoising process
        T = x_0.shape[1] * x_0.shape[2]
        t = random.randint(0, T)

        # noise / mask the sample
        x_t = self.noise(x_0, t)

        # denoise / unmask the sample
        x_0_hat = self(x_t.float())

        # loss
        loss = self.criterion(x_0_hat, Octuple.encode(x_0).float())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._sample_forward_loss(batch["input_ids"])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._sample_forward_loss(batch["input_ids"])
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._sample_forward_loss(batch["input_ids"])
        self.log("test/loss", loss)
        return loss

    def predict_step(self, x):
        x = x[0]
        n = x.item()
        T = 8 * n
        x_t = torch.zeros((n, 8), device=x.device)
        for t in reversed(range(T)):
            m = self.mask(x_t, t)
            x_t = m * Octuple.decode(self(x_t.float()))
        return x_t

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())
