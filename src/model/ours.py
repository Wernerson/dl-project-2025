import random

import lightning as L
import torch


class OurModel(L.LightningModule):

    def __init__(self, net, optimizer, criterion):
        super(OurModel, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, x):
        return self.net(x)

    def mask(self, x_0, t):
        T = x_0.shape[0] * x_0.shape[1]
        p = torch.randperm(T, device=x_0.device).reshape(x_0.shape)
        return p > t

    def noise(self, x_0, t):
        m = self.mask(x_0, t)
        return m * x_0

    def _sample_forward_loss(self, x_0):
        x_0 = x_0.squeeze(0).float()
        # sample time stamp in denoising process
        T = x_0.shape[0] * x_0.shape[1]
        t = random.randint(0, T)

        # noise / mask the sample
        x_t = self.noise(x_0, t)

        # denoise / unmask the sample
        x_0_hat = self(x_t)

        # loss
        loss = self.criterion(torch.round(x_0_hat), x_0)
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
            x_t = m * self(x_t)
        x_t[x_t < 0] = 0
        return torch.round(x_t).long()

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())
