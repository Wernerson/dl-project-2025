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
        _, seq_len, _ = x_0.shape
        p = torch.randperm(seq_len, device=self.device)
        return p < t

    def _sample_forward_loss(self, x_0):
        # sample time stamp in denoising process
        batch_size, seq_len, dim = x_0.shape
        t = random.randint(1, seq_len)

        # mask the sample & create padding mask
        padding = x_0[:, :, 0] == 0
        mask = self.mask(x_0, t)
        x_t = (~mask).unsqueeze(1).expand(-1, 4) * x_0

        # denoise / unmask the sample
        x_0_hat = self.net(x_t, mask, padding)

        # loss
        loss = self.criterion(x_0, x_0_hat, mask, padding)
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
        x_t = torch.zeros((1, n, 4), device=self.device)
        for t in reversed(range(n)):
            mask = self.mask(x_t, t)
            x_t = self.net.predict(x_t, mask)
        return x_t

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())
