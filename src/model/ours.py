import random

import lightning as L


class OurModel(L.LightningModule):

    def __init__(self, net, optimizer, criterion):
        super(OurModel, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, x):
        return self.net(x)

    def noise(self, x_0, t):
        pass  # todo

    def training_step(self, batch, batch_idx):
        x_0 = batch

        # sample time stamp in denoising process
        T = 1000  # todo configure, including max?
        t = random.randint(0, T)

        # noise / mask the sample
        x_t = self.noise(x_0, t)

        # denoise / unmask the sample
        x_0_hat = self(x_t)

        # loss
        loss = self.criterion(x_0, x_0_hat)

        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # todo
        loss = None
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())
