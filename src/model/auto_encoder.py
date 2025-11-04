import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(L.LightningModule):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
