import lightning as L
import torch.nn.functional as F


class MNIST(L.LightningModule):

    def __init__(self, optimizer, net):
        super(MNIST, self).__init__()
        self.optimizer = optimizer
        self.net = net

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.net(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())
