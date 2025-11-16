import argparse
import math

import lightning as L
import torch
import torch.nn.functional as F
from libs.schmubert.hparams.default_hparams import HparamsAbsorbingConv
from libs.schmubert.hparams.set_up_hparams import add_common_args, add_train_args
from libs.schmubert.models import ConVormer
from libs.schmubert.models.absorbing_diffusion import AbsorbingDiffusion


def fake_args(args):
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_train_args(parser)
    return parser.parse_args(args)

class Schmubert(L.LightningModule):

    def __init__(self, optimizer):
        super(Schmubert, self).__init__()
        H = HparamsAbsorbingConv(fake_args([
            "--dataset=data/lakh_trio.npy",
            "--bars=64",
            "--batch_size=64",
            "--tracks=trio",
            "--model=conv_transformer"
        ]))
        self.optimizer = optimizer
        self.net = ConVormer(H)
        self.absorb_diff = AbsorbingDiffusion(H, self.net, H.codebook_size)

    def forward(self, x):
        return self.net(x)

    def _sample_forward_loss(self, x_0):
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t, pt = self.absorb_diff.sample_time(b, device, 'uniform')

        # make x noisy
        x_t, x_0_ignore, mask = self.absorb_diff.q_sample(x_0=x_0, t=t)

        # sample p(x_0 | x_t)
        x_0_hat_logits = self(x_t)
        x_0_hat_logits = [el.permute(0, 2, 1) for el in x_0_hat_logits]

        # Always compute ELBO for comparison purposes
        cross_entropy_loss = [F.cross_entropy(x, x_0_ignore[:, :, i], ignore_index=-1, reduction='none').sum(1)
                              for i, x in enumerate(x_0_hat_logits)]
        cross_entropy_loss = torch.stack(cross_entropy_loss).sum(0)

        loss = cross_entropy_loss / t
        loss = loss / pt
        loss = loss / (math.log(2) * x_0.shape[1:].numel())
        return loss.mean()

    def training_step(self, batch, batch_idx):
        loss = self._sample_forward_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._sample_forward_loss(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer(params=self.parameters())
