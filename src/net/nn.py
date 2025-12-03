# file for torch neural net (nn) components not already in torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class Octuple(nn.Module):
    """
    Layer that reshapes (8x256) and masks invalid output/logits.
    """

    # Output Channels for OctupleMIDI
    Pit = 0
    Pos = 1
    Bar = 2
    Vel = 3
    Dur = 4
    Pro = 5
    Tem = 6
    Tim = 7

    def __init__(self):
        super(Octuple, self).__init__()
        # =========================
        # CURRENTLY WE DON'T USE THE MASK
        # =========================
        # self.mask = torch.full((1, 8, 256), False)
        # self.mask[:, Octuple.Pit, 128:] = True  # Pitch/PitchDrum
        # self.mask[:, Octuple.Pos, 128:] = True  # Position
        # # self.mask[:, Octuple.Bar, :] = True  # Bar, no limits here
        # self.mask[:, Octuple.Vel, 32:] = True  # Velocity
        # self.mask[:, Octuple.Dur, 128:] = True  # Duration
        # self.mask[:, Octuple.Pro, 129:] = True  # Program
        # self.mask[:, Octuple.Tem, 49:] = True  # Tempo
        # self.mask[:, Octuple.Tim, 254:] = True  # Time Signature

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 4, 129)
        # mask = self.mask.to(x.device)
        # x = x.masked_fill(mask, float("-inf"))
        return x

    @staticmethod
    def encode(x: torch.Tensor) -> torch.Tensor:
        """
        Encodes a OctupleMIDI tensor one-hot.
        This is the same as the layer output.
        :param x: the tensor to be encoded, must be OctupleMIDI format (last dim=8).
        :return: One-hot encoded tensor, same as Octuple layer output (last dim=256).
        """
        z = F.one_hot(x, 129)
        return z

    @staticmethod
    def encode_mask(m: torch.Tensor) -> torch.Tensor:
        return m.unsqueeze(-1).expand(m.shape[0], m.shape[1], m.shape[2], 129)

    @staticmethod
    def decode(z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a one-hot encoded (same as Octuple layer) tensor into OctupleMIDI format (just number).
        :param z: one-hot encoded Octuple layer output tensor (last dim=256).
        :return: OctupleMIDI formatted tensor (just numbers, last dim=8).
        """
        x = z.argmax(dim=-1)
        return x


if __name__ == "__main__":
    """
    ONLY FOR TESTING PURPOSES!
    """
    net = nn.Sequential(
        nn.Linear(4, 256),
        nn.ReLU(),
        nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, batch_first=True
            ), num_layers=6
        ),
        nn.Linear(256, 516),
        nn.ReLU(),
        Octuple(),
        nn.Softmax(dim=-1)
    )
    x = torch.round(torch.abs(torch.randn(64, 20, 4)) * 10)
    y = net(x)
    print(y.shape)
    print(Octuple.decode(y))
