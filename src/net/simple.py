import torch.nn as nn

class SimpleDenseNet(nn.Module):

    def __init__(self, input_size, lin1_size, lin2_size, lin3_size, output_size):
        super(SimpleDenseNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x):
        return self.model(x)