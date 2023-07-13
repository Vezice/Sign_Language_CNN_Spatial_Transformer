import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.localization = nn.Sequential(
                nn.Conv2d(1, 256, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
                nn.Linear(256 * 52 * 52, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.effnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)

        conv = nn.Conv2d(1, 48, kernel_size=3, stride=2, padding=1, bias=False)
        self.effnet.stem.conv = conv

        fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1000, 100)),
            ('fc2', nn.Linear(100, 26)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        self.effnet.fc = fc


    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 256 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid,padding_mode="border")

        return x


    def forward(self, x):
        x = self.stn(x)
        x = self.effnet(x)

        return x


