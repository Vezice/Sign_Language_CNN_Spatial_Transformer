import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict

"""class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 80, kernel_size = 5)
        #self.conv1 = nn.Conv2d(1, 80, kernel_size = 5)
        self.conv2 = nn.Conv2d(80, 80, kernel_size = 5)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.batch_norm1 = nn.BatchNorm2d(80)
        self.batch_norm2 = nn.BatchNorm2d(80)

        self.fc1 = nn.Linear(1280, 250)
        #self.fc2 = nn.Linear(250, 26)
        self.fc2 = nn.Linear(250, 25)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x """

""" class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=False)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(64*12*12, 9216)
        self.fc2 = nn.Linear(9216,26)

        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self,input):
        # Input [batch, 1, 28, 28]
        conv1 = self.conv1(input) # 19
        self.conv1_out = conv1
        # [batch, 8, 26, 26]
        relu1 = F.relu(conv1)
        self.relu1_out = relu1

        conv2 = self.conv2(relu1)
        self.conv2_out = conv2
        # [batch, 16, 24, 24]
        relu2 = F.relu(conv2)
        self.relu2_out = relu2

        pool = self.pool(relu2)
        self.pool_out = pool
        # [batch, 16, 12, 12]
        dropout = self.dropout(pool)
        self.dropout_out = dropout

        self.linear1 = dropout.view(-1, 64 * 12 * 12)
        self.linear1_out = self.linear1
        self.linear2 = self.fc1(self.linear1)
        self.linear2_out = self.linear2
        self.output = self.softmax(self.fc2(self.linear2))

        return(self.output) #43 """

""" class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(3, 80, kernel_size = 5)
        self.conv2 = nn.Conv2d(80, 80, kernel_size = 5)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)

        self.batch_norm1 = nn.BatchNorm2d(80)
        self.batch_norm2 = nn.BatchNorm2d(80)

        #self.fc1 = nn.Linear(15680, 250)
        self.fc1 = nn.Linear(1280, 250)
        self.fc2 = nn.Linear(250, 26)

    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x """


"""
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.localization = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
                nn.Linear(10 * 441, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, 100)),
            ('fc2', nn.Linear(100, 26)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.conv1 = conv1

        self.resnet18.fc = fc


    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 441)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    
    def forward(self, x):
        x = self.stn(x)
        x = self.resnet18(x)
        
        return x
"""
#Model for combi_STN_ResNet2_IMPORTANT.pt

"""
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.localization = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(64, 128, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
                nn.Linear(128 * 21 * 21, 128),
                nn.ReLU(True),
                nn.Linear(128, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, 100)),
            ('fc2', nn.Linear(100, 26)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.conv1 = conv1

        self.resnet18.fc = fc


    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 128 * 21 * 21)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid,padding_mode="border")

        return x

    
    def forward(self, x):
        x = self.stn(x)
        x = self.resnet18(x)
        
        return x """


# Real Pengujian

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.localization = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
                nn.Linear(64 * 52 * 52, 32),
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
        xs = xs.view(-1, 64 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid,padding_mode="border")

        return x


    def forward(self, x):
        x = self.stn(x)
        x = self.effnet(x)

        return x


