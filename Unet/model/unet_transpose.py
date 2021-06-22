from torch.nn import functional as F
from torchvision import transforms
from torch import nn

import typing as tp
import torch


class ConvReLUBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
        padding=0, stride=1, dilation=1, bias=False, separable=False):
        super(ConvReLUBN, self).__init__()

        groups = in_channels if separable else 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
            padding=padding, stride=stride, dilation=dilation, bias=bias, groups=groups)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

class Exploration(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Exploration, self).__init__()
        self.sequence = nn.Sequential(
            ConvReLUBN(in_channels, out_channels, kernel_size=3, padding=1),
            ConvReLUBN(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.sequence(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features: tp.List[int]=[64, 128, 256, 512]):
        super(Unet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for feature in features:
            self.downs.append(Exploration(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(Exploration(feature*2, feature))

        self.bottleneck = Exploration(features[-1], features[-1]*2)
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for index in range(0, len(self.ups), 2):
            x = self.ups[index](x)
            skip_connection = skip_connections[index//2]
            concat = torch.cat([x, skip_connection], dim=1)
            x = self.ups[index+1](concat)

        return self.output(x)
