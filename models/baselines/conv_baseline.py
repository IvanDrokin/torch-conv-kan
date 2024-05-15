import torch
import torch.nn as nn


class SimpleConv(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            groups: int = 1):
        super(SimpleConv, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, layer_sizes[0], kernel_size=3, groups=1, padding=1, stride=1,
                      dilation=1),
            nn.BatchNorm2d(layer_sizes[0]),
            nn.ReLU(),
            nn.Conv2d(layer_sizes[0], layer_sizes[1], kernel_size=3, groups=groups, padding=1,
                      stride=2, dilation=1),
            nn.BatchNorm2d(layer_sizes[1]),
            nn.ReLU(),
            nn.Conv2d(layer_sizes[1], layer_sizes[2], kernel_size=3, groups=groups, padding=1,
                      stride=2, dilation=1),
            nn.BatchNorm2d(layer_sizes[2]),
            nn.ReLU(),
            nn.Conv2d(layer_sizes[2], layer_sizes[3], kernel_size=3, groups=groups, padding=1,
                      stride=1, dilation=1),
            nn.BatchNorm2d(layer_sizes[3]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output = nn.Linear(layer_sizes[3], num_classes)

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.output(x)
        return x


class EightSimpleConv(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            groups: int = 1):
        super(EightSimpleConv, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, layer_sizes[0], kernel_size=3, groups=1, padding=1, stride=1,
                      dilation=1),
            nn.BatchNorm2d(layer_sizes[0]),
            nn.ReLU(),
            nn.Conv2d(layer_sizes[0], layer_sizes[1], kernel_size=3, groups=groups, padding=1,
                      stride=2, dilation=1),
            nn.BatchNorm2d(layer_sizes[1]),
            nn.ReLU(),
            nn.Conv2d(layer_sizes[1], layer_sizes[2], kernel_size=3, groups=groups, padding=1,
                      stride=2, dilation=1),
            nn.BatchNorm2d(layer_sizes[2]),
            nn.ReLU(),
            nn.Conv2d(layer_sizes[2], layer_sizes[3], kernel_size=3, groups=groups, padding=1,
                      stride=1, dilation=1),
            nn.BatchNorm2d(layer_sizes[3]),
            nn.ReLU(),
            nn.Conv2d(layer_sizes[3], layer_sizes[4], kernel_size=3, groups=groups, padding=1,
                      stride=1, dilation=1),
            nn.BatchNorm2d(layer_sizes[4]),
            nn.ReLU(),
            nn.Conv2d(layer_sizes[4], layer_sizes[5], kernel_size=3, groups=groups, padding=1,
                      stride=2, dilation=1),
            nn.BatchNorm2d(layer_sizes[5]),
            nn.ReLU(),
            nn.Conv2d(layer_sizes[5], layer_sizes[6], kernel_size=3, groups=groups, padding=1,
                      stride=1, dilation=1),
            nn.BatchNorm2d(layer_sizes[6]),
            nn.ReLU(),
            nn.Conv2d(layer_sizes[6], layer_sizes[7], kernel_size=3, groups=groups, padding=1,
                      stride=1, dilation=1),
            nn.BatchNorm2d(layer_sizes[7]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output = nn.Linear(layer_sizes[7], num_classes)

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.output(x)
        return x
