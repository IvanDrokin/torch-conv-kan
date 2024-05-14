import torch
import torch.nn as nn

from kan_convs import KANConv2DLayer


class SimpleConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 3,
            groups: int = 1):
        super(SimpleConvKAN, self).__init__()

        self.layers = nn.Sequential(
            KANConv2DLayer(input_channels, layer_sizes[0], spline_order, kernel_size=3, groups=1, padding=1, stride=1,
                           dilation=1),
            KANConv2DLayer(layer_sizes[0], layer_sizes[1], spline_order, kernel_size=3, groups=groups, padding=1,
                           stride=2, dilation=1),
            KANConv2DLayer(layer_sizes[1], layer_sizes[2], spline_order, kernel_size=3, groups=groups, padding=1,
                           stride=2, dilation=1),
            KANConv2DLayer(layer_sizes[2], layer_sizes[3], spline_order, kernel_size=3, groups=groups, padding=1,
                           stride=1, dilation=1),
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
