import torch
import torch.nn as nn

from kan_convs import WavKANConv2DLayer
from kans import WavKAN
from utils import L1


class SimpleConvWavKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            wavelet_type='mexican_hat',
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0):
        super(SimpleConvWavKAN, self).__init__()

        self.layers = nn.Sequential(
            WavKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, wavelet_type=wavelet_type, groups=1,
                              padding=1, stride=1,
                              dilation=1),
            L1(WavKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, wavelet_type=wavelet_type,
                                 groups=groups, padding=1,
                                 stride=2, dilation=1, dropout=dropout), l1_penalty),
            L1(WavKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, wavelet_type=wavelet_type,
                                 groups=groups, padding=1,
                                 stride=2, dilation=1, dropout=dropout), l1_penalty),
            L1(WavKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, wavelet_type=wavelet_type,
                                 groups=groups, padding=1,
                                 stride=1, dilation=1, dropout=dropout), l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:
            self.output = WavKAN([layer_sizes[3], num_classes], dropout=dropout_linear, first_dropout=True,
                                 wavelet_type=wavelet_type)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class EightSimpleConvWavKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            wavelet_type='mexican_hat',
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0):
        super(EightSimpleConvWavKAN, self).__init__()

        self.layers = nn.Sequential(
            WavKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, wavelet_type=wavelet_type, groups=1,
                              padding=1, stride=1,
                              dilation=1),
            L1(WavKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, wavelet_type=wavelet_type,
                                 groups=groups, padding=1,
                                 stride=2, dilation=1, dropout=dropout), l1_penalty),
            L1(WavKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, wavelet_type=wavelet_type,
                                 groups=groups, padding=1,
                                 stride=2, dilation=1, dropout=dropout), l1_penalty),
            L1(WavKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, wavelet_type=wavelet_type,
                                 groups=groups, padding=1,
                                 stride=1, dilation=1, dropout=dropout), l1_penalty),
            L1(WavKANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, wavelet_type=wavelet_type,
                                 groups=groups, padding=1,
                                 stride=1, dilation=1, dropout=dropout), l1_penalty),
            L1(WavKANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, wavelet_type=wavelet_type,
                                 groups=groups, padding=1,
                                 stride=2, dilation=1, dropout=dropout), l1_penalty),
            L1(WavKANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, wavelet_type=wavelet_type,
                                 groups=groups, padding=1,
                                 stride=1, dilation=1, dropout=dropout), l1_penalty),
            L1(WavKANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, wavelet_type=wavelet_type,
                                 groups=groups, padding=1,
                                 stride=1, dilation=1, dropout=dropout), l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = WavKAN([layer_sizes[7], num_classes], dropout=dropout_linear, first_dropout=True,
                                 wavelet_type=wavelet_type)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
