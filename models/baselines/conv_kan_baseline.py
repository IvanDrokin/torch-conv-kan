import torch
import torch.nn as nn

from kan_convs import KANConv2DLayer
from kans import KAN
from utils import L1


class SimpleConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 3,
            degree_out: int = 2,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(SimpleConvKAN, self).__init__()

        self.layers = nn.Sequential(
            KANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:

            self.output = KAN([layer_sizes[3], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class EightSimpleConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleConvKAN, self).__init__()

        self.layers = nn.Sequential(
            KANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
