import torch
import torch.nn as nn

from kan_convs import MoEBottleNeckKAGNConv2DLayer
from kans import KAGN
from utils import L1


class SimpleMoEConvKAGNBN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d,
            num_experts=16,
            noisy_gating=True,
            k=4
    ):
        super(SimpleMoEConvKAGNBN, self).__init__()

        self.layers = nn.ModuleList([
            MoEBottleNeckKAGNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1,
                                         padding=1, stride=1,
                                         dilation=1, affine=affine, norm_layer=norm_layer, num_experts=num_experts,
                                         noisy_gating=noisy_gating, k=k),
            L1(MoEBottleNeckKAGNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups,
                                            padding=1,
                                            stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k),
               l1_penalty),
            L1(MoEBottleNeckKAGNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups,
                                            padding=1,
                                            stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k),
               l1_penalty),
            L1(MoEBottleNeckKAGNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups,
                                            padding=1,
                                            stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k),
               l1_penalty)
        ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:
            self.output = KAGN([layer_sizes[3], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)

    def forward(self, x, train=True, loss_coef=1):
        moe_loss = 0
        for layer in self.layers:
            x, _moe_loss = layer(x, train=train, loss_coef=loss_coef)
            moe_loss += moe_loss
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x, moe_loss


class EightSimpleMoEConvKAGNBN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d,
            num_experts=16,
            noisy_gating=True,
            k=4
    ):
        super(EightSimpleMoEConvKAGNBN, self).__init__()

        self.layers = nn.ModuleList([
            MoEBottleNeckKAGNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1,
                                         padding=1, stride=1,
                                         dilation=1, affine=affine, norm_layer=norm_layer, num_experts=num_experts,
                                         noisy_gating=noisy_gating, k=k),
            L1(MoEBottleNeckKAGNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups,
                                            padding=1,
                                            stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k),
               l1_penalty),
            L1(MoEBottleNeckKAGNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups,
                                            padding=1,
                                            stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k),
               l1_penalty),
            L1(MoEBottleNeckKAGNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups,
                                            padding=1,
                                            stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k),
               l1_penalty),
            L1(MoEBottleNeckKAGNConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, degree=degree, groups=groups,
                                            padding=1,
                                            stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k),
               l1_penalty),
            L1(MoEBottleNeckKAGNConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, degree=degree, groups=groups,
                                            padding=1,
                                            stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k),
               l1_penalty),
            L1(MoEBottleNeckKAGNConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, degree=degree, groups=groups,
                                            padding=1,
                                            stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k),
               l1_penalty),
            L1(MoEBottleNeckKAGNConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, degree=degree, groups=groups,
                                            padding=1,
                                            stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k),
               l1_penalty)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAGN([layer_sizes[7], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)

    def forward(self, x, train=True, loss_coef=1):
        moe_loss = 0
        for layer in self.layers:
            x, _moe_loss = layer(x, train=train, loss_coef=loss_coef)
            moe_loss += moe_loss
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x, moe_loss
