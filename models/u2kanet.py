from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import kan_conv3x3, kagn_conv3x3, kacn_conv3x3, kaln_conv3x3, fast_kan_conv3x3, conv3x3, \
    bottleneck_kagn_conv3x3, bottleneck_kagn_conv1x1


def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode="bilinear")

    return src


class ResidualUNetBase(nn.Module):
    def __init__(self, conv_func, conf_fun_first: Callable = None, depth: int = 7, in_ch: int = 3, mid_ch: int = 12,
                 out_ch: int = 3):
        super(ResidualUNetBase, self).__init__()

        assert depth > 3, f"Minimum supported depth = 4, but provided {depth}"

        self.depth = depth
        if conf_fun_first is not None:
            self.input_conv = conf_fun_first(in_ch, out_ch, dilation=1)
        else:
            self.input_conv = conv_func(in_ch, out_ch, dilation=1)

        self.encoder_list = nn.ModuleList([conv_func(mid_ch if i > 0 else out_ch,
                                                     mid_ch,
                                                     dilation=1 if i < depth - 1 else 2) for i in range(depth)])
        self.decoder_list = nn.ModuleList([conv_func(mid_ch * 2,
                                                     mid_ch if i < depth - 2 else out_ch,
                                                     dilation=1) for i in range(depth - 1)])

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):

        x_input = self.input_conv(x)

        x_enc = x_input
        encoder_list = []
        for layer_index, layer in enumerate(self.encoder_list):
            x_enc = layer(x_enc)
            # save this to something
            if layer_index < self.depth - 1:
                encoder_list.append(x_enc)
            if layer_index < self.depth - 2:
                x_enc = self.pool(x_enc)

        x_dec = x_enc
        for layer_index, layer in enumerate(self.decoder_list):
            skip = encoder_list.pop()
            x_dec = layer(torch.cat((x_dec, skip), 1))
            if layer_index < self.depth - 2:
                x_dec = self.upsample(x_dec)

        x = x_dec + x_input
        return x


class ResidualUNetBaseF(nn.Module):
    def __init__(self, conv_func, depth: int = 4, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3):
        super(ResidualUNetBaseF, self).__init__()

        assert depth > 3, f"Minimum supported depth = 4, but provided {depth}"
        self.depth = depth

        self.input_conv = conv_func(in_ch, out_ch, dilation=1)

        self.encoder_list = nn.ModuleList([conv_func(mid_ch if i > 0 else out_ch,
                                                     mid_ch, dilation=2 ** i) for i in range(depth)])
        self.decoder_list = nn.ModuleList([conv_func(mid_ch * 2,
                                                     mid_ch if i < depth - 2 else out_ch,
                                                     dilation=2 ** (depth - 2 - i), ) for i in range(depth - 1)])

    def forward(self, x):

        x_input = self.input_conv(x)

        x_enc = x_input
        encoder_list = []
        for layer_index, layer in enumerate(self.encoder_list):
            x_enc = layer(x_enc)
            # save this to something
            if layer_index < self.depth - 1:
                encoder_list.append(x_enc)

        x_dec = x_enc
        for layer_index, layer in enumerate(self.decoder_list):
            x_dec = layer(torch.cat((x_dec, encoder_list.pop()), 1))

        x = x_dec + x_input
        return x


class U2KANet(nn.Module):
    def __init__(self, conv_func, conf_fun_first, in_ch: int = 3, out_ch: int = 1, width_factor: int = 1):
        super(U2KANet, self).__init__()

        # ResidualUNetBase(self, conv_func, depth: int = 7, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3):

        self.stage1 = ResidualUNetBase(conv_func, conf_fun_first=conf_fun_first, depth=7, in_ch=in_ch,
                                       mid_ch=8 * width_factor, out_ch=16 * width_factor)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = ResidualUNetBase(conv_func, depth=6, in_ch=16 * width_factor, mid_ch=8 * width_factor,
                                       out_ch=32 * width_factor)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = ResidualUNetBase(conv_func, depth=5, in_ch=32 * width_factor, mid_ch=16 * width_factor,
                                       out_ch=64 * width_factor)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = ResidualUNetBase(conv_func, depth=4, in_ch=64 * width_factor, mid_ch=32 * width_factor,
                                       out_ch=128 * width_factor)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # ResidualUNetBaseF (self, conv_func, depth: int = 4, in_ch: int = 3, mid_ch: int = 12, out_ch: int= 3)
        self.stage5 = ResidualUNetBaseF(conv_func, depth=4, in_ch=128 * width_factor, mid_ch=64 * width_factor,
                                        out_ch=128 * width_factor)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = ResidualUNetBaseF(conv_func, depth=4, in_ch=128 * width_factor, mid_ch=64 * width_factor,
                                        out_ch=128 * width_factor)

        # decoder
        self.stage5d = ResidualUNetBaseF(conv_func, depth=4, in_ch=256 * width_factor, mid_ch=64 * width_factor,
                                         out_ch=128 * width_factor)
        self.stage4d = ResidualUNetBase(conv_func, depth=4, in_ch=256 * width_factor, mid_ch=32 * width_factor,
                                        out_ch=64 * width_factor)
        self.stage3d = ResidualUNetBase(conv_func, depth=5, in_ch=128 * width_factor, mid_ch=16 * width_factor,
                                        out_ch=32 * width_factor)
        self.stage2d = ResidualUNetBase(conv_func, depth=6, in_ch=64 * width_factor, mid_ch=8 * width_factor,
                                        out_ch=16 * width_factor)
        self.stage1d = ResidualUNetBase(conv_func, depth=7, in_ch=32 * width_factor, mid_ch=4 * width_factor,
                                        out_ch=16 * width_factor)

        self.side1 = nn.Conv2d(16 * width_factor, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(16 * width_factor, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(32 * width_factor, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64 * width_factor, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(128 * width_factor, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(128 * width_factor, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x, **kwargs):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        """
        del hx1, hx2, hx3, hx4, hx5, hx6
        del hx5d, hx4d, hx3d, hx2d, hx1d
        del hx6up, hx5dup, hx4dup, hx3dup, hx2dup
        """

        return d0, d1, d2, d3, d4, d5, d6


class U2KANetSmall(nn.Module):
    def __init__(self, conv_func, conf_fun_first, in_ch: int = 3, out_ch: int = 1, width_factor: int = 1):
        super(U2KANetSmall, self).__init__()

        self.stage1 = ResidualUNetBase(conv_func, conf_fun_first=conf_fun_first, depth=7, in_ch=in_ch,
                                       mid_ch=4 * width_factor, out_ch=16 * width_factor)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = ResidualUNetBase(conv_func, depth=6, in_ch=16 * width_factor, mid_ch=4 * width_factor,
                                       out_ch=16 * width_factor)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = ResidualUNetBase(conv_func, depth=5, in_ch=16 * width_factor, mid_ch=4 * width_factor,
                                       out_ch=16 * width_factor)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = ResidualUNetBase(conv_func, depth=4, in_ch=16 * width_factor, mid_ch=4 * width_factor,
                                       out_ch=16 * width_factor)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = ResidualUNetBaseF(conv_func, depth=4, in_ch=16 * width_factor, mid_ch=4 * width_factor,
                                        out_ch=16 * width_factor)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = ResidualUNetBaseF(conv_func, depth=4, in_ch=16 * width_factor, mid_ch=4 * width_factor,
                                        out_ch=16 * width_factor)

        # decoder
        self.stage5d = ResidualUNetBaseF(conv_func, depth=4, in_ch=32 * width_factor, mid_ch=4 * width_factor,
                                         out_ch=16 * width_factor)
        self.stage4d = ResidualUNetBase(conv_func, depth=4, in_ch=32 * width_factor, mid_ch=4 * width_factor,
                                        out_ch=16 * width_factor)
        self.stage3d = ResidualUNetBase(conv_func, depth=5, in_ch=32 * width_factor, mid_ch=4 * width_factor,
                                        out_ch=16 * width_factor)
        self.stage2d = ResidualUNetBase(conv_func, depth=6, in_ch=32 * width_factor, mid_ch=4 * width_factor,
                                        out_ch=16 * width_factor)
        self.stage1d = ResidualUNetBase(conv_func, depth=7, in_ch=32 * width_factor, mid_ch=4 * width_factor,
                                        out_ch=16 * width_factor)

        self.side1 = nn.Conv2d(16 * width_factor, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(16 * width_factor, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(16 * width_factor, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(16 * width_factor, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(16 * width_factor, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(16 * width_factor, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x, **kwargs):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        """
        del hx1, hx2, hx3, hx4, hx5, hx6
        del hx5d, hx4d, hx3d, hx2d, hx1d
        del hx6up, hx5dup, hx4dup, hx3dup, hx2dup
        """

        return d0, d1, d2, d3, d4, d5, d6


def u2kagnet(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
             dropout: float = 0.0, l1_decay: float = 0.0, affine: bool = True,
             norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(kagn_conv3x3, degree=degree, groups=groups, dropout=dropout, l1_decay=l1_decay, affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(kagn_conv3x3, degree=degree, groups=1, dropout=dropout, l1_decay=l1_decay, affine=affine,
                             norm_layer=norm_layer)

    return U2KANet(conf_fun, conf_fun_first=conf_fun_first,
                   in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2kagnet_bn(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                dropout: float = 0.0, l1_decay: float = 0.0, affine: bool = True, dim_reduction: float = 8,
                norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(bottleneck_kagn_conv3x3, degree=degree, groups=groups, dropout=dropout, l1_decay=l1_decay,
                       affine=affine,
                       norm_layer=norm_layer, dim_reduction=dim_reduction)
    conf_fun_first = partial(bottleneck_kagn_conv3x3, degree=degree, groups=1, dropout=dropout, l1_decay=l1_decay,
                             affine=affine,
                             norm_layer=norm_layer, dim_reduction=dim_reduction)

    return U2KANet(conf_fun, conf_fun_first=conf_fun_first,
                   in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2kalnet(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
             dropout: float = 0.0, l1_decay: float = 0.0, affine: bool = True,
             norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(kaln_conv3x3, degree=degree, groups=groups, dropout=dropout, l1_decay=l1_decay, affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(kaln_conv3x3, degree=degree, groups=1, dropout=dropout, l1_decay=l1_decay, affine=affine,
                             norm_layer=norm_layer)

    return U2KANet(conf_fun, conf_fun_first, in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2kacnet(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
             dropout: float = 0.0, l1_decay: float = 0.0, affine: bool = True,
             norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(kacn_conv3x3, degree=degree, groups=groups, dropout=dropout, l1_decay=l1_decay, affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(kacn_conv3x3, degree=degree, groups=1, dropout=dropout, l1_decay=l1_decay, affine=affine,
                             norm_layer=norm_layer)

    return U2KANet(conf_fun, conf_fun_first, in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2kanet(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
            base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
            grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0, width_scale: int = 1,
            affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(kan_conv3x3, spline_order=spline_order, groups=groups, grid_size=grid_size, dropout=dropout,
                       l1_decay=l1_decay, base_activation=base_activation, grid_range=grid_range, affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(kan_conv3x3, spline_order=spline_order, groups=1, grid_size=grid_size, dropout=dropout,
                             l1_decay=l1_decay, base_activation=base_activation, grid_range=grid_range, affine=affine,
                             norm_layer=norm_layer)

    return U2KANet(conf_fun, conf_fun_first, in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def fast_u2kanet(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                 base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
                 grid_range: List = [-2, 2], dropout: float = 0.0, l1_decay: float = 0.0, width_scale: int = 1,
                 affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(fast_kan_conv3x3, groups=groups, grid_size=grid_size, dropout=dropout,
                       l1_decay=l1_decay, base_activation=base_activation, grid_range=grid_range, affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(fast_kan_conv3x3, groups=1, grid_size=grid_size, dropout=dropout,
                             l1_decay=l1_decay, base_activation=base_activation, grid_range=grid_range, affine=affine,
                             norm_layer=norm_layer)

    return U2KANet(conf_fun, conf_fun_first, in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2kagnet_small(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                   dropout: float = 0.0, l1_decay: float = 0.0,
                   affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(kagn_conv3x3, degree=degree, groups=groups, dropout=dropout, l1_decay=l1_decay, affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(kagn_conv3x3, degree=degree, groups=1, dropout=dropout, l1_decay=l1_decay, affine=affine,
                             norm_layer=norm_layer)

    return U2KANetSmall(conf_fun, conf_fun_first=conf_fun_first,
                        in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2kagnet_bn_small(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      dropout: float = 0.0, l1_decay: float = 0.0,
                      affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(bottleneck_kagn_conv3x3, degree=degree, groups=groups, dropout=dropout, l1_decay=l1_decay,
                       affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(bottleneck_kagn_conv1x1, degree=degree, groups=1, dropout=dropout, l1_decay=l1_decay,
                             affine=affine,
                             norm_layer=norm_layer)
    return U2KANetSmall(conf_fun, conf_fun_first=conf_fun_first,
                        in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2kalnet_small(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                   dropout: float = 0.0, l1_decay: float = 0.0, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(kaln_conv3x3, degree=degree, groups=groups, dropout=dropout, l1_decay=l1_decay, affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(kaln_conv3x3, degree=degree, groups=1, dropout=dropout, l1_decay=l1_decay, affine=affine,
                             norm_layer=norm_layer)

    return U2KANetSmall(conf_fun, conf_fun_first, in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2kacnet_small(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                   dropout: float = 0.0, l1_decay: float = 0.0, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(kacn_conv3x3, degree=degree, groups=groups, dropout=dropout, l1_decay=l1_decay, affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(kacn_conv3x3, degree=degree, groups=1, dropout=dropout, l1_decay=l1_decay, affine=affine,
                             norm_layer=norm_layer)

    return U2KANetSmall(conf_fun, conf_fun_first, in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2kanet_small(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
                  base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                  grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0, width_scale: int = 1,
                  affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(kan_conv3x3, spline_order=spline_order, groups=groups, grid_size=grid_size, dropout=dropout,
                       l1_decay=l1_decay, base_activation=base_activation, grid_range=grid_range, affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(kan_conv3x3, spline_order=spline_order, groups=1, grid_size=grid_size, dropout=dropout,
                             l1_decay=l1_decay, base_activation=base_activation, grid_range=grid_range, affine=affine,
                             norm_layer=norm_layer)

    return U2KANetSmall(conf_fun, conf_fun_first, in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def fast_u2kanet_small(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                       base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
                       grid_range: List = [-2, 2], dropout: float = 0.0, l1_decay: float = 0.0,
                       width_scale: int = 1, affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    conf_fun = partial(fast_kan_conv3x3, groups=groups, grid_size=grid_size, dropout=dropout,
                       l1_decay=l1_decay, base_activation=base_activation, grid_range=grid_range, affine=affine,
                       norm_layer=norm_layer)
    conf_fun_first = partial(fast_kan_conv3x3, groups=1, grid_size=grid_size, dropout=dropout,
                             l1_decay=l1_decay, base_activation=base_activation, grid_range=grid_range, affine=affine,
                             norm_layer=norm_layer)

    return U2KANetSmall(conf_fun, conf_fun_first, in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2net(input_channels, num_classes, groups: int = 1, width_scale: int = 1,
          dropout: float = 0.0, l1_decay: float = 0.0):
    conf_fun = partial(conv3x3, groups=groups, dropout=dropout, l1_decay=l1_decay)
    conf_fun_first = partial(conv3x3, groups=1, dropout=dropout, l1_decay=l1_decay)

    return U2KANet(conf_fun, conf_fun_first=conf_fun_first,
                   in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)


def u2net_small(input_channels, num_classes, groups: int = 1, width_scale: int = 1,
                dropout: float = 0.0, l1_decay: float = 0.0):
    conf_fun = partial(conv3x3, groups=groups, dropout=dropout, l1_decay=l1_decay)
    conf_fun_first = partial(conv3x3, groups=1, dropout=dropout, l1_decay=l1_decay)

    return U2KANetSmall(conf_fun, conf_fun_first, in_ch=input_channels, out_ch=num_classes, width_factor=width_scale)
