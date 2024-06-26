from typing import Callable, List, Optional

import torch.nn as nn

from kan_convs import KALNConv2DLayer, KANConv2DLayer, KACNConv2DLayer, FastKANConv2DLayer, KAGNConv2DLayer, \
    WavKANConv2DLayer
from kan_convs import MoEKALNConv2DLayer, MoEKAGNConv2DLayer, BottleNeckKAGNConv2DLayer
from kan_convs import SelfKAGNtention2D, BottleNeckSelfKAGNtention2D
from kan_convs import MoEBottleNeckKAGNConv2DLayer
from utils import L1


def kan_conv3x3(in_planes: int, out_planes: int, spline_order: int = 3, groups: int = 1, stride: int = 1,
                dilation: int = 1, grid_size: int = 5, base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                grid_range: List = [-1, 1], l1_decay: float = 0.0,
                dropout: float = 0.0, **norm_kwargs) -> KANConv2DLayer:
    """3x3 convolution with padding"""

    conv = KANConv2DLayer(
        in_planes,
        out_planes,
        kernel_size=3,
        spline_order=spline_order,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        dropout=dropout,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def conv3x3(in_planes: int, out_planes: int, groups: int = 1, stride: int = 1,
            dilation: int = 1, base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
            l1_decay: float = 0.0, dropout: float = 0.0) -> nn.Sequential:
    """3x3 convolution with padding"""

    conv = nn.Conv2d(in_planes, out_planes, groups=groups, stride=stride,
                     kernel_size=3, dilation=dilation, padding=dilation)
    norm = nn.BatchNorm2d(out_planes)
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    if dropout > 0:
        return nn.Sequential(nn.Dropout(p=dropout), conv, norm, base_activation())

    return nn.Sequential(conv, norm, base_activation())


def kan_conv1x1(in_planes: int, out_planes: int, spline_order: int = 3, stride: int = 1,
                grid_size: int = 5, base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                grid_range: List = [-1, 1], l1_decay: float = 0.0,
                dropout: float = 0.0, **norm_kwargs) -> KANConv2DLayer:
    """1x1 convolution"""
    conv = KANConv2DLayer(in_planes, out_planes,
                          kernel_size=1,
                          spline_order=spline_order,
                          stride=stride,
                          grid_size=grid_size,
                          base_activation=base_activation,
                          grid_range=grid_range,
                          dropout=dropout,
                          **norm_kwargs)
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kaln_conv3x3(in_planes: int, out_planes: int, degree: int = 3, groups: int = 1, stride: int = 1,
                 dilation: int = 1, dropout: float = 0.0, norm_layer=nn.InstanceNorm2d,
                 l1_decay: float = 0.0, **norm_kwargs) -> KALNConv2DLayer:
    """3x3 convolution with padding"""
    conv = KALNConv2DLayer(
        in_planes,
        out_planes,
        degree=degree,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kagn_conv3x3(in_planes: int, out_planes: int, degree: int = 3, groups: int = 1, stride: int = 1,
                 dilation: int = 1, dropout: float = 0.0, norm_layer=nn.InstanceNorm2d,
                 l1_decay: float = 0.0, **norm_kwargs) -> KAGNConv2DLayer:
    """3x3 convolution with padding"""
    conv = KAGNConv2DLayer(
        in_planes,
        out_planes,
        degree=degree,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def self_kagn_conv3x3(in_planes: int, inner_projection: int = None, degree: int = 3, groups: int = 1, stride: int = 1,
                      dilation: int = 1, dropout: float = 0.0, norm_layer=nn.InstanceNorm2d,
                      **norm_kwargs) -> SelfKAGNtention2D:
    """3x3 convolution with padding"""
    conv = SelfKAGNtention2D(
        in_planes,
        inner_projection=inner_projection if inner_projection is None else in_planes // inner_projection,
        degree=degree,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    return conv


def bottleneck_kagn_conv3x3(in_planes: int, out_planes: int, degree: int = 3, groups: int = 1, stride: int = 1,
                            dilation: int = 1, dropout: float = 0.0, norm_layer=nn.BatchNorm2d,
                            l1_decay: float = 0.0, dim_reduction: float = 8,
                            **norm_kwargs) -> BottleNeckKAGNConv2DLayer:
    """3x3 convolution with padding"""
    conv = BottleNeckKAGNConv2DLayer(
        in_planes,
        out_planes,
        degree=degree,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        dim_reduction=dim_reduction,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def moe_bottleneck_kagn_conv3x3(in_planes: int, out_planes: int, degree: int = 3, groups: int = 1, stride: int = 1,
                            dilation: int = 1, dropout: float = 0.0, norm_layer=nn.BatchNorm2d,
                            l1_decay: float = 0.0, dim_reduction: float = 8,
                            num_experts: int = 8, noisy_gating: bool = True, k: int = 2,
                            **norm_kwargs) -> MoEBottleNeckKAGNConv2DLayer:
    """3x3 convolution with padding"""
    conv = MoEBottleNeckKAGNConv2DLayer(
        in_planes,
        out_planes,
        degree=degree,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        dim_reduction=dim_reduction,
        norm_layer=norm_layer,
        num_experts=num_experts,
        noisy_gating=noisy_gating,
        k=k,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def bottleneck_kagn_conv1x1(in_planes: int, out_planes: int, degree: int = 3, stride: int = 1,
                            dropout: float = 0.0, norm_layer=nn.BatchNorm2d,
                            l1_decay: float = 0.0, **norm_kwargs) -> KAGNConv2DLayer:
    """1x1 convolution"""
    conv = BottleNeckKAGNConv2DLayer(in_planes, out_planes, degree=degree,
                                     kernel_size=1,
                                     stride=stride, dropout=dropout, norm_layer=norm_layer, **norm_kwargs)
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def self_bottleneck_kagn_conv3x3(in_planes: int, inner_projection: int = None, degree: int = 3, groups: int = 1,
                                 stride: int = 1,
                                 dilation: int = 1, dropout: float = 0.0, norm_layer=nn.InstanceNorm2d,
                                 dim_reduction: float = 8, **norm_kwargs) -> SelfKAGNtention2D:
    """3x3 convolution with padding"""
    conv = BottleNeckSelfKAGNtention2D(
        in_planes,
        inner_projection=inner_projection if inner_projection is None else in_planes // inner_projection,
        degree=degree,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        dim_reduction=dim_reduction,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    return conv


def moe_kaln_conv3x3(in_planes: int, out_planes: int, degree: int = 3, groups: int = 1, stride: int = 1,
                     dilation: int = 1, num_experts: int = 8,
                     noisy_gating: bool = True, k: int = 2, dropout: float = 0.0,
                     l1_decay: float = 0.0, **norm_kwargs) -> MoEKALNConv2DLayer:
    """3x3 convolution with padding"""
    conv = MoEKALNConv2DLayer(
        in_planes,
        out_planes,
        degree=degree,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        num_experts=num_experts,
        noisy_gating=noisy_gating,
        k=k,
        dropout=dropout,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def moe_kagn_conv3x3(in_planes: int, out_planes: int, degree: int = 3, groups: int = 1, stride: int = 1,
                     dilation: int = 1, num_experts: int = 8,
                     noisy_gating: bool = True, k: int = 2, dropout: float = 0.0,
                     l1_decay: float = 0.0, **norm_kwargs) -> MoEKAGNConv2DLayer:
    """3x3 convolution with padding"""
    conv = MoEKAGNConv2DLayer(
        in_planes,
        out_planes,
        degree=degree,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        num_experts=num_experts,
        noisy_gating=noisy_gating,
        k=k,
        dropout=dropout,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kaln_conv1x1(in_planes: int, out_planes: int, degree: int = 3, stride: int = 1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d,
                 l1_decay: float = 0.0, **norm_kwargs) -> KALNConv2DLayer:
    """1x1 convolution"""
    conv = KALNConv2DLayer(in_planes, out_planes, degree=degree,
                           kernel_size=1,
                           stride=stride, dropout=dropout, norm_layer=norm_layer, **norm_kwargs)
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kagn_conv1x1(in_planes: int, out_planes: int, degree: int = 3, stride: int = 1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d,
                 l1_decay: float = 0.0, **norm_kwargs) -> KAGNConv2DLayer:
    """1x1 convolution"""
    conv = KAGNConv2DLayer(in_planes, out_planes, degree=degree,
                           kernel_size=1,
                           stride=stride, dropout=dropout, norm_layer=norm_layer, **norm_kwargs)
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kacn_conv3x3(in_planes: int, out_planes: int, degree: int = 3, groups: int = 1, stride: int = 1,
                 dilation: int = 1, l1_decay: float = 0.0, dropout: float = 0.0, **norm_kwargs) -> KACNConv2DLayer:
    """3x3 convolution with padding"""
    conv = KACNConv2DLayer(
        in_planes,
        out_planes,
        kernel_size=3,
        degree=degree,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def kacn_conv1x1(in_planes: int, out_planes: int, degree: int = 3, stride: int = 1,
                 l1_decay: float = 0.0, dropout: float = 0.0, **norm_kwargs) -> KACNConv2DLayer:
    """1x1 convolution"""
    conv = KACNConv2DLayer(in_planes,
                           out_planes,
                           kernel_size=1,
                           degree=degree,
                           stride=stride,
                           dropout=dropout,
                           **norm_kwargs)
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def fast_kan_conv3x3(in_planes: int, out_planes: int, groups: int = 1, stride: int = 1,
                     dilation: int = 1, grid_size=8, base_activation=nn.SiLU,
                     grid_range=[-2, 2], l1_decay: float = 0.0,
                     dropout: float = 0.0, **norm_kwargs) -> FastKANConv2DLayer:
    """3x3 convolution with padding"""
    conv = FastKANConv2DLayer(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        dropout=dropout,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def fast_kan_conv1x1(in_planes: int, out_planes: int, stride: int = 1,
                     grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2],
                     l1_decay: float = 0.0, dropout: float = 0.0, **norm_kwargs) -> FastKANConv2DLayer:
    """1x1 convolution"""
    conv = FastKANConv2DLayer(in_planes,
                              out_planes,
                              kernel_size=1,
                              stride=stride,
                              grid_size=grid_size,
                              base_activation=base_activation,
                              grid_range=grid_range,
                              dropout=dropout,
                              **norm_kwargs)
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def wav_kan_conv3x3(in_planes: int, out_planes: int, groups: int = 1, stride: int = 1,
                    dilation: int = 1, l1_decay: float = 0.0, dropout: float = 0.0,
                    wavelet_type: str = 'mexican_hat', wav_version: str = 'fast', **norm_kwargs) -> WavKANConv2DLayer:
    """3x3 convolution with padding"""
    conv = WavKANConv2DLayer(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        wavelet_type=wavelet_type,
        wav_version=wav_version,
        dropout=dropout,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def wav_kan_conv1x1(in_planes: int, out_planes: int, stride: int = 1,
                    wavelet_type: str = 'mexican_hat', wav_version: str = 'fast',
                    l1_decay: float = 0.0, dropout: float = 0.0, **norm_kwargs) -> WavKANConv2DLayer:
    """3x3 convolution with padding"""
    conv = WavKANConv2DLayer(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        wavelet_type=wavelet_type,
        wav_version=wav_version,
        dropout=dropout,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv
