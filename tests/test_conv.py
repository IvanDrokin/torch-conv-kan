import itertools

import pytest
import torch
import torch.nn as nn

from kan_convs import BottleNeckReLUKANConv1DLayer, BottleNeckReLUKANConv2DLayer, BottleNeckReLUKANConv3DLayer
from kan_convs import FastKANConv1DLayer, FastKANConv2DLayer, FastKANConv3DLayer
from kan_convs import KABNConv1DLayer, KABNConv2DLayer, KABNConv3DLayer
from kan_convs import KACNConv1DLayer, KACNConv2DLayer, KACNConv3DLayer
from kan_convs import KAGNConv1DLayer, KAGNConv2DLayer, KAGNConv3DLayer
from kan_convs import KAJNConv1DLayer, KAJNConv2DLayer, KAJNConv3DLayer
from kan_convs import KALNConv1DLayer, KALNConv2DLayer, KALNConv3DLayer
from kan_convs import KANConv1DLayer, KANConv2DLayer, KANConv3DLayer
from kan_convs import MoEFastKANConv1DLayer, MoEFastKANConv2DLayer, MoEFastKANConv3DLayer
from kan_convs import MoEKACNConv1DLayer, MoEKACNConv2DLayer, MoEKACNConv3DLayer
from kan_convs import MoEKAGNConv1DLayer, MoEKAGNConv2DLayer, MoEKAGNConv3DLayer
from kan_convs import MoEKALNConv1DLayer, MoEKALNConv2DLayer, MoEKALNConv3DLayer
from kan_convs import MoEKANConv1DLayer, MoEKANConv2DLayer, MoEKANConv3DLayer
from kan_convs import MoEWavKANConv1DLayer, MoEWavKANConv2DLayer, MoEWavKANConv3DLayer
from kan_convs import ReLUKANConv1DLayer, ReLUKANConv2DLayer, ReLUKANConv3DLayer
from kan_convs import WavKANConv1DLayer, WavKANConv2DLayer, WavKANConv3DLayer


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_kan_conv_1d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = KANConv1DLayer(input_dim, output_dim, kernel_size, spline_order=3, groups=groups, padding=padding,
                          stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                          grid_range=[-1, 1], dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_kan_conv_2d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = KANConv2DLayer(input_dim, output_dim, kernel_size, spline_order=3, groups=groups, padding=padding,
                          stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                          grid_range=[-1, 1], dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_kan_conv_3d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = KANConv3DLayer(input_dim, output_dim, kernel_size, spline_order=3, groups=groups, padding=padding,
                          stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                          grid_range=[-1, 1], dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_fastkan_conv_1d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = FastKANConv1DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                              stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                              grid_range=[-1, 1], dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_fastkan_conv_2d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = FastKANConv2DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                              stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                              grid_range=[-1, 1], dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_fastkan_conv_3d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = FastKANConv3DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                              stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                              grid_range=[-1, 1], dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, wavelets, implementation",
                         itertools.product([0.0, 0.5], [1, 4],
                                           ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'],
                                           ['base', 'fast', 'fast_plus_one']))
def test_wavkan_conv_1d(dropout, groups, wavelets, implementation):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = WavKANConv1DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                             stride=1, dilation=1, wavelet_type=wavelets, dropout=dropout,
                             wav_version=implementation)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, wavelets, implementation",
                         itertools.product([0.0, 0.5], [1, 4],
                                           ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'],
                                           ['base', 'fast', 'fast_plus_one']))
def test_wavkan_conv_2d(dropout, groups, wavelets, implementation):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = WavKANConv2DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                             stride=1, dilation=1, wavelet_type=wavelets, dropout=dropout,
                             wav_version=implementation)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, wavelets, implementation",
                         itertools.product([0.0, 0.5], [1, 4],
                                           ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'],
                                           ['base', 'fast']))
def test_wavkan_conv_3d(dropout, groups, wavelets, implementation):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = WavKANConv3DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                             stride=1, dilation=1, wavelet_type=wavelets, dropout=dropout,
                             wav_version=implementation)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, conv_class", itertools.product([0.0, 0.5],
                                                                          [1, 4],
                                                                          [KALNConv1DLayer, KAGNConv1DLayer,
                                                                           KACNConv1DLayer, KAJNConv1DLayer,
                                                                           KABNConv1DLayer]))
def test_kalgcn_conv_1d(dropout, groups, conv_class):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = conv_class(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                      stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, conv_class", itertools.product([0.0, 0.5],
                                                                          [1, 4],
                                                                          [KALNConv2DLayer, KAGNConv2DLayer,
                                                                           KACNConv2DLayer, KAJNConv2DLayer,
                                                                           KABNConv2DLayer]))
def test_kalgcn_conv_2d(dropout, groups, conv_class):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = conv_class(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                      stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, conv_class", itertools.product([0.0, 0.5],
                                                                          [1, 4],
                                                                          [KALNConv3DLayer, KAGNConv3DLayer,
                                                                           KACNConv3DLayer, KAJNConv3DLayer,
                                                                           KABNConv3DLayer]))
def test_kalgcn_conv_3d(dropout, groups, conv_class):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = conv_class(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                      stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating", itertools.product([0.0, 0.5], [1, 4], [True, False]))
def test_moekan_conv_1d(dropout, groups, noisy_gating):
    bs = 32
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = MoEKANConv1DLayer(input_dim, output_dim, kernel_size=kernel_size, spline_order=3, groups=groups,
                             padding=padding,
                             stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                             grid_range=[-1, 1], dropout=dropout, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating", itertools.product([0.0, 0.5], [1, 4], [True, False]))
def test_moekan_conv_2d(dropout, groups, noisy_gating):
    bs = 32
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = MoEKANConv2DLayer(input_dim, output_dim, kernel_size=kernel_size, spline_order=3, groups=groups,
                             padding=padding,
                             stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                             grid_range=[-1, 1], dropout=dropout, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating", itertools.product([0.0, 0.5], [1, 4], [True, False]))
def test_moekan_conv_3d(dropout, groups, noisy_gating):
    bs = 32
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = MoEKANConv3DLayer(input_dim, output_dim, kernel_size=kernel_size, spline_order=3, groups=groups,
                             padding=padding,
                             stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                             grid_range=[-1, 1], dropout=dropout, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating", itertools.product([0.0, 0.5], [1, 4], [True, False]))
def test_moefastkan_conv_1d(dropout, groups, noisy_gating):
    bs = 32
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = MoEFastKANConv1DLayer(input_dim, output_dim, kernel_size=kernel_size, groups=groups, padding=padding,
                                 stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                                 grid_range=[-1, 1], dropout=dropout, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating", itertools.product([0.0, 0.5], [1, 4], [True, False]))
def test_moekan_conv_2d(dropout, groups, noisy_gating):
    bs = 32
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = MoEFastKANConv2DLayer(input_dim, output_dim, kernel_size=kernel_size, groups=groups, padding=padding,
                                 stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                                 grid_range=[-1, 1], dropout=dropout, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating", itertools.product([0.0, 0.5], [1, 4], [True, False]))
def test_moekan_conv_3d(dropout, groups, noisy_gating):
    bs = 32
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = MoEFastKANConv3DLayer(input_dim, output_dim, kernel_size=kernel_size, groups=groups, padding=padding,
                                 stride=1, dilation=1, grid_size=5, base_activation=nn.GELU,
                                 grid_range=[-1, 1], dropout=dropout, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating, conv_class", itertools.product([0.0, 0.5],
                                                                                        [1, 4], [True, False],
                                                                                        [MoEKALNConv1DLayer,
                                                                                         MoEKAGNConv1DLayer,
                                                                                         MoEKACNConv1DLayer]))
def test_moekalgcn_conv_1d(dropout, groups, noisy_gating, conv_class):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = conv_class(input_dim, output_dim, kernel_size=kernel_size, groups=groups, padding=padding,
                      stride=1, dilation=1, dropout=dropout, degree=3, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating, conv_class", itertools.product([0.0, 0.5],
                                                                                        [1, 4], [True, False],
                                                                                        [MoEKALNConv2DLayer,
                                                                                         MoEKAGNConv2DLayer,
                                                                                         MoEKACNConv2DLayer]))
def test_moekalgcn_conv_2d(dropout, groups, noisy_gating, conv_class):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = conv_class(input_dim, output_dim, kernel_size=kernel_size, groups=groups, padding=padding,
                      stride=1, dilation=1, dropout=dropout, degree=3, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating, conv_class", itertools.product([0.0, 0.5],
                                                                                        [1, 4], [True, False],
                                                                                        [MoEKALNConv3DLayer,
                                                                                         MoEKAGNConv3DLayer,
                                                                                         MoEKACNConv3DLayer]))
def test_moekalgcn_conv_3d(dropout, groups, noisy_gating, conv_class):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = conv_class(input_dim, output_dim, kernel_size=kernel_size, groups=groups, padding=padding,
                      stride=1, dilation=1, dropout=dropout, degree=3, num_experts=8, noisy_gating=noisy_gating,
                      k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating", itertools.product([0.0, 0.5], [1, 4], [True, False]))
def test_moewavkan_conv_1d(dropout, groups, noisy_gating):
    bs = 32
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = MoEWavKANConv1DLayer(input_dim, output_dim, kernel_size=kernel_size, groups=groups, padding=padding,
                                stride=1, dilation=1, wavelet_type='mexican_hat',
                                dropout=dropout, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating", itertools.product([0.0, 0.5], [1, 4], [True, False]))
def test_moewavkan_conv_2d(dropout, groups, noisy_gating):
    bs = 32
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = MoEWavKANConv2DLayer(input_dim, output_dim, kernel_size=kernel_size, groups=groups, padding=padding,
                                stride=1, dilation=1, wavelet_type='mexican_hat',
                                dropout=dropout, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, noisy_gating", itertools.product([0.0, 0.5], [1, 4], [True, False]))
def test_moewavkan_conv_3d(dropout, groups, noisy_gating):
    bs = 32
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = MoEWavKANConv3DLayer(input_dim, output_dim, kernel_size=kernel_size, groups=groups, padding=padding,
                                stride=1, dilation=1, wavelet_type='mexican_hat',
                                dropout=dropout, num_experts=8, noisy_gating=noisy_gating, k=2)
    out, loss = conv(input_tensor, True)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)
    out, loss = conv(input_tensor, False)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_relukan_conv_1d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = ReLUKANConv1DLayer(input_dim, output_dim, kernel_size, g=5, k=3, train_ab=True,
                              groups=groups, padding=padding,
                              stride=1, dilation=1, dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_relukan_conv_2d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = ReLUKANConv2DLayer(input_dim, output_dim, kernel_size, g=5, k=3, train_ab=True,
                              groups=groups, padding=padding,
                              stride=1, dilation=1, dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_relukan_conv_3d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = ReLUKANConv3DLayer(input_dim, output_dim, kernel_size, g=5, k=3, train_ab=True,
                              groups=groups, padding=padding,
                              stride=1, dilation=1, dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_bottleneck_relukan_conv_1d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = BottleNeckReLUKANConv1DLayer(input_dim, output_dim, kernel_size, g=5, k=3, train_ab=True,
                                        groups=groups, padding=padding,
                                        stride=1, dilation=1, dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_bottleneck_relukan_conv_2d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = BottleNeckReLUKANConv2DLayer(input_dim, output_dim, kernel_size, g=5, k=3, train_ab=True,
                                        groups=groups, padding=padding,
                                        stride=1, dilation=1, dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_bottleneck_relukan_conv_3d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = BottleNeckReLUKANConv3DLayer(input_dim, output_dim, kernel_size, g=5, k=3, train_ab=True,
                                        groups=groups, padding=padding,
                                        stride=1, dilation=1, dropout=dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)
