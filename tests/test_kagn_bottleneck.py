import itertools

import pytest
import torch

from kan_convs import BottleNeckKAGNConv1DLayer, BottleNeckKAGNConv2DLayer, BottleNeckKAGNConv3DLayer
from kan_convs import MoEBottleNeckKAGNConv1DLayer, MoEBottleNeckKAGNConv2DLayer, MoEBottleNeckKAGNConv3DLayer


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_kagn_conv_1d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = BottleNeckKAGNConv1DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                                     stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5],
                                                              [1, 4]))
def test_kagn_conv_2d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = BottleNeckKAGNConv2DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                                     stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5],
                                                              [1, 4]))
def test_kagn_conv_3d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = BottleNeckKAGNConv3DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                                        stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_moe_kagn_conv_1d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = MoEBottleNeckKAGNConv1DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                                     stride=1, dilation=1, dropout=dropout, degree=3)
    out, loss = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, pregate", itertools.product([0.0, 0.5],
                                                              [1, 4], [True, False]))
def test_moe_kagn_conv_2d(dropout, groups, pregate):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = MoEBottleNeckKAGNConv2DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                                        stride=1, dilation=1, dropout=dropout, degree=3, pregate=pregate)
    out, loss = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5],
                                                              [1, 4]))
def test_moe_kagn_conv_3d(dropout, groups):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = MoEBottleNeckKAGNConv3DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                                        stride=1, dilation=1, dropout=dropout, degree=3)
    out, loss = conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)
