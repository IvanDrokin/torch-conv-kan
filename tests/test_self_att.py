import itertools

import pytest
import torch

from kan_convs import BottleNeckSelfKAGNtention1D, BottleNeckSelfKAGNtention2D, BottleNeckSelfKAGNtention3D
from kan_convs import SelfKAGNtention1D, SelfKAGNtention2D, SelfKAGNtention3D


@pytest.mark.parametrize("dropout, groups, inner_projection", itertools.product([0.0, 0.5], [1, 4], [None, 8]))
def test_sa_kagn_conv_1d(dropout, groups, inner_projection):
    bs = 6
    spatial_dim = 8
    input_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = SelfKAGNtention1D(input_dim, inner_projection, kernel_size=kernel_size, groups=groups, padding=padding,
                             stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, input_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, inner_projection", itertools.product([0.0, 0.5],
                                                                                [1, 4], [None, 8]))
def test_sa_kagn_conv_2d(dropout, groups, inner_projection):
    bs = 6
    spatial_dim = 8
    input_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = SelfKAGNtention2D(input_dim, inner_projection, kernel_size=kernel_size, groups=groups, padding=padding,
                             stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, input_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, inner_projection", itertools.product([0.0, 0.5],
                                                                                [1, 4], [None, 8]))
def test_sa_kagn_conv_3d(dropout, groups, inner_projection):
    bs = 6
    spatial_dim = 8
    input_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = SelfKAGNtention3D(input_dim, inner_projection, kernel_size=kernel_size, groups=groups, padding=padding,
                             stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, input_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, inner_projection", itertools.product([0.0, 0.5], [1, 4], [None, 8]))
def test_sa_bn_kagn_conv_1d(dropout, groups, inner_projection):
    bs = 6
    spatial_dim = 8
    input_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = BottleNeckSelfKAGNtention1D(input_dim, inner_projection, kernel_size=kernel_size, groups=groups,
                                       padding=padding,
                                       stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, input_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, inner_projection", itertools.product([0.0, 0.5],
                                                                                [1, 4], [None, 8]))
def test_sa_bn_kagn_conv_2d(dropout, groups, inner_projection):
    bs = 6
    spatial_dim = 8
    input_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = BottleNeckSelfKAGNtention2D(input_dim, inner_projection, kernel_size=kernel_size, groups=groups,
                                       padding=padding,
                                       stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, input_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups, inner_projection", itertools.product([0.0, 0.5],
                                                                                [1, 4], [None, 8]))
def test_sa_bn_kagn_conv_3d(dropout, groups, inner_projection):
    bs = 6
    spatial_dim = 8
    input_dim = 16
    kernel_size = 3
    padding = 1

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = BottleNeckSelfKAGNtention3D(input_dim, inner_projection, kernel_size=kernel_size, groups=groups,
                                       padding=padding,
                                       stride=1, dilation=1, dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, input_dim, spatial_dim, spatial_dim, spatial_dim)
