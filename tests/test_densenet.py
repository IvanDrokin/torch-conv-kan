import itertools

import pytest
import torch
import torch.nn as nn

from models import densekagnet121bn, tiny_densekagnet_bn
from models import densekanet121, densekalnet121, densekacnet121, densekagnet121, fast_densekanet121
from models import tiny_densekalnet, tiny_densekanet, tiny_densekagnet, tiny_densekacnet, tiny_fast_densekanet


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1]))
def test_tiny_densekanet(dropout, dropout_linear, groups, l1_decay):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = tiny_densekanet(input_dim, num_classes, spline_order=3, groups=groups,
                           grid_size=5, base_activation=nn.GELU,
                           grid_range=[-1, 1], dropout=dropout, l1_decay=l1_decay,
                           growth_rate=32, num_init_features=64)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, use_first_maxpool",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [True, False]))
def test_densekanet121(dropout, dropout_linear, groups, l1_decay, use_first_maxpool):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = densekanet121(input_dim, num_classes, spline_order=3, groups=groups,
                         grid_size=5, base_activation=nn.GELU,
                         grid_range=[-1, 1], dropout=dropout, l1_decay=l1_decay,
                         growth_rate=32, num_init_features=64, use_first_maxpool=use_first_maxpool)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1]))
def test_tiny_fast_densekanet(dropout, dropout_linear, groups, l1_decay):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = tiny_fast_densekanet(input_dim, num_classes, groups=groups,
                                grid_size=5, base_activation=nn.GELU,
                                grid_range=[-1, 1], dropout=dropout, l1_decay=l1_decay,
                                growth_rate=32, num_init_features=64)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, use_first_maxpool",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [True, False]))
def test_fast_densekanet121(dropout, dropout_linear, groups, l1_decay, use_first_maxpool):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = fast_densekanet121(input_dim, num_classes, groups=groups,
                              grid_size=5, base_activation=nn.GELU,
                              grid_range=[-1, 1], dropout=dropout, l1_decay=l1_decay,
                              growth_rate=32, num_init_features=64, use_first_maxpool=use_first_maxpool)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1]))
def test_tiny_densekalnet(dropout, dropout_linear, groups, l1_decay):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = tiny_densekalnet(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                            growth_rate=32, num_init_features=64, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, use_first_maxpool",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [True, False]))
def test_densekalnet121(dropout, dropout_linear, groups, l1_decay, use_first_maxpool):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = densekalnet121(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                          growth_rate=32, num_init_features=64, degree=3, use_first_maxpool=use_first_maxpool)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1]))
def test_tiny_densekagnet(dropout, dropout_linear, groups, l1_decay):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = tiny_densekagnet(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                            growth_rate=32, num_init_features=64, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, use_first_maxpool",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [True, False]))
def test_densekagnet121(dropout, dropout_linear, groups, l1_decay, use_first_maxpool):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = densekagnet121(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                          growth_rate=32, num_init_features=64, degree=3, use_first_maxpool=use_first_maxpool)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1]))
def test_tiny_densekacnet(dropout, dropout_linear, groups, l1_decay):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = tiny_densekacnet(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                            growth_rate=32, num_init_features=64, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, use_first_maxpool",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [True, False]))
def test_densekacnet121(dropout, dropout_linear, groups, l1_decay, use_first_maxpool):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = densekacnet121(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                          growth_rate=32, num_init_features=64, degree=3, use_first_maxpool=use_first_maxpool)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1]))
def test_tiny_densekagnet_bn(dropout, dropout_linear, groups, l1_decay):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = tiny_densekagnet_bn(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                               growth_rate=32, num_init_features=64, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, use_first_maxpool",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [True, False]))
def test_densekagnet121bn(dropout, dropout_linear, groups, l1_decay, use_first_maxpool):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = densekagnet121bn(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                            growth_rate=32, num_init_features=64, degree=3, use_first_maxpool=use_first_maxpool)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)
