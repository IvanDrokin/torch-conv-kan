import itertools

import pytest
import torch
import torch.nn as nn

from models import reskalnet_18x32p, reskagnet_18x32p, reskacnet_18x32p, reskanet_18x32p, fast_reskanet_18x32p


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, hidden_layer_dim",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [None, 128]))
def test_reskanet_18x32p(dropout, dropout_linear, groups, l1_decay, hidden_layer_dim):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))

    conv = reskanet_18x32p(input_dim, num_classes, spline_order=3, groups=groups,
                           grid_size=5, base_activation=nn.GELU,
                           grid_range=[-1, 1], dropout=dropout, l1_decay=l1_decay,
                           dropout_linear=dropout_linear, hidden_layer_dim=hidden_layer_dim)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, hidden_layer_dim",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [None, 128]))
def test_fast_reskanet_18x32p(dropout, dropout_linear, groups, l1_decay, hidden_layer_dim):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))

    conv = fast_reskanet_18x32p(input_dim, num_classes, groups=groups,
                                grid_size=5, base_activation=nn.GELU,
                                grid_range=[-1, 1], dropout=dropout, l1_decay=l1_decay,
                                dropout_linear=dropout_linear, hidden_layer_dim=hidden_layer_dim)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, hidden_layer_dim",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [None, 128]))
def test_reskacnet_18x32p(dropout, dropout_linear, groups, l1_decay, hidden_layer_dim):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))

    conv = reskacnet_18x32p(input_dim, num_classes, groups=groups, degree=3, dropout=dropout, l1_decay=l1_decay,
                            dropout_linear=dropout_linear, hidden_layer_dim=hidden_layer_dim)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, hidden_layer_dim",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [None, 128]))
def test_reskagnet_18x32p(dropout, dropout_linear, groups, l1_decay, hidden_layer_dim):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))

    conv = reskagnet_18x32p(input_dim, num_classes, groups=groups, degree=3, dropout=dropout, l1_decay=l1_decay,
                            dropout_linear=dropout_linear, hidden_layer_dim=hidden_layer_dim)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, hidden_layer_dim",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1], [None, 128]))
def test_reskalnet_18x32p(dropout, dropout_linear, groups, l1_decay, hidden_layer_dim):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))

    conv = reskalnet_18x32p(input_dim, num_classes, groups=groups, degree=3, dropout=dropout, l1_decay=l1_decay,
                            dropout_linear=dropout_linear, hidden_layer_dim=hidden_layer_dim)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)
