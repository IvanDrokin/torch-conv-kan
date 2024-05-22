import pytest
import torch
import torch.nn as nn

from models import ukanet_18, ukalnet_18, fast_ukanet_18, ukacnet_18, ukagnet_18


@pytest.mark.parametrize("groups", [1, 4])
def test_ukanet_18(groups):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = ukanet_18(input_dim, num_classes, spline_order=3, groups=groups,
                     grid_size=5, base_activation=nn.GELU,
                     grid_range=[-1, 1])
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_fast_ukanet_18(groups):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = fast_ukanet_18(input_dim, num_classes, groups=groups,
                          grid_size=5, base_activation=nn.GELU,
                          grid_range=[-1, 1])
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_ukalnet_18(groups):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = ukalnet_18(input_dim, num_classes, groups=groups, degree=degree)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_fast_ukacnet_18(groups):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = ukacnet_18(input_dim, num_classes, groups=groups, degree=degree)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_fast_ukagnet_18(groups):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = ukagnet_18(input_dim, num_classes, groups=groups, degree=degree)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes, spatial_dim, spatial_dim)
