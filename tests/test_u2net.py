import pytest
import torch
import torch.nn as nn

from models import u2kagnet, u2kacnet, u2kalnet, u2kanet, fast_u2kanet
from models import u2kagnet_small, u2kacnet_small, u2kalnet_small, u2kanet_small, fast_u2kanet_small


@pytest.mark.parametrize("groups", [1, 4])
def test_u2kanet(groups):
    bs = 6
    spatial_dim = 256
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = u2kanet(input_dim, num_classes, spline_order=3, groups=groups,
                   grid_size=5, base_activation=nn.GELU,
                   grid_range=[-1, 1])
    out = conv(input_tensor)
    for i in range(len(out)):
        assert out[i].shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_fast_u2kanet(groups):
    bs = 6
    spatial_dim = 256
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = fast_u2kanet(input_dim, num_classes, groups=groups,
                        grid_size=5, base_activation=nn.GELU,
                        grid_range=[-1, 1])
    out = conv(input_tensor)
    for i in range(len(out)):
        assert out[i].shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_u2kalnet(groups):
    bs = 6
    spatial_dim = 256
    input_dim = 3
    num_classes = 128
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = u2kalnet(input_dim, num_classes, groups=groups, degree=degree)
    out = conv(input_tensor)
    for i in range(len(out)):
        assert out[i].shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_u2kacnet(groups):
    bs = 6
    spatial_dim = 256
    input_dim = 3
    num_classes = 128
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = u2kacnet(input_dim, num_classes, groups=groups, degree=degree)
    out = conv(input_tensor)
    for i in range(len(out)):
        assert out[i].shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_u2kagnet(groups):
    bs = 6
    spatial_dim = 256
    input_dim = 3
    num_classes = 128
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = u2kagnet(input_dim, num_classes, groups=groups, degree=degree)
    out = conv(input_tensor)

    for i in range(len(out)):
        assert out[i].shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_u2kanet_small(groups):
    bs = 6
    spatial_dim = 256
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = u2kanet_small(input_dim, num_classes, spline_order=3, groups=groups,
                         grid_size=5, base_activation=nn.GELU,
                         grid_range=[-1, 1])
    out = conv(input_tensor)
    for i in range(len(out)):
        assert out[i].shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_fast_u2kanet_small(groups):
    bs = 6
    spatial_dim = 256
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = fast_u2kanet_small(input_dim, num_classes, groups=groups,
                              grid_size=5, base_activation=nn.GELU,
                              grid_range=[-1, 1])
    out = conv(input_tensor)
    for i in range(len(out)):
        assert out[i].shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_u2kalnet_small(groups):
    bs = 6
    spatial_dim = 256
    input_dim = 3
    num_classes = 128
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = u2kalnet_small(input_dim, num_classes, groups=groups, degree=degree)
    out = conv(input_tensor)
    for i in range(len(out)):
        assert out[i].shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_u2kacnet_small(groups):
    bs = 6
    spatial_dim = 256
    input_dim = 3
    num_classes = 128
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = u2kacnet_small(input_dim, num_classes, groups=groups, degree=degree)
    out = conv(input_tensor)
    for i in range(len(out)):
        assert out[i].shape == (bs, num_classes, spatial_dim, spatial_dim)


@pytest.mark.parametrize("groups", [1, 4])
def test_u2kagnet_small(groups):
    bs = 6
    spatial_dim = 256
    input_dim = 3
    num_classes = 128
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = u2kagnet_small(input_dim, num_classes, groups=groups, degree=degree)
    out = conv(input_tensor)

    for i in range(len(out)):
        assert out[i].shape == (bs, num_classes, spatial_dim, spatial_dim)
