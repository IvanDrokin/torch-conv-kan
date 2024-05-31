import itertools

import pytest
import torch

from models import SimpleConvKACN, EightSimpleConvKACN
from models import SimpleConvKAGN, EightSimpleConvKAGN
from models import SimpleConvKALN, EightSimpleConvKALN
from models import SimpleConvKAN, EightSimpleConvKAN
from models import SimpleConvWavKAN, EightSimpleConvWavKAN
from models import SimpleFastConvKAN, EightSimpleFastConvKAN


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty, affine",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]))
def test_simple_conv_kan(groups, dropout, dropout_linear, l1_penalty, affine):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = SimpleConvKAN(
        [32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        spline_order=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty, affine",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]))
def test_simple_conv_kan8(groups, dropout, dropout_linear, l1_penalty, affine):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = EightSimpleConvKAN(
        [32, 32, 32, 32, 32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        spline_order=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty, affine",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]))
def test_simple_fast_conv_kan(groups, dropout, dropout_linear, l1_penalty, affine):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = SimpleFastConvKAN(
        [32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        grid_size=8,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty, affine",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]))
def test_simple_fast_conv_kan8(groups, dropout, dropout_linear, l1_penalty, affine):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = EightSimpleFastConvKAN(
        [32, 32, 32, 32, 32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        grid_size=8,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty, affine",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]))
def test_simple_conv_kaln(groups, dropout, dropout_linear, l1_penalty, affine):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = SimpleConvKALN(
        [32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        degree=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty, affine",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]))
def test_simple_conv_kaln8(groups, dropout, dropout_linear, l1_penalty, affine):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = EightSimpleConvKALN(
        [32, 32, 32, 32, 32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        degree=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty, affine",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]))
def test_simple_conv_kacn(groups, dropout, dropout_linear, l1_penalty, affine):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = SimpleConvKACN(
        [32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        degree=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty, affine",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]))
def test_simple_conv_kacn8(groups, dropout, dropout_linear, l1_penalty, affine):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = EightSimpleConvKACN(
        [32, 32, 32, 32, 32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        degree=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty, affine",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]))
def test_simple_conv_kagn(groups, dropout, dropout_linear, l1_penalty, affine):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = SimpleConvKAGN(
        [32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        degree=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty, affine",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5], [True, False]))
def test_simple_conv_kagn8(groups, dropout, dropout_linear, l1_penalty, affine):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = EightSimpleConvKAGN(
        [32, 32, 32, 32, 32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        degree=3,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty,
        affine=affine
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5]))
def test_simple_wav_conv_kan(groups, dropout, dropout_linear, l1_penalty):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = SimpleConvWavKAN(
        [32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("groups, dropout, dropout_linear, l1_penalty",
                         itertools.product([1, 4], [0, 0.5], [0, 0.5], [0, 0.5]))
def test_simple_wav_conv_kan8(groups, dropout, dropout_linear, l1_penalty):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = EightSimpleConvWavKAN(
        [32, 32, 32, 32, 32, 32, 32, 32],
        num_classes=num_classes,
        input_channels=input_dim,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_penalty=l1_penalty
    )
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)
