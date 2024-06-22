import itertools

import pytest
import torch
import torch.nn as nn

from kans import KAN, KALN, KACN, KAGN, FastKAN, WavKAN, KAJN, KABN, ReLUKAN, BottleNeckKAGN


@pytest.mark.parametrize("dropout, first_dropout, l1_decay",
                         itertools.product([0.0, 0.5], [True, False], [0, 0.1]))
def test_kan(dropout, first_dropout, l1_decay):
    bs = 6
    hidden_dim = 64
    input_dim = 32
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim))
    layers_hidden = [input_dim, hidden_dim, num_classes]

    conv = KAN(layers_hidden, spline_order=3,
               grid_size=5, base_activation=nn.GELU,
               grid_range=[-1, 1], dropout=dropout, l1_decay=l1_decay, first_dropout=first_dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, first_dropout, l1_decay",
                         itertools.product([0.0, 0.5], [True, False], [0, 0.1]))
def test_fast_kan(dropout, first_dropout, l1_decay):
    bs = 6
    hidden_dim = 64
    input_dim = 32
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim))
    layers_hidden = [input_dim, hidden_dim, num_classes]

    conv = FastKAN(layers_hidden, grid_size=5, base_activation=nn.GELU,
                   grid_range=[-1, 1], dropout=dropout, l1_decay=l1_decay, first_dropout=first_dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, first_dropout, l1_decay, wavelet_type",
                         itertools.product([0.0, 0.5], [True, False], [0, 0.1],
                                           ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon']))
def test_wav_kan(dropout, first_dropout, l1_decay, wavelet_type):
    bs = 6
    hidden_dim = 64
    input_dim = 32
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim))
    layers_hidden = [input_dim, hidden_dim, num_classes]

    conv = WavKAN(layers_hidden, wavelet_type=wavelet_type, dropout=dropout,
                  l1_decay=l1_decay, first_dropout=first_dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, first_dropout, l1_decay",
                         itertools.product([0.0, 0.5], [True, False], [0, 0.1]))
def test_kaln(dropout, first_dropout, l1_decay):
    bs = 6
    hidden_dim = 64
    input_dim = 32
    degree = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim))
    layers_hidden = [input_dim, hidden_dim, num_classes]

    conv = KALN(layers_hidden, base_activation=nn.GELU, degree=degree, dropout=dropout,
                l1_decay=l1_decay, first_dropout=first_dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, first_dropout, l1_decay",
                         itertools.product([0.0, 0.5], [True, False], [0, 0.1]))
def test_kajn(dropout, first_dropout, l1_decay):
    bs = 6
    hidden_dim = 64
    input_dim = 32
    degree = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim))
    layers_hidden = [input_dim, hidden_dim, num_classes]

    conv = KAJN(layers_hidden, base_activation=nn.GELU, degree=degree, dropout=dropout,
                l1_decay=l1_decay, first_dropout=first_dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, first_dropout, l1_decay",
                         itertools.product([0.0, 0.5], [True, False], [0, 0.1]))
def test_kagn(dropout, first_dropout, l1_decay):
    bs = 6
    hidden_dim = 64
    input_dim = 32
    degree = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim))
    layers_hidden = [input_dim, hidden_dim, num_classes]

    conv = KAGN(layers_hidden, base_activation=nn.GELU, degree=degree, dropout=dropout,
                l1_decay=l1_decay, first_dropout=first_dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, first_dropout, l1_decay, dim_reduction, min_internal",
                         itertools.product([0.0, 0.5], [True, False], [0, 0.1], [2, 4, 8, 16], [4, 8, 16, 32]))
def test_bn_kagn(dropout, first_dropout, l1_decay, dim_reduction, min_internal):
    bs = 6
    hidden_dim = 64
    input_dim = 32
    degree = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim))
    layers_hidden = [input_dim, hidden_dim, num_classes]

    conv = BottleNeckKAGN(layers_hidden, base_activation=nn.SiLU, degree=degree, dropout=dropout,
                          l1_decay=l1_decay, first_dropout=first_dropout,
                          dim_reduction=dim_reduction, min_internal=min_internal)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, first_dropout, l1_decay",
                         itertools.product([0.0, 0.5], [True, False], [0, 0.1]))
def test_kacn(dropout, first_dropout, l1_decay):
    bs = 6
    hidden_dim = 64
    input_dim = 32
    degree = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim))
    layers_hidden = [input_dim, hidden_dim, num_classes]

    conv = KACN(layers_hidden, degree=degree, dropout=dropout,
                l1_decay=l1_decay, first_dropout=first_dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, first_dropout, l1_decay",
                         itertools.product([0.0, 0.5], [True, False], [0, 0.1]))
def test_kabn(dropout, first_dropout, l1_decay):
    bs = 6
    hidden_dim = 64
    input_dim = 32
    degree = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim))
    layers_hidden = [input_dim, hidden_dim, num_classes]

    conv = KABN(layers_hidden, degree=degree, dropout=dropout,
                l1_decay=l1_decay, first_dropout=first_dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, first_dropout, l1_decay",
                         itertools.product([0.0, 0.5], [True, False], [0, 0.1]))
def test_relukan(dropout, first_dropout, l1_decay):
    bs = 6
    hidden_dim = 64
    input_dim = 32

    g = 5
    k = 3
    train_ab = True

    num_classes = 128

    input_tensor = torch.rand((bs, input_dim))
    layers_hidden = [input_dim, hidden_dim, num_classes]

    conv = ReLUKAN(layers_hidden, g=g, k=k, train_ab=train_ab, dropout=dropout,
                   l1_decay=l1_decay, first_dropout=first_dropout)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)
