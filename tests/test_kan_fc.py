import pytest
import torch
import torch.nn as nn

from kans import KANLayer, KALNLayer, ChebyKANLayer, GRAMLayer, FastKANLayer, WavKANLayer, JacobiKANLayer, \
    BernsteinKANLayer, ReLUKANLayer, BottleNeckGRAMLayer


def test_kan_fc():
    bs = 6
    input_dim = 4
    output_dim = 16

    input_tensor = torch.rand((bs, input_dim))
    conv = KANLayer(input_dim, output_dim, spline_order=3, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1])
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim)


@pytest.mark.parametrize("use_base_update", [True, False])
def test_fastkan_fc(use_base_update):
    bs = 6
    input_dim = 4
    output_dim = 16

    input_tensor = torch.rand((bs, input_dim))
    conv = FastKANLayer(input_dim, output_dim, grid_min=-2.,
                        grid_max=2.,
                        num_grids=8,
                        use_base_update=use_base_update,
                        base_activation=nn.SiLU,
                        spline_weight_init_scale=0.1)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim)


@pytest.mark.parametrize("wavelet_type", ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'])
def test_wavkan_fc(wavelet_type):
    bs = 6
    input_dim = 4
    output_dim = 16

    input_tensor = torch.rand((bs, input_dim))
    conv = WavKANLayer(input_dim, output_dim, wavelet_type=wavelet_type)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim)


def test_kacn_fc():
    bs = 6
    input_dim = 4
    output_dim = 16
    degree = 3

    input_tensor = torch.rand((bs, input_dim))
    conv = ChebyKANLayer(input_dim, output_dim, degree)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim)


def test_kagn_fc():
    bs = 6
    input_dim = 4
    output_dim = 16
    degree = 3

    input_tensor = torch.rand((bs, input_dim))
    conv = GRAMLayer(input_dim, output_dim, degree, act=nn.SiLU)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim)


def test_bn_kagn_fc():
    bs = 6
    input_dim = 64
    output_dim = 128
    degree = 3

    input_tensor = torch.rand((bs, input_dim))
    conv = BottleNeckGRAMLayer(input_dim, output_dim, degree, act=nn.SiLU, dim_reduction=4, min_internal=8)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim)


def test_kajn_fc():
    bs = 6
    input_dim = 4
    output_dim = 16
    degree = 3

    input_tensor = torch.rand((bs, input_dim))
    conv = JacobiKANLayer(input_dim, output_dim, degree)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim)


def test_kaln_fc():
    bs = 6
    input_dim = 4
    output_dim = 16
    degree = 3

    input_tensor = torch.rand((bs, input_dim))
    conv = KALNLayer(input_dim, output_dim, degree=degree, base_activation=nn.SiLU)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim)


def test_kabn_fc():
    bs = 6
    input_dim = 4
    output_dim = 16
    degree = 3

    input_tensor = torch.rand((bs, input_dim))
    conv = BernsteinKANLayer(input_dim, output_dim, degree=degree, act=nn.SiLU)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim)


def test_relukan_fc():
    bs = 6
    input_dim = 4
    output_dim = 16
    g = 5
    k = 3
    train_ab = True

    input_tensor = torch.rand((bs, input_dim))
    conv = ReLUKANLayer(input_dim, g, k, output_dim, train_ab=train_ab)
    out = conv(input_tensor)
    assert out.shape == (bs, output_dim)

