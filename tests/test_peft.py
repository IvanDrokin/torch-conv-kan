import itertools

import pytest
import torch
import torch.nn as nn

from kan_convs import KAGNConv1DLayer, KAGNConv2DLayer, KAGNConv3DLayer
from kan_peft import PEFTKAGNConvNDLayer


@pytest.mark.parametrize("dropout, groups, extra_degrees, finetune_base, trainable_degrees",
                         itertools.product([0.0, 0.5], [1, 4], [0, 1, 3], [True, False],
                                           [None, [0, ], [2, ], [1], [3], [0, 1, 2, 3]]))
def test_peft_kagn_conv_1d(dropout, groups, extra_degrees, finetune_base, trainable_degrees):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = KAGNConv1DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                           stride=1, dilation=1, dropout=dropout, degree=degree)

    peft_conv = PEFTKAGNConvNDLayer.wrap_layer(conv, trainable_degrees=trainable_degrees,
                                               extra_degrees=extra_degrees, finetune_base=finetune_base)

    out = conv(input_tensor)
    out_p = peft_conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim)
    assert out_p.shape == (bs, output_dim, spatial_dim)
    if dropout == 0:
        assert torch.norm(out_p - out, 2) < 1e-5
    if trainable_degrees is not None:
        peft_conv.poly_weights_delta = nn.Parameter(torch.rand_like(peft_conv.poly_weights_delta))
        peft_conv.beta_weights_delta = nn.Parameter(torch.rand_like(peft_conv.beta_weights_delta))
    peft_conv.poly_weights_extra = nn.Parameter(torch.rand_like(peft_conv.poly_weights_extra))
    peft_conv.beta_weights_extra = nn.Parameter(torch.rand_like(peft_conv.beta_weights_extra))

    out_p = peft_conv(input_tensor)
    merged_layer = peft_conv.fuse_layer()
    out_f = merged_layer(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim)
    assert out_p.shape == (bs, output_dim, spatial_dim)
    if dropout == 0:
        assert torch.norm(out_p - out_f, 2) < 5e-4


@pytest.mark.parametrize("dropout, groups, extra_degrees, finetune_base, trainable_degrees",
                         itertools.product([0.0, 0.5], [1, 4], [0, 1, 3], [True, False],
                                           [None, [0, ], [2, ], [1], [3], [0, 1, 2, 3]]))
def test_peft_kagn_conv_2d(dropout, groups, extra_degrees, finetune_base, trainable_degrees):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = KAGNConv2DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                           stride=1, dilation=1, dropout=dropout, degree=degree)

    peft_conv = PEFTKAGNConvNDLayer.wrap_layer(conv, trainable_degrees=trainable_degrees,
                                               extra_degrees=extra_degrees, finetune_base=finetune_base)

    out = conv(input_tensor)
    out_p = peft_conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)
    assert out_p.shape == (bs, output_dim, spatial_dim, spatial_dim)
    if dropout == 0:
        assert torch.norm(out_p - out, 2) < 1e-5
    if trainable_degrees is not None:
        peft_conv.poly_weights_delta = nn.Parameter(torch.rand_like(peft_conv.poly_weights_delta))
        peft_conv.beta_weights_delta = nn.Parameter(torch.rand_like(peft_conv.beta_weights_delta))
    peft_conv.poly_weights_extra = nn.Parameter(torch.rand_like(peft_conv.poly_weights_extra))
    peft_conv.beta_weights_extra = nn.Parameter(torch.rand_like(peft_conv.beta_weights_extra))

    out_p = peft_conv(input_tensor)
    merged_layer = peft_conv.fuse_layer()
    out_f = merged_layer(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim)
    assert out_p.shape == (bs, output_dim, spatial_dim, spatial_dim)
    if dropout == 0:
        assert torch.norm(out_p - out_f, 2) < 1e-3



@pytest.mark.parametrize("dropout, groups, extra_degrees, finetune_base, trainable_degrees",
                         itertools.product([0.0, 0.5], [1, 4], [0, 1, 3], [True, False],
                                           [None, [0, ], [2, ], [1], [3], [0, 1, 2, 3]]))
def test_peft_kagn_conv_3d(dropout, groups, extra_degrees, finetune_base, trainable_degrees):
    bs = 6
    spatial_dim = 32
    input_dim = 4
    output_dim = 16
    kernel_size = 3
    padding = 1
    degree = 3

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = KAGNConv3DLayer(input_dim, output_dim, kernel_size, groups=groups, padding=padding,
                           stride=1, dilation=1, dropout=dropout, degree=degree)

    peft_conv = PEFTKAGNConvNDLayer.wrap_layer(conv, trainable_degrees=trainable_degrees,
                                               extra_degrees=extra_degrees, finetune_base=finetune_base)

    out = conv(input_tensor)
    out_p = peft_conv(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)
    assert out_p.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)
    if dropout == 0:
        assert torch.norm(out_p - out, 2) < 1e-5

    if trainable_degrees is not None:
        peft_conv.poly_weights_delta = nn.Parameter(torch.rand_like(peft_conv.poly_weights_delta))
        peft_conv.beta_weights_delta = nn.Parameter(torch.rand_like(peft_conv.beta_weights_delta))
    peft_conv.poly_weights_extra = nn.Parameter(torch.rand_like(peft_conv.poly_weights_extra))
    peft_conv.beta_weights_extra = nn.Parameter(torch.rand_like(peft_conv.beta_weights_extra))

    out_p = peft_conv(input_tensor)
    merged_layer = peft_conv.fuse_layer()
    out_f = merged_layer(input_tensor)
    assert out.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)
    assert out_p.shape == (bs, output_dim, spatial_dim, spatial_dim, spatial_dim)
    if dropout == 0:
        assert torch.norm(out_p - out_f, 2) < 7e-3
