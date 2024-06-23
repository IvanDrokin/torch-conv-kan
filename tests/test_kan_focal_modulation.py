import itertools

import pytest
import torch

from kan_convs import BottleNeckKAGNFocalModulation1D, BottleNeckKAGNFocalModulation2D, BottleNeckKAGNFocalModulation3D


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_sa_bn_kagn_conv_2d(dropout, groups):
    bs = 6
    spatial_dim = 8
    input_dim = 16

    focal_level = 2
    focal_window = 3
    focal_factor = 2
    use_postln_in_modulation = True
    normalize_modulator = True
    full_kan = True,

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = BottleNeckKAGNFocalModulation2D(input_dim, focal_window=focal_window, focal_level=focal_level,
                                           focal_factor=focal_factor,
                                           use_postln_in_modulation=use_postln_in_modulation,
                                           normalize_modulator=normalize_modulator, full_kan=full_kan,
                                           dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, input_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_sa_bn_kagn_conv_3d(dropout, groups):
    bs = 6
    spatial_dim = 8
    input_dim = 16
    padding = 1

    focal_level = 2
    focal_window = 3
    focal_factor = 2
    use_postln_in_modulation = True
    normalize_modulator = True
    full_kan = True,

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim, spatial_dim))
    conv = BottleNeckKAGNFocalModulation3D(input_dim, focal_window=focal_window, focal_level=focal_level,
                                           focal_factor=focal_factor,
                                           use_postln_in_modulation=use_postln_in_modulation,
                                           normalize_modulator=normalize_modulator, full_kan=full_kan,
                                           dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, input_dim, spatial_dim, spatial_dim, spatial_dim)


@pytest.mark.parametrize("dropout, groups", itertools.product([0.0, 0.5], [1, 4]))
def test_sa_bn_kagn_conv_1d(dropout, groups):
    bs = 6
    spatial_dim = 8
    input_dim = 16
    padding = 1

    focal_level = 2
    focal_window = 3
    focal_factor = 2
    use_postln_in_modulation = True
    normalize_modulator = True
    full_kan = True,

    input_tensor = torch.rand((bs, input_dim, spatial_dim))
    conv = BottleNeckKAGNFocalModulation1D(input_dim, focal_window=focal_window, focal_level=focal_level,
                                           focal_factor=focal_factor,
                                           use_postln_in_modulation=use_postln_in_modulation,
                                           normalize_modulator=normalize_modulator, full_kan=full_kan,
                                           dropout=dropout, degree=3)
    out = conv(input_tensor)
    assert out.shape == (bs, input_dim, spatial_dim)
