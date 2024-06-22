# taken from and based on https://github.com/1ssb/torchkan/blob/main/torchkan.py
# and https://github.com/1ssb/torchkan/blob/main/KALnet.py
# and https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py
# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# and https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
# and https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py
# and https://github.com/zavareh1/Wav-KAN
# and https://github.com/quiqi/relu_kan/issues/2
from typing import List

import torch.nn as nn

from utils import L1
from .layers import KANLayer, KALNLayer, FastKANLayer, ChebyKANLayer, GRAMLayer, WavKANLayer, JacobiKANLayer, \
    BernsteinKANLayer, ReLUKANLayer, BottleNeckGRAMLayer


class KAN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, layers_hidden, dropout: float = 0.0, grid_size=5, spline_order=3, base_activation=nn.GELU,
                 grid_range: List = [-1, 1], l1_decay: float = 0.0, first_dropout: bool = True, **kwargs):
        super(KAN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation
        self.grid_range = grid_range

        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = KANLayer(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                             base_activation=base_activation, grid_range=grid_range)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KALN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, degree=3,
                 base_activation=nn.SiLU, first_dropout: bool = True, **kwargs):
        super(KALN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = KALNLayer(in_features, out_features, degree, base_activation=base_activation)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FastKAN(nn.Module):
    def __init__(
            self,
            layers_hidden: List[int],
            dropout: float = 0.0,
            l1_decay: float = 0.0,
            grid_range: List[float] = [-2, 2],
            grid_size: int = 8,
            use_base_update: bool = True,
            base_activation=nn.SiLU,
            spline_weight_init_scale: float = 0.1,
            first_dropout: bool = True, **kwargs
    ) -> None:
        super().__init__()
        self.layers_hidden = layers_hidden
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.num_layers = len(layers_hidden[:-1])

        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = FastKANLayer(in_features, out_features,
                                 grid_min=self.grid_min,
                                 grid_max=self.grid_max,
                                 num_grids=grid_size,
                                 use_base_update=use_base_update,
                                 base_activation=base_activation,
                                 spline_weight_init_scale=spline_weight_init_scale)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KACN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0,
                 degree=3, first_dropout: bool = True, **kwargs):
        super(KACN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = ChebyKANLayer(in_features, out_features, degree)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KAGN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, degree=3,
                 base_activation=nn.SiLU, first_dropout: bool = True, **kwargs):
        super(KAGN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = GRAMLayer(in_features, out_features, degree, act=base_activation)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BottleNeckKAGN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, degree=3,
                 base_activation=nn.SiLU, first_dropout: bool = True,
                 dim_reduction: float = 8, min_internal: int = 16, **kwargs):
        super(BottleNeckKAGN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = BottleNeckGRAMLayer(in_features, out_features, degree, act=base_activation,
                                        dim_reduction=dim_reduction, min_internal=min_internal)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KABN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, degree=3,
                 base_activation=nn.SiLU, first_dropout: bool = True, **kwargs):
        super(KABN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = BernsteinKANLayer(in_features, out_features, degree, act=base_activation)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KAJN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, degree=3, a: float = 1, b: float = 1,
                 base_activation=nn.SiLU, first_dropout: bool = True, **kwargs):
        super(KAJN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = JacobiKANLayer(in_features, out_features, degree, a=a, b=b, act=base_activation)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class WavKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0,
                 first_dropout: bool = True, wavelet_type: str = 'mexican_hat', **kwargs):
        super(WavKAN, self).__init__()  # Initialize the parent nn.Module class

        assert wavelet_type in ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'], \
            ValueError(f"Unsupported wavelet type: {wavelet_type}")
        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.wavelet_type = wavelet_type
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = WavKANLayer(in_features, out_features, wavelet_type=wavelet_type)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ReLUKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, g: int = 1, k: int = 1,
                 train_ab: bool = True,
                 first_dropout: bool = True, **kwargs):
        super(ReLUKAN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = ReLUKANLayer(in_features, g, k, out_features, train_ab=train_ab)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def mlp_kan(layers_hidden, dropout: float = 0.0, grid_size: int = 5, spline_order: int = 3,
            base_activation: nn.Module = nn.GELU,
            grid_range: List = [-1, 1], l1_decay: float = 0.0) -> KAN:
    return KAN(layers_hidden, dropout=dropout, grid_size=grid_size, spline_order=spline_order,
               base_activation=base_activation, grid_range=grid_range, l1_decay=l1_decay)


def mlp_fastkan(layers_hidden, dropout: float = 0.0, grid_size: int = 5, base_activation: nn.Module = nn.GELU,
                grid_range: List = [-1, 1], l1_decay: float = 0.0) -> FastKAN:
    return FastKAN(layers_hidden, dropout=dropout, grid_size=grid_size,
                   base_activation=base_activation, grid_range=grid_range, l1_decay=l1_decay)


def mlp_kaln(layers_hidden, dropout: float = 0.0, degree: int = 3,
             base_activation: nn.Module = nn.GELU, l1_decay: float = 0.0) -> KALN:
    return KALN(layers_hidden, dropout=dropout, base_activation=base_activation, degree=degree, l1_decay=l1_decay)


def mlp_kabn(layers_hidden, dropout: float = 0.0, degree: int = 3,
             base_activation: nn.Module = nn.GELU, l1_decay: float = 0.0) -> KABN:
    return KABN(layers_hidden, dropout=dropout, base_activation=base_activation, degree=degree, l1_decay=l1_decay)


def mlp_kacn(layers_hidden, dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0) -> KACN:
    return KACN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay)


def mlp_kajn(layers_hidden, dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0,
             a: float = 1.0, b: float = 1.0) -> KAJN:
    return KAJN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay, a=a, b=b)


def mlp_kagn(layers_hidden, dropout: float = 0.0, degree: int = 3,
             base_activation: nn.Module = nn.GELU, l1_decay: float = 0.0) -> KAGN:
    return KAGN(layers_hidden, dropout=dropout, base_activation=base_activation,
                degree=degree, l1_decay=l1_decay)


def mlp_bottleneck_kagn(layers_hidden, dropout: float = 0.0, degree: int = 3,
                        base_activation: nn.Module = nn.GELU, l1_decay: float = 0.0,
                        dim_reduction: float = 8, min_internal: int = 16, ) -> BottleNeckKAGN:
    return BottleNeckKAGN(layers_hidden, dropout=dropout, base_activation=base_activation,
                          degree=degree, l1_decay=l1_decay, dim_reduction=dim_reduction, min_internal=min_internal)


def mlp_relukan(layers_hidden, dropout: float = 0.0, a: int = 1, b: int = 1, train_ab: bool = True,
                l1_decay: float = 0.0) -> ReLUKAN:
    return ReLUKAN(layers_hidden, dropout=dropout, a=a, b=b, train_ab=train_ab, l1_decay=l1_decay)


def mlp_wav_kan(layers_hidden, dropout: float = 0.0,
                wavelet_type: str = 'mexican_hat', l1_decay: float = 0.0) -> WavKAN:
    return WavKAN(layers_hidden, dropout=dropout, wavelet_type=wavelet_type, l1_decay=l1_decay)
