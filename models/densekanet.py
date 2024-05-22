from collections import OrderedDict
from functools import partial
from typing import List, Tuple, Type, Union, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from kan_convs import KALNConv2DLayer, KANConv2DLayer, KACNConv2DLayer, FastKANConv2DLayer, KAGNConv2DLayer
from .reskanet import kan_conv1x1, fast_kan_conv1x1, kaln_conv1x1, kacn_conv1x1, kan_conv3x3, kaln_conv3x3, \
    fast_kan_conv3x3, kacn_conv3x3, kagn_conv1x1, kagn_conv3x3
from kans import KAN, KALN, KAGN, KACN, FastKAN


class _DenseLayer(nn.Module):
    def __init__(
            self,
            conv1x1_fun,
            conv3x3_fun,
            num_input_features: int,
            growth_rate: int,
            bn_size: int,
            dropout: float,
            memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1_fun(num_input_features, bn_size * growth_rate, stride=1)

        self.conv2 = conv3x3_fun(bn_size * growth_rate, growth_rate, stride=1, dropout=dropout)

        self.memory_efficient = memory_efficient

    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(concated_features)  # noqa: T484
        return bottleneck_output

    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input, use_reentrant=False)

    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(bottleneck_output)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
            self,
            conv1x1x1_fun,
            conv3x3x3_fun,
            num_layers: int,
            num_input_features: int,
            bn_size: int,
            growth_rate: int,
            dropout: float,
            memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                conv1x1x1_fun,
                conv3x3x3_fun,
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout=dropout,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _KANDenseBlock(_DenseBlock):
    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 dropout: float = 0.0,
                 memory_efficient: bool = False,
                 spline_order: int = 3,
                 groups: int = 1,
                 l1_decay: float = 0.0,
                 grid_size: int = 5, base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                 grid_range: List = [-1, 1]
                 ):
        conv1x1x1_fun = partial(kan_conv1x1, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                l1_decay=l1_decay)
        conv3x3x3_fun = partial(kan_conv3x3, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                l1_decay=l1_decay, groups=groups)

        super(_KANDenseBlock, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             num_layers,
                                             num_input_features,
                                             bn_size,
                                             growth_rate,
                                             dropout,
                                             memory_efficient=memory_efficient)


class _KALNDenseBlock(_DenseBlock):
    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 dropout: float = 0.0,
                 memory_efficient: bool = False,
                 groups: int = 1,
                 l1_decay: float = 0.0,
                 degree: int = 3
                 ):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, l1_decay=l1_decay)
        conv3x3x3_fun = partial(kaln_conv3x3, degree=degree, l1_decay=l1_decay, groups=groups)

        super(_KALNDenseBlock, self).__init__(conv1x1x1_fun,
                                              conv3x3x3_fun,
                                              num_layers,
                                              num_input_features,
                                              bn_size,
                                              growth_rate,
                                              dropout,
                                              memory_efficient=memory_efficient)


class _KAGNDenseBlock(_DenseBlock):
    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 dropout: float = 0.0,
                 memory_efficient: bool = False,
                 groups: int = 1,
                 l1_decay: float = 0.0,
                 degree: int = 3
                 ):
        conv1x1x1_fun = partial(kagn_conv1x1, degree=degree, l1_decay=l1_decay)
        conv3x3x3_fun = partial(kagn_conv3x3, degree=degree, l1_decay=l1_decay, groups=groups)

        super(_KAGNDenseBlock, self).__init__(conv1x1x1_fun,
                                              conv3x3x3_fun,
                                              num_layers,
                                              num_input_features,
                                              bn_size,
                                              growth_rate,
                                              dropout,
                                              memory_efficient=memory_efficient)


class _KACNDenseBlock(_DenseBlock):
    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 dropout: float = 0.0,
                 memory_efficient: bool = False,
                 groups: int = 1,
                 l1_decay: float = 0.0,
                 degree: int = 3
                 ):
        conv1x1x1_fun = partial(kacn_conv1x1, degree=degree, l1_decay=l1_decay)
        conv3x3x3_fun = partial(kacn_conv3x3, degree=degree, l1_decay=l1_decay, groups=groups)

        super(_KACNDenseBlock, self).__init__(conv1x1x1_fun,
                                              conv3x3x3_fun,
                                              num_layers,
                                              num_input_features,
                                              bn_size,
                                              growth_rate,
                                              dropout,
                                              memory_efficient=memory_efficient)


class _FastKANDenseBlock(_DenseBlock):
    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 dropout: float = 0.0,
                 memory_efficient: bool = False,
                 groups: int = 1,
                 l1_decay: float = 0.0,
                 grid_size: int = 8,
                 grid_range=[-2, 2],
                 base_activation: nn.Module = nn.SiLU
                 ):
        conv1x1x1_fun = partial(fast_kan_conv1x1, grid_range=grid_range, grid_size=grid_size, l1_decay=l1_decay,
                                base_activation=base_activation)
        conv3x3x3_fun = partial(fast_kan_conv3x3, grid_range=grid_range, grid_size=grid_size, l1_decay=l1_decay,
                                groups=groups, base_activation=base_activation)

        super(_FastKANDenseBlock, self).__init__(conv1x1x1_fun,
                                                 conv3x3x3_fun,
                                                 num_layers,
                                                 num_input_features,
                                                 bn_size,
                                                 growth_rate,
                                                 dropout,
                                                 memory_efficient=memory_efficient)


class _Transition(nn.Sequential):
    # switch to KAN Convs?
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()

        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.SELU()
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseKANet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
            self,
            block_class: Type[Union[_KANDenseBlock, _FastKANDenseBlock,
                                    _KALNDenseBlock, _KACNDenseBlock, _KAGNDenseBlock]],
            use_first_maxpool: bool = True,
            mp_kernel_size: int = 3, mp_stride: int = 2, mp_padding: int = 1,
            fcnv_kernel_size: int = 7, fcnv_stride: int = 2, fcnv_padding: int = 3,
            input_channels: int = 3,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            dropout: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False,
            **kan_kwargs
    ) -> None:

        super().__init__()

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        kan_kwargs_clean.pop('groups', None)

        first_block = []

        if block_class in (_KANDenseBlock,):
            conv1 = KANConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size, stride=fcnv_stride,
                                   padding=fcnv_padding, **kan_kwargs_clean)

        elif block_class in (_FastKANDenseBlock,):
            conv1 = FastKANConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                       stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        elif block_class in (_KALNDenseBlock,):
            conv1 = KALNConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                    stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        elif block_class in (_KAGNDenseBlock,):
            conv1 = KAGNConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                    stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        elif block_class in (_KACNDenseBlock,):
            conv1 = KACNConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                    stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        else:
            raise TypeError(f"Block {type(block_class)} is not supported")
        first_block.append(("conv0", conv1))
        if use_first_maxpool:
            first_block.append(
                ("pool0", nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=mp_padding)))

        # First convolution
        self.features = nn.Sequential(OrderedDict(first_block))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = block_class(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout=dropout,
                memory_efficient=memory_efficient,
                **kan_kwargs
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # # Final batch norm
        # self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class TinyDenseKANet(nn.Module):
    r"""Densenet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>` and https://arxiv.org/pdf/1904.10429v2.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
            self,
            block_class: Type[Union[_KANDenseBlock, _FastKANDenseBlock,
                                    _KALNDenseBlock, _KACNDenseBlock, _KAGNDenseBlock]],
            fcnv_kernel_size: int = 5, fcnv_stride: int = 2, fcnv_padding: int = 2,
            input_channels: int = 3,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int] = (5, 5, 5),
            num_init_features: int = 64,
            bn_size: int = 4,
            dropout: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False,
            **kan_kwargs
    ) -> None:

        super().__init__()

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        kan_kwargs_clean.pop('groups', None)

        first_block = []

        if block_class in (_KANDenseBlock,):
            conv1 = KANConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size, stride=fcnv_stride,
                                   padding=fcnv_padding, **kan_kwargs_clean)
        elif block_class in (_FastKANDenseBlock,):
            conv1 = FastKANConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                       stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        elif block_class in (_KALNDenseBlock,):
            conv1 = KALNConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                    stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        elif block_class in (_KAGNDenseBlock,):
            conv1 = KAGNConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                    stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        elif block_class in (_KACNDenseBlock,):
            conv1 = KACNConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                    stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        else:
            raise TypeError(f"Block {type(block_class)} is not supported")
        first_block.append(("conv0", conv1))

        # First convolution
        self.features = nn.Sequential(OrderedDict(first_block))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = block_class(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout=dropout,
                memory_efficient=memory_efficient,
                **kan_kwargs
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
            self.features.add_module("transition%d" % (i + 1), trans)
            num_features = num_features // 2

        # # Final batch norm
        # self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        if block_class in (_KANDenseBlock,):
            self.classifier = KAN([num_features, num_classes], **kan_kwargs_clean)

        elif block_class in (_FastKANDenseBlock,):
            self.classifier = FastKAN([num_features, num_classes], **kan_kwargs_clean)
        elif block_class in (_KALNDenseBlock,):
            self.classifier = KALN([num_features, num_classes], **kan_kwargs_clean)
        elif block_class in (_KAGNDenseBlock,):
            self.classifier = KAGN([num_features, num_classes], **kan_kwargs_clean)
        elif block_class in (_KACNDenseBlock,):
            self.classifier = KACN([num_features, num_classes], **kan_kwargs_clean)
        else:
            raise TypeError(f"Block {type(block_class)} is not supported")

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def tiny_densekanet(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
                    base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                    grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                    growth_rate: int = 32, num_init_features: int = 64) -> TinyDenseKANet:
    return TinyDenseKANet(_KANDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, spline_order=spline_order, grid_size=grid_size,
                          base_activation=base_activation,
                          dropout=dropout, l1_decay=l1_decay, grid_range=grid_range, memory_efficient=True)


def tiny_fast_densekanet(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                         base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                         grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                         growth_rate: int = 32, num_init_features: int = 64) -> TinyDenseKANet:
    return TinyDenseKANet(_FastKANDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, grid_size=grid_size, base_activation=base_activation,
                          dropout=dropout, l1_decay=l1_decay, grid_range=grid_range, memory_efficient=True)


def tiny_densekalnet(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     dropout: float = 0.0, l1_decay: float = 0.0,
                     growth_rate: int = 32, num_init_features: int = 64) -> TinyDenseKANet:
    return TinyDenseKANet(_KALNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, degree=degree,
                          dropout=dropout, l1_decay=l1_decay, memory_efficient=True
                          )


def tiny_densekagnet(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     dropout: float = 0.0, l1_decay: float = 0.0,
                     growth_rate: int = 32, num_init_features: int = 64) -> TinyDenseKANet:
    return TinyDenseKANet(_KAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, degree=degree,
                          dropout=dropout, l1_decay=l1_decay, memory_efficient=True
                          )


def tiny_densekacnet(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     dropout: float = 0.0, l1_decay: float = 0.0,
                     growth_rate: int = 32, num_init_features: int = 64) -> TinyDenseKANet:
    return TinyDenseKANet(_KACNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, degree=degree,
                          dropout=dropout, l1_decay=l1_decay, memory_efficient=True
                          )


def densekanet121(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
                  base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                  grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                  use_first_maxpool: bool = True,
                  growth_rate: int = 32, num_init_features: int = 64) -> DenseKANet:
    return DenseKANet(_KANDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, spline_order=spline_order, grid_size=grid_size, base_activation=base_activation,
                      dropout=dropout, l1_decay=l1_decay, grid_range=grid_range, use_first_maxpool=use_first_maxpool
                      )


def fast_densekanet121(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                  base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                  grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                  use_first_maxpool: bool = True,
                  growth_rate: int = 32, num_init_features: int = 64) -> DenseKANet:
    return DenseKANet(_FastKANDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, grid_size=grid_size, base_activation=base_activation,
                      dropout=dropout, l1_decay=l1_decay, grid_range=grid_range, use_first_maxpool=use_first_maxpool
                      )


def densekalnet121(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64) -> DenseKANet:
    return DenseKANet(_KALNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, degree=degree,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool
                      )


def densekacnet121(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64) -> DenseKANet:
    return DenseKANet(_KACNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, degree=degree,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool
                      )


def densekagnet121(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64) -> DenseKANet:
    return DenseKANet(_KAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, degree=degree,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool
                      )


def densekalnet161(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 48, num_init_features: int = 64) -> DenseKANet:
    return DenseKANet(_KALNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 36, 24), num_init_features=num_init_features,
                      groups=groups, degree=degree,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool
                      )


def densekalnet169(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64) -> DenseKANet:
    return DenseKANet(_KALNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 32, 32), num_init_features=num_init_features,
                      groups=groups, degree=degree,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool
                      )


def densekalnet201(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64) -> DenseKANet:
    return DenseKANet(_KALNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 48, 32), num_init_features=num_init_features,
                      groups=groups, degree=degree,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool
                      )
