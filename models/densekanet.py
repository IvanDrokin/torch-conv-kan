from functools import partial
from functools import partial
from typing import List, Tuple, Type, Union, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from kan_convs import KALNConv2DLayer, KANConv2DLayer, KACNConv2DLayer, FastKANConv2DLayer, KAGNConv2DLayer, \
    BottleNeckKAGNConv2DLayer
from kans import KAN, KALN, KAGN, KACN, FastKAN, BottleNeckKAGN
from .model_utils import kan_conv1x1, fast_kan_conv1x1, kaln_conv1x1, kacn_conv1x1, kan_conv3x3, kaln_conv3x3, \
    fast_kan_conv3x3, kacn_conv3x3, kagn_conv1x1, kagn_conv3x3, bottleneck_kagn_conv1x1, bottleneck_kagn_conv3x3, \
    moe_bottleneck_kagn_conv3x3


class _DenseLayer(nn.Module):
    def __init__(
            self,
            conv1x1_fun,
            conv3x3_fun,
            num_input_features: int,
            growth_rate: int,
            bn_size: int,
            dropout: float,
            memory_efficient: bool = False,
            is_moe: bool = False
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1_fun(num_input_features, bn_size * growth_rate, stride=1)

        self.conv2 = conv3x3_fun(bn_size * growth_rate, growth_rate, stride=1, dropout=dropout)

        self.memory_efficient = memory_efficient
        self.is_moe = is_moe

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

    def forward(self, input: Tensor, **kwargs) -> Union[Tensor, tuple]:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)
        if self.is_moe:
            return self.conv2(bottleneck_output, **kwargs)
        return self.conv2(bottleneck_output)


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
            is_moe: bool = False
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
                is_moe=is_moe
            )
            self.add_module("denselayer%d" % (i + 1), layer)
        self.is_moe = is_moe

    def forward(self, init_features: Tensor, **kwargs) -> Union[Tensor, tuple]:
        features = [init_features]
        moe_loss = 0.
        for name, layer in self.items():
            new_features = layer(features, **kwargs)
            if self.is_moe:
                new_features, _moe_loss = new_features
                moe_loss += _moe_loss
            features.append(new_features)
        if self.is_moe:
            return torch.cat(features, 1), moe_loss
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
                 grid_range: List = [-1, 1],
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(kan_conv1x1, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kan_conv3x3, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                l1_decay=l1_decay, groups=groups, **norm_kwargs)

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
                 degree: int = 3,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kaln_conv3x3, degree=degree, l1_decay=l1_decay, groups=groups, **norm_kwargs)

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
                 degree: int = 3,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(kagn_conv1x1, degree=degree, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kagn_conv3x3, degree=degree, l1_decay=l1_decay, groups=groups, **norm_kwargs)

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
                 degree: int = 3,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(kacn_conv1x1, degree=degree, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kacn_conv3x3, degree=degree, l1_decay=l1_decay, groups=groups, **norm_kwargs)

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
                 base_activation: nn.Module = nn.SiLU,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(fast_kan_conv1x1, grid_range=grid_range, grid_size=grid_size, l1_decay=l1_decay,
                                base_activation=base_activation, **norm_kwargs)
        conv3x3x3_fun = partial(fast_kan_conv3x3, grid_range=grid_range, grid_size=grid_size, l1_decay=l1_decay,
                                groups=groups, base_activation=base_activation, **norm_kwargs)

        super(_FastKANDenseBlock, self).__init__(conv1x1x1_fun,
                                                 conv3x3x3_fun,
                                                 num_layers,
                                                 num_input_features,
                                                 bn_size,
                                                 growth_rate,
                                                 dropout,
                                                 memory_efficient=memory_efficient)


class _BottleNeckKAGNDenseBlock(_DenseBlock):
    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 dropout: float = 0.0,
                 memory_efficient: bool = False,
                 groups: int = 1,
                 l1_decay: float = 0.0,
                 degree: int = 3,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(bottleneck_kagn_conv1x1, degree=degree, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(bottleneck_kagn_conv3x3, degree=degree, l1_decay=l1_decay, groups=groups, **norm_kwargs)

        super(_BottleNeckKAGNDenseBlock, self).__init__(conv1x1x1_fun,
                                                        conv3x3x3_fun,
                                                        num_layers,
                                                        num_input_features,
                                                        bn_size,
                                                        growth_rate,
                                                        dropout,
                                                        memory_efficient=memory_efficient)


class _MoEBottleNeckKAGNDenseBlock(_DenseBlock):
    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 dropout: float = 0.0,
                 memory_efficient: bool = False,
                 groups: int = 1,
                 l1_decay: float = 0.0,
                 degree: int = 3,
                 num_experts: int = 8,
                 noisy_gating: bool = True,
                 k: int = 2,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(bottleneck_kagn_conv1x1, degree=degree, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(moe_bottleneck_kagn_conv3x3, num_experts=num_experts, k=k, noisy_gating=noisy_gating,
                                degree=degree, l1_decay=l1_decay, groups=groups,
                                **norm_kwargs)

        super(_MoEBottleNeckKAGNDenseBlock, self).__init__(conv1x1x1_fun,
                                                           conv3x3x3_fun,
                                                           num_layers,
                                                           num_input_features,
                                                           bn_size,
                                                           growth_rate,
                                                           dropout,
                                                           memory_efficient=memory_efficient,
                                                           is_moe=True)


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
                                    _KALNDenseBlock, _KACNDenseBlock, _KAGNDenseBlock,
                                    _BottleNeckKAGNDenseBlock, _MoEBottleNeckKAGNDenseBlock]],
            use_first_maxpool: bool = True,
            mp_kernel_size: int = 3, mp_stride: int = 2, mp_padding: int = 1,
            fcnv_kernel_size: int = 7, fcnv_stride: int = 2, fcnv_padding: int = 3,
            input_channels: int = 3,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            dropout: float = 0,
            dropout_linear: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False,
            **kan_kwargs
    ) -> None:

        super().__init__()

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        kan_kwargs_clean.pop('dropout', None)
        kan_kwargs_clean.pop('groups', None)
        kan_kwargs_clean.pop('num_experts', None)
        kan_kwargs_clean.pop('k', None)
        kan_kwargs_clean.pop('noisy_gating', None)

        self.is_moe = False
        if _MoEBottleNeckKAGNDenseBlock == block_class:
            self.is_moe = True

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
        elif block_class in (_BottleNeckKAGNDenseBlock, _MoEBottleNeckKAGNDenseBlock):
            conv1 = BottleNeckKAGNConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                              stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        elif block_class in (_KACNDenseBlock,):
            conv1 = KACNConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                    stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        else:
            raise TypeError(f"Block {type(block_class)} is not supported")

        # First convolution

        self.layers_order = ["conv0", ]
        self.features = nn.ModuleDict()
        self.features.add_module("conv0", conv1)
        if use_first_maxpool:
            self.features.add_module(
                "pool0", nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=mp_padding))
            self.layers_order.append('pool0')

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
            self.layers_order.append("denseblock%d" % (i + 1))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)

                self.layers_order.append("transition%d" % (i + 1))
                num_features = num_features // 2

        # # Final batch norm
        # self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.dropout_lin = None
        if dropout_linear > 0:
            self.dropout_lin = nn.Dropout(p=dropout_linear)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor, **kwargs) -> Union[Tensor, tuple]:
        moe_loss = 0.
        for layer_name in self.layers_order:
            if self.is_moe and 'denseblock' in layer_name:
                x, _moe_loss = self.features[layer_name](x, **kwargs)
                moe_loss += _moe_loss
            else:
                x = self.features[layer_name](x)

        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = torch.flatten(out, 1)
        if self.dropout_lin is not None:
            out = self.dropout_lin(out)
        out = self.classifier(out)
        if self.is_moe:
            return out, moe_loss
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
                                    _KALNDenseBlock, _KACNDenseBlock, _KAGNDenseBlock, _BottleNeckKAGNDenseBlock,
                                    _MoEBottleNeckKAGNDenseBlock]],
            fcnv_kernel_size: int = 5, fcnv_stride: int = 2, fcnv_padding: int = 2,
            input_channels: int = 3,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int] = (5, 5, 5),
            num_init_features: int = 64,
            bn_size: int = 4,
            dropout: float = 0,
            dropout_linear: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False,
            **kan_kwargs
    ) -> None:

        super().__init__()

        self.is_moe = False
        if _MoEBottleNeckKAGNDenseBlock == block_class:
            self.is_moe = True
        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        kan_kwargs_clean.pop('groups', None)
        kan_kwargs_clean.pop('num_experts', None)
        kan_kwargs_clean.pop('k', None)
        kan_kwargs_clean.pop('noisy_gating', None)
        kan_kwargs_clean.pop('dropout', None)

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

        elif block_class in (_BottleNeckKAGNDenseBlock, _MoEBottleNeckKAGNDenseBlock):
            conv1 = BottleNeckKAGNConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                              stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        elif block_class in (_KACNDenseBlock,):
            conv1 = KACNConv2DLayer(input_channels, num_init_features, kernel_size=fcnv_kernel_size,
                                    stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
        else:
            raise TypeError(f"Block {type(block_class)} is not supported")
        self.layers_order = ["conv0", ]
        self.features = nn.ModuleDict()
        self.features.add_module("conv0", conv1)

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
            self.layers_order.append("denseblock%d" % (i + 1))
            num_features = num_features + num_layers * growth_rate
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
            self.features.add_module("transition%d" % (i + 1), trans)
            self.layers_order.append("transition%d" % (i + 1))
            num_features = num_features // 2

        # # Final batch norm

        self.dropout_lin = None
        if dropout_linear > 0:
            self.dropout_lin = nn.Dropout(p=dropout_linear)
        # Linear layer
        if block_class in (_KANDenseBlock,):
            self.classifier = KAN([num_features, num_classes], **kan_kwargs_clean)

        elif block_class in (_FastKANDenseBlock,):
            self.classifier = FastKAN([num_features, num_classes], **kan_kwargs_clean)
        elif block_class in (_KALNDenseBlock,):
            self.classifier = KALN([num_features, num_classes], **kan_kwargs_clean)
        elif block_class in (_KAGNDenseBlock,):
            self.classifier = KAGN([num_features, num_classes], **kan_kwargs_clean)
        elif block_class in (_BottleNeckKAGNDenseBlock, _MoEBottleNeckKAGNDenseBlock):
            self.classifier = BottleNeckKAGN([num_features, num_classes], **kan_kwargs_clean)
        elif block_class in (_KACNDenseBlock,):
            self.classifier = KACN([num_features, num_classes], **kan_kwargs_clean)
        else:
            raise TypeError(f"Block {type(block_class)} is not supported")

    def forward(self, x: Tensor, **kwargs) -> Union[Tensor, tuple]:
        moe_loss = 0.
        for layer_name in self.layers_order:
            if self.is_moe and 'denseblock' in layer_name:
                x, _moe_loss = self.features[layer_name](x, **kwargs)
                moe_loss += _moe_loss
            else:
                x = self.features[layer_name](x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        if self.dropout_lin is not None:
            x = self.dropout_lin(x)
        x = self.classifier(x)
        if self.is_moe:
            return x, moe_loss
        return x


def tiny_densekanet(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
                    base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                    grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                    growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                    norm_layer: nn.Module = nn.InstanceNorm2d) -> TinyDenseKANet:
    return TinyDenseKANet(_KANDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, spline_order=spline_order, grid_size=grid_size,
                          base_activation=base_activation, affine=affine,
                          dropout=dropout, l1_decay=l1_decay, grid_range=grid_range, memory_efficient=True,
                          norm_layer=norm_layer)


def tiny_fast_densekanet(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                         base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                         grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                         growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                         norm_layer: nn.Module = nn.InstanceNorm2d) -> TinyDenseKANet:
    return TinyDenseKANet(_FastKANDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, grid_size=grid_size, base_activation=base_activation, affine=affine,
                          dropout=dropout, l1_decay=l1_decay, grid_range=grid_range, memory_efficient=True,
                          norm_layer=norm_layer)


def tiny_densekalnet(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     dropout: float = 0.0, l1_decay: float = 0.0,
                     growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                     norm_layer: nn.Module = nn.InstanceNorm2d) -> TinyDenseKANet:
    return TinyDenseKANet(_KALNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, degree=degree, affine=affine,
                          dropout=dropout, l1_decay=l1_decay, memory_efficient=True,
                          norm_layer=norm_layer
                          )


def tiny_densekagnet(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     dropout: float = 0.0, l1_decay: float = 0.0,
                     growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                     norm_layer: nn.Module = nn.InstanceNorm2d) -> TinyDenseKANet:
    return TinyDenseKANet(_KAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, degree=degree, affine=affine,
                          dropout=dropout, l1_decay=l1_decay, memory_efficient=True,
                          norm_layer=norm_layer
                          )


def tiny_densekagnet_bn(input_channels, num_classes, groups: int = 1, degree: int = 3,
                        dropout: float = 0.0, dropout_linear: float = 0.0, l1_decay: float = 0.0,
                        growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                        norm_layer: nn.Module = nn.BatchNorm2d) -> TinyDenseKANet:
    return TinyDenseKANet(_BottleNeckKAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, degree=degree, affine=affine,
                          dropout=dropout, l1_decay=l1_decay, memory_efficient=True,
                          norm_layer=norm_layer, dropout_linear=dropout_linear
                          )


def tiny_densekagnet_moebn(input_channels, num_classes, groups: int = 1, degree: int = 3,
                           dropout: float = 0.0, dropout_linear: float = 0.0, l1_decay: float = 0.0,
                           growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                           norm_layer: nn.Module = nn.BatchNorm2d,
                           num_experts: int = 8, noisy_gating: bool = True, k: int = 2
                           ) -> TinyDenseKANet:
    return TinyDenseKANet(_MoEBottleNeckKAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, degree=degree, affine=affine,
                          dropout=dropout, l1_decay=l1_decay, memory_efficient=False,
                          norm_layer=norm_layer, noisy_gating=noisy_gating, k=k, num_experts=num_experts,
                          dropout_linear=dropout_linear
                          )


def tiny_densekacnet(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     dropout: float = 0.0, l1_decay: float = 0.0,
                     growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                     norm_layer: nn.Module = nn.InstanceNorm2d) -> TinyDenseKANet:
    return TinyDenseKANet(_KACNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                          growth_rate=growth_rate, block_config=(5, 5, 5), num_init_features=num_init_features,
                          groups=groups, degree=degree, affine=affine,
                          dropout=dropout, l1_decay=l1_decay, memory_efficient=True,
                          norm_layer=norm_layer
                          )


def densekanet121(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
                  base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                  grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                  use_first_maxpool: bool = True,
                  growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                  norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_KANDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, spline_order=spline_order, grid_size=grid_size, base_activation=base_activation,
                      dropout=dropout, l1_decay=l1_decay, grid_range=grid_range, use_first_maxpool=use_first_maxpool,
                      affine=affine,
                      norm_layer=norm_layer
                      )


def fast_densekanet121(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                       base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                       grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                       use_first_maxpool: bool = True,
                       growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                       norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_FastKANDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, grid_size=grid_size, base_activation=base_activation, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, grid_range=grid_range, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer
                      )


def densekalnet121(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_KALNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer
                      )


def densekacnet121(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_KACNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer
                      )


def densekagnet121(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_KAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer
                      )


def densekalnet161(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 48, num_init_features: int = 64, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_KALNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 36, 24), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer
                      )


def densekalnet169(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_KALNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 32, 32), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer
                      )


def densekalnet201(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_KALNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 48, 32), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer
                      )


def densekagnet161(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 48, num_init_features: int = 64, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_KAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 36, 24), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer
                      )


def densekagnet169(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_KAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 32, 32), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer
                      )


def densekagnet201(input_channels, num_classes, groups: int = 1, degree: int = 3,
                   dropout: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                   growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                   norm_layer: nn.Module = nn.InstanceNorm2d) -> DenseKANet:
    return DenseKANet(_KAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 48, 32), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer
                      )


def densekagnet121bn(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     dropout: float = 0.0, dropout_linear: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                     growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                     norm_layer: nn.Module = nn.BatchNorm2d) -> DenseKANet:
    return DenseKANet(_BottleNeckKAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer, dropout_linear=dropout_linear
                      )


def densekagnet161bn(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     dropout: float = 0.0, dropout_linear: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                     growth_rate: int = 48, num_init_features: int = 64, affine: bool = True,
                     norm_layer: nn.Module = nn.BatchNorm2d) -> DenseKANet:
    return DenseKANet(_BottleNeckKAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 36, 24), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer, dropout_linear=dropout_linear
                      )


def densekagnet169bn(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     dropout: float = 0.0, dropout_linear: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                     growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                     norm_layer: nn.Module = nn.BatchNorm2d) -> DenseKANet:
    return DenseKANet(_BottleNeckKAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 32, 32), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer, dropout_linear=dropout_linear
                      )


def densekagnet201bn(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     dropout: float = 0.0, dropout_linear: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                     growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                     norm_layer: nn.Module = nn.BatchNorm2d) -> DenseKANet:
    return DenseKANet(_BottleNeckKAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 48, 32), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer, dropout_linear=dropout_linear
                      )


def densekagnet121moebn(input_channels, num_classes, groups: int = 1, degree: int = 3,
                        dropout: float = 0.0, dropout_linear: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                        growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                        norm_layer: nn.Module = nn.BatchNorm2d,
                        num_experts: int = 8, noisy_gating: bool = True, k: int = 2) -> DenseKANet:
    return DenseKANet(_MoEBottleNeckKAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer, num_experts=num_experts, noisy_gating=noisy_gating, k=k,
                      dropout_linear=dropout_linear
                      )


def densekagnet161moebn(input_channels, num_classes, groups: int = 1, degree: int = 3,
                        dropout: float = 0.0, dropout_linear: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                        growth_rate: int = 48, num_init_features: int = 64, affine: bool = True,
                        norm_layer: nn.Module = nn.BatchNorm2d,
                        num_experts: int = 8, noisy_gating: bool = True, k: int = 2) -> DenseKANet:
    return DenseKANet(_MoEBottleNeckKAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 36, 24), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer, num_experts=num_experts, noisy_gating=noisy_gating, k=k,
                      dropout_linear=dropout_linear
                      )


def densekagnet169moebn(input_channels, num_classes, groups: int = 1, degree: int = 3,
                        dropout: float = 0.0, dropout_linear: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                        growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                        norm_layer: nn.Module = nn.BatchNorm2d,
                        num_experts: int = 8, noisy_gating: bool = True, k: int = 2) -> DenseKANet:
    return DenseKANet(_MoEBottleNeckKAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 32, 32), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer, num_experts=num_experts, noisy_gating=noisy_gating, k=k,
                      dropout_linear=dropout_linear
                      )


def densekagnet201moebn(input_channels, num_classes, groups: int = 1, degree: int = 3,
                        dropout: float = 0.0, dropout_linear: float = 0.0, l1_decay: float = 0.0, use_first_maxpool: bool = True,
                        growth_rate: int = 32, num_init_features: int = 64, affine: bool = True,
                        norm_layer: nn.Module = nn.BatchNorm2d,
                        num_experts: int = 8, noisy_gating: bool = True, k: int = 2) -> DenseKANet:
    return DenseKANet(_MoEBottleNeckKAGNDenseBlock, input_channels=input_channels, num_classes=num_classes,
                      growth_rate=growth_rate, block_config=(6, 12, 48, 32), num_init_features=num_init_features,
                      groups=groups, degree=degree, affine=affine,
                      dropout=dropout, l1_decay=l1_decay, use_first_maxpool=use_first_maxpool,
                      norm_layer=norm_layer, num_experts=num_experts, noisy_gating=noisy_gating, k=k,
                      dropout_linear=dropout_linear
                      )
