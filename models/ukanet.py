from functools import partial
from typing import Callable, Optional, List, Type, Union

import torch
import torch.nn as nn

from kan_convs import KALNConv2DLayer, KANConv2DLayer, KACNConv2DLayer, FastKANConv2DLayer, KAGNConv2DLayer
from .reskanet import KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, KANBottleneck, \
    FastKANBottleneck, KALNBottleneck, KACNBottleneck, KAGNBottleneck, KAGNBasicBlock
from .reskanet import kan_conv1x1, fast_kan_conv1x1, kaln_conv1x1, kacn_conv1x1, kagn_conv1x1


class UKANet(nn.Module):
    def __init__(self,
                 block: Type[Union[KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, KAGNBasicBlock,
                                   KANBottleneck, FastKANBottleneck, KALNBottleneck, KACNBottleneck, KAGNBottleneck]],
                 layers: List[int],
                 input_channels: int = 3,
                 num_classes: int = 1000,
                 groups: int = 1,
                 width_per_group: int = 64,
                 fcnv_kernel_size=7, fcnv_stride=1, fcnv_padding=3,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 **kan_kwargs
                 ):
        super(UKANet, self).__init__()
        self.input_channels = input_channels
        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        if block in (KANBasicBlock, KANBottleneck):
            self.conv1 = KANConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size, stride=fcnv_stride,
                                        padding=fcnv_padding, **kan_kwargs)

            self.merge1 = KANConv2DLayer((32 + 64) * block.expansion, 32 * block.expansion, kernel_size=3,
                                         groups=groups, stride=1, padding=1, **kan_kwargs)
            self.merge2 = KANConv2DLayer((128 + 64) * block.expansion, 64 * block.expansion, kernel_size=3,
                                         groups=groups, stride=1, padding=1, **kan_kwargs)
            self.merge3 = KANConv2DLayer((128 + 256) * block.expansion, 128 * block.expansion, kernel_size=3,
                                         groups=groups, stride=1, padding=1, **kan_kwargs)
        elif block in (FastKANBasicBlock, FastKANBottleneck):
            self.conv1 = FastKANConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                            stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs)

            self.merge1 = FastKANConv2DLayer((32 + 64) * block.expansion, 32 * block.expansion, kernel_size=3,
                                             groups=groups, stride=1, padding=1, **kan_kwargs)
            self.merge2 = FastKANConv2DLayer((128 + 64) * block.expansion, 64 * block.expansion, kernel_size=3,
                                             groups=groups, stride=1, padding=1, **kan_kwargs)
            self.merge3 = FastKANConv2DLayer((128 + 256) * block.expansion, 128 * block.expansion, kernel_size=3,
                                             groups=groups, stride=1, padding=1, **kan_kwargs)
        elif block in (KALNBasicBlock, KALNBottleneck):
            self.conv1 = KALNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs)
            self.merge1 = KALNConv2DLayer((32 + 64) * block.expansion, 32 * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **kan_kwargs)
            self.merge2 = KALNConv2DLayer((128 + 64) * block.expansion, 64 * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **kan_kwargs)
            self.merge3 = KALNConv2DLayer((128 + 256) * block.expansion, 128 * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **kan_kwargs)
        elif block in (KAGNBasicBlock, KAGNBottleneck):
            self.conv1 = KAGNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs)
            self.merge1 = KAGNConv2DLayer((32 + 64) * block.expansion, 32 * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **kan_kwargs)
            self.merge2 = KAGNConv2DLayer((128 + 64) * block.expansion, 64 * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **kan_kwargs)
            self.merge3 = KAGNConv2DLayer((128 + 256) * block.expansion, 128 * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **kan_kwargs)
        elif block in (KACNBasicBlock, KACNBottleneck):
            self.conv1 = KACNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs)
            self.merge1 = KACNConv2DLayer((32 + 64) * block.expansion, 32 * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **kan_kwargs)
            self.merge2 = KACNConv2DLayer((128 + 64) * block.expansion, 64 * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **kan_kwargs)
            self.merge3 = KACNConv2DLayer((128 + 256) * block.expansion, 128 * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **kan_kwargs)
        else:
            raise TypeError(f"Block {type(block)} is not supported")

        self.layer1e = self._make_layer(block, 32, layers[0], **kan_kwargs)
        l1e_inplanes = self.inplanes
        self.layer2e = self._make_layer(block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                        **kan_kwargs)
        l2e_inplanes = self.inplanes
        self.layer3e = self._make_layer(block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                        **kan_kwargs)
        l3e_inplanes = self.inplanes
        self.layer4e = self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                        **kan_kwargs)

        self.layer4d = self._make_layer(block, 256, layers[3], **kan_kwargs)
        self.inplanes = l1e_inplanes
        self.layer1d = self._make_layer(block, 32, layers[0], **kan_kwargs)
        self.inplanes = l2e_inplanes
        self.layer2d = self._make_layer(block, 64, layers[1], **kan_kwargs)
        self.inplanes = l3e_inplanes
        self.layer3d = self._make_layer(block, 128, layers[2], **kan_kwargs)

        self.output = nn.Conv2d(32 * block.expansion, num_classes, kernel_size=1, padding=0, stride=1)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def _make_layer(
            self,
            block: Type[Union[KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, KAGNBasicBlock,
                              KANBottleneck, FastKANBottleneck, KALNBottleneck, KACNBottleneck, KAGNBottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            **kan_kwargs
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block in (KANBasicBlock, KANBottleneck):
                conv1x1 = partial(kan_conv1x1, **kan_kwargs)
            elif block in (FastKANBasicBlock, FastKANBottleneck):
                conv1x1 = partial(fast_kan_conv1x1, **kan_kwargs)
            elif block in (KALNBasicBlock, KALNBottleneck):
                conv1x1 = partial(kaln_conv1x1, **kan_kwargs)
            elif block in (KAGNBasicBlock, KAGNBottleneck):
                conv1x1 = partial(kagn_conv1x1, **kan_kwargs)
            elif block in (KACNBasicBlock, KACNBottleneck):
                conv1x1 = partial(kacn_conv1x1, **kan_kwargs)
            else:
                raise TypeError(f"Block {type(block)} is not supported")

            downsample = conv1x1(self.inplanes, planes * block.expansion, stride=stride)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                base_width=self.base_width, dilation=previous_dilation, **kan_kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        enc1 = self.layer1e(x)
        enc2 = self.layer2e(enc1)
        enc3 = self.layer3e(enc2)
        x = self.layer4e(enc3)

        x = self.layer4d(x)
        x = self.upsample(x)
        x = self.merge3(torch.concatenate([x, enc3], dim=1))
        x = self.layer3d(x)
        x = self.upsample(x)
        x = self.merge2(torch.concatenate([x, enc2], dim=1))
        x = self.layer2d(x)
        x = self.upsample(x)
        x = self.merge1(torch.concatenate([x, enc1], dim=1))
        x = self.layer1d(x)
        x = self.output(x)
        return x


def ukanet_18(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
              base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
              grid_range: List = [-1, 1]):
    return UKANet(KANBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  spline_order=spline_order, grid_size=grid_size, base_activation=base_activation,
                  grid_range=grid_range)


def fast_ukanet_18(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                   base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                   grid_range: List = [-1, 1]):
    return UKANet(FastKANBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  grid_size=grid_size, base_activation=base_activation,
                  grid_range=grid_range)


def ukalnet_18(input_channels, num_classes, groups: int = 1, degree: int = 3):
    return UKANet(KALNBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  degree=degree)


def ukagnet_18(input_channels, num_classes, groups: int = 1, degree: int = 3):
    return UKANet(KALNBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  degree=degree)


def ukacnet_18(input_channels, num_classes, groups: int = 1, degree: int = 3):
    return UKANet(KACNBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  degree=degree)
