from copy import deepcopy
from functools import partial
from typing import Callable, Optional, List, Type, Union

import torch
import torch.nn as nn

from kan_convs import BottleNeckKAGNConv2DLayer, KAGNFocalModulation2D, BottleNeckKAGNFocalModulation2D
from kan_convs import KALNConv2DLayer, KANConv2DLayer, KACNConv2DLayer, FastKANConv2DLayer, KAGNConv2DLayer
from kan_convs import SelfKAGNtention2D, BottleNeckSelfKAGNtention2D
from .model_utils import kan_conv1x1, fast_kan_conv1x1, kaln_conv1x1, kacn_conv1x1, kagn_conv1x1, \
    bottleneck_kagn_conv1x1
from .reskanet import KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, KANBottleneck, \
    FastKANBottleneck, KALNBottleneck, KACNBottleneck, KAGNBottleneck, KAGNBasicBlock, BottleneckKAGNBasicBlock


class UKANet(nn.Module):
    def __init__(self,
                 block: Type[Union[KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, KAGNBasicBlock,
                                   KANBottleneck, FastKANBottleneck, KALNBottleneck, KACNBottleneck, KAGNBottleneck,
                                   BottleneckKAGNBasicBlock]],
                 layers: List[int],
                 input_channels: int = 3,
                 num_classes: int = 1000,
                 groups: int = 1,
                 width_per_group: int = 64,
                 fcnv_kernel_size=7, fcnv_stride=1, fcnv_padding=3,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 width_scale: int = 1,
                 **kan_kwargs
                 ):
        super(UKANet, self).__init__()
        self.input_channels = input_channels
        self.inplanes = 8 * width_scale
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

        clean_params = deepcopy(kan_kwargs)
        clean_params.pop('l1_decay', None)

        if block in (KANBasicBlock, KANBottleneck):
            self.conv1 = KANConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size, stride=fcnv_stride,
                                        padding=fcnv_padding, **clean_params)

            self.merge1 = KANConv2DLayer((8 + 16) * width_scale * block.expansion,
                                         8 * width_scale * block.expansion, kernel_size=3,
                                         groups=groups, stride=1, padding=1, **clean_params)
            self.merge2 = KANConv2DLayer((32 + 16) * width_scale * block.expansion,
                                         16 * width_scale * block.expansion, kernel_size=3,
                                         groups=groups, stride=1, padding=1, **clean_params)
            self.merge3 = KANConv2DLayer((32 + 64) * width_scale * block.expansion,
                                         32 * width_scale * block.expansion, kernel_size=3,
                                         groups=groups, stride=1, padding=1, **clean_params)
        elif block in (FastKANBasicBlock, FastKANBottleneck):
            self.conv1 = FastKANConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                            stride=fcnv_stride, padding=fcnv_padding, **clean_params)

            self.merge1 = FastKANConv2DLayer((8 + 16) * width_scale * block.expansion,
                                             8 * width_scale * block.expansion, kernel_size=3,
                                             groups=groups, stride=1, padding=1, **clean_params)
            self.merge2 = FastKANConv2DLayer((32 + 16) * width_scale * block.expansion,
                                             16 * width_scale * block.expansion, kernel_size=3,
                                             groups=groups, stride=1, padding=1, **clean_params)
            self.merge3 = FastKANConv2DLayer((32 + 64) * width_scale * block.expansion,
                                             32 * width_scale * block.expansion, kernel_size=3,
                                             groups=groups, stride=1, padding=1, **clean_params)
        elif block in (KALNBasicBlock, KALNBottleneck):
            self.conv1 = KALNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **clean_params)
            self.merge1 = KALNConv2DLayer((8 + 16) * width_scale * block.expansion,
                                          8 * width_scale * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **clean_params)
            self.merge2 = KALNConv2DLayer((32 + 16) * width_scale * block.expansion,
                                          16 * width_scale * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **clean_params)
            self.merge3 = KALNConv2DLayer((32 + 64) * width_scale * block.expansion,
                                          32 * width_scale * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **clean_params)
        elif block in (KAGNBasicBlock, KAGNBottleneck):
            self.conv1 = KAGNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **clean_params)
            self.merge1 = KAGNConv2DLayer((8 + 16) * width_scale * block.expansion,
                                          8 * width_scale * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **clean_params)
            self.merge2 = KAGNConv2DLayer((32 + 16) * width_scale * block.expansion,
                                          16 * width_scale * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **clean_params)
            self.merge3 = KAGNConv2DLayer((32 + 64) * width_scale * block.expansion,
                                          32 * width_scale * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **clean_params)
        elif block in (BottleneckKAGNBasicBlock,):
            self.conv1 = BottleNeckKAGNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                                   stride=fcnv_stride, padding=fcnv_padding, **clean_params)
            self.merge1 = BottleNeckKAGNConv2DLayer((8 + 16) * width_scale * block.expansion,
                                                    8 * width_scale * block.expansion, kernel_size=3,
                                                    groups=groups, stride=1, padding=1, **clean_params)
            self.merge2 = BottleNeckKAGNConv2DLayer((32 + 16) * width_scale * block.expansion,
                                                    16 * width_scale * block.expansion, kernel_size=3,
                                                    groups=groups, stride=1, padding=1, **clean_params)
            self.merge3 = BottleNeckKAGNConv2DLayer((32 + 64) * width_scale * block.expansion,
                                                    32 * width_scale * block.expansion, kernel_size=3,
                                                    groups=groups, stride=1, padding=1, **clean_params)
        elif block in (KACNBasicBlock, KACNBottleneck):
            self.conv1 = KACNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **clean_params)
            self.merge1 = KACNConv2DLayer((8 + 16) * width_scale * block.expansion,
                                          8 * width_scale * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **clean_params)
            self.merge2 = KACNConv2DLayer((32 + 16) * width_scale * block.expansion,
                                          16 * width_scale * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **clean_params)
            self.merge3 = KACNConv2DLayer((32 + 64) * width_scale * block.expansion,
                                          32 * width_scale * block.expansion, kernel_size=3,
                                          groups=groups, stride=1, padding=1, **clean_params)
        else:
            raise TypeError(f"Block {type(block)} is not supported")

        self.layer1e = self._make_layer(block, 8 * block.expansion * width_scale, layers[0], **kan_kwargs)
        l1e_inplanes = self.inplanes
        self.layer2e = self._make_layer(block, 16 * block.expansion * width_scale, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0],
                                        **kan_kwargs)
        l2e_inplanes = self.inplanes
        self.layer3e = self._make_layer(block, 32 * block.expansion * width_scale, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1],
                                        **kan_kwargs)
        l3e_inplanes = self.inplanes
        self.layer4e = self._make_layer(block, 64 * block.expansion * width_scale, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2],
                                        **kan_kwargs)

        self.layer4d = self._make_layer(block, 64 * block.expansion * width_scale, layers[3], **kan_kwargs)
        self.inplanes = l1e_inplanes
        self.layer1d = self._make_layer(block, 8 * block.expansion * width_scale, layers[0], **kan_kwargs)
        self.inplanes = l2e_inplanes
        self.layer2d = self._make_layer(block, 16 * block.expansion * width_scale, layers[1], **kan_kwargs)
        self.inplanes = l3e_inplanes
        self.layer3d = self._make_layer(block, 32 * block.expansion * width_scale, layers[2], **kan_kwargs)

        self.output = nn.Conv2d(8 * block.expansion * width_scale, num_classes, kernel_size=1, padding=0, stride=1)

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
            elif block in (BottleneckKAGNBasicBlock,):
                conv1x1 = partial(bottleneck_kagn_conv1x1, **kan_kwargs)
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

    def forward(self, x, **kwargs):
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
              grid_range: List = [-1, 1], width_scale: int = 1,
              dropout: float = 0., l1_decay: float = 0.,
              affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return UKANet(KANBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  spline_order=spline_order,
                  grid_size=grid_size,
                  base_activation=base_activation,
                  grid_range=grid_range,
                  width_scale=width_scale,
                  affine=affine,
                  norm_layer=norm_layer,
                  dropout=dropout,
                  l1_decay=l1_decay
                  )


def fast_ukanet_18(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                   base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                   grid_range: List = [-1, 1], width_scale: int = 1,
                   dropout: float = 0., l1_decay: float = 0.,
                   affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return UKANet(FastKANBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  grid_size=grid_size,
                  base_activation=base_activation,
                  grid_range=grid_range,
                  width_scale=width_scale,
                  affine=affine,
                  norm_layer=norm_layer,
                  dropout=dropout,
                  l1_decay=l1_decay)


def ukalnet_18(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
               affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d,
               dropout: float = 0., l1_decay: float = 0.):
    return UKANet(KALNBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  degree=degree, width_scale=width_scale,
                  affine=affine,
                  norm_layer=norm_layer,
                  dropout=dropout,
                  l1_decay=l1_decay)


def ukagnet_18(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
               affine: bool = True, dropout: float = 0., l1_decay: float = 0.,
               norm_layer: nn.Module = nn.InstanceNorm2d):
    return UKANet(KAGNBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  degree=degree,
                  width_scale=width_scale,
                  affine=affine,
                  norm_layer=norm_layer,
                  dropout=dropout,
                  l1_decay=l1_decay)


def ukacnet_18(input_channels, num_classes, groups: int = 1, degree: int = 3,
               width_scale: int = 1,
               dropout: float = 0., l1_decay: float = 0.,
               affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return UKANet(KACNBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  degree=degree,
                  width_scale=width_scale,
                  affine=affine,
                  norm_layer=norm_layer,
                  dropout=dropout,
                  l1_decay=l1_decay)


def ukagnetnb_18(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                 affine: bool = True, dropout: float = 0., l1_decay: float = 0.,
                 norm_layer: nn.Module = nn.InstanceNorm2d):
    return UKANet(BottleneckKAGNBasicBlock, [2, 2, 2, 2],
                  input_channels=input_channels,
                  fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                  num_classes=num_classes,
                  groups=groups,
                  width_per_group=64,
                  degree=degree,
                  width_scale=width_scale,
                  affine=affine,
                  norm_layer=norm_layer,
                  dropout=dropout,
                  l1_decay=l1_decay)


class UKAGNet(nn.Module):
    def __init__(self,
                 input_channels: int = 3,
                 num_classes: int = 1,
                 unet_depth: int = 4,
                 unet_layers: int = 2,
                 groups: int = 1,
                 width_scale: int = 1,
                 use_bottleneck: bool = True,
                 mixer_type: str = 'conv',
                 degree: int = 3,
                 affine: bool = True,
                 dropout: float = 0.,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 inner_projection_attention: int = None,
                 focal_window=3,
                 focal_level=2,
                 focal_factor=2,
                 use_postln_in_modulation=True,
                 normalize_modulator=True,
                 full_kan: bool = True,
                 ):
        super(UKAGNet, self).__init__()
        self.unet_depth = unet_depth
        assert mixer_type in ['conv', 'self-att',
                              'focal'], f'Unsupported mixer type {mixer_type}; Mut be one of: conv, self-att, focal'

        attention = None
        if use_bottleneck:
            layer = BottleNeckKAGNConv2DLayer
            if mixer_type == "self-att":
                attention = BottleNeckSelfKAGNtention2D
            if mixer_type == "focal":
                attention = BottleNeckKAGNFocalModulation2D

        else:
            layer = KAGNConv2DLayer
            if mixer_type == "self-att":
                attention = SelfKAGNtention2D
            if mixer_type == "focal":
                attention = KAGNFocalModulation2D

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for depth_index in range(unet_depth):
            if depth_index == 0:
                layer_list_enc = [
                                     layer(input_channels, 16 * width_scale * 2 ** depth_index, 3,
                                           degree=degree, groups=groups, padding=1,
                                           stride=1, dilation=1, dropout=0, norm_layer=norm_layer, affine=affine),
                                 ] + [layer(16 * width_scale * 2 ** depth_index, 16 * width_scale * 2 ** depth_index, 3,
                                            degree=degree, groups=groups, padding=1,
                                            stride=1,
                                            dilation=1, dropout=dropout,
                                            norm_layer=norm_layer, affine=affine) for _ in range(unet_layers - 1)]

            else:
                layer_list_enc = [
                                     layer(16 * width_scale * 2 ** (depth_index - 1),
                                           16 * width_scale * 2 ** depth_index, 3,
                                           degree=degree, groups=groups, padding=1,
                                           stride=2, dilation=1, dropout=dropout, norm_layer=norm_layer, affine=affine),
                                 ] + [layer(16 * width_scale * 2 ** depth_index, 16 * width_scale * 2 ** depth_index, 3,
                                            degree=degree, groups=groups, padding=1,
                                            stride=1,
                                            dilation=1, dropout=dropout,
                                            norm_layer=norm_layer, affine=affine) for _ in
                                      range(unet_layers - 1)]

            self.encoder.append(nn.Sequential(*layer_list_enc))

        for depth_index in reversed(range(0, unet_depth-1)):
            if depth_index < unet_depth - 1:
                if attention is not None:

                    if mixer_type == "self-att":
                        layer_list_dec = [
                            attention(16 * 3 * width_scale * 2 ** depth_index, inner_projection_attention, kernel_size=3,
                                      degree=degree, groups=groups, padding=1,
                                      stride=1, dilation=1, norm_layer=norm_layer, affine=affine,

                                      dropout=dropout),
                        ]
                    elif mixer_type == "focal":
                        layer_list_dec = [
                            attention(16 * 3 * width_scale * 2 ** depth_index,
                                      degree=degree, dropout=dropout, norm_layer=norm_layer, affine=affine,
                                      focal_window=focal_window,
                                      focal_level=focal_level,
                                      focal_factor=focal_factor,
                                      use_postln_in_modulation=use_postln_in_modulation,
                                      normalize_modulator=normalize_modulator,
                                      full_kan=full_kan
                                      ), ]
                    else:
                        layer_list_dec = []
                else:
                    layer_list_dec = []

                layer_list_dec += [
                                      layer(16 * 3 * width_scale * 2 ** depth_index,
                                            16 * width_scale * 2 ** depth_index, 3,
                                            degree=degree, groups=groups, padding=1,
                                            stride=1, dilation=1, dropout=dropout, norm_layer=norm_layer,
                                            affine=affine),
                                  ] + [
                                      layer(16 * width_scale * 2 ** depth_index, 16 * width_scale * 2 ** depth_index, 3,
                                            degree=degree, groups=groups, padding=1,
                                            stride=1,
                                            dilation=1, dropout=dropout, norm_layer=norm_layer, affine=affine) for _ in
                                      range(unet_layers - 1)]

            else:
                if attention is not None:

                    if mixer_type == "self-att":
                        layer_list_dec = [
                            attention(16 * 3 * width_scale * 2 ** depth_index, inner_projection_attention,
                                      kernel_size=3,
                                      degree=degree, groups=groups, padding=1,
                                      stride=1, dilation=1, norm_layer=norm_layer, affine=affine,
                                      dropout=dropout),
                        ]
                    elif mixer_type == "focal":
                        layer_list_dec = [
                            attention(16 * 3 * width_scale * 2 ** depth_index,
                                      degree=degree, dropout=dropout, norm_layer=norm_layer, affine=affine,
                                      focal_window=focal_window,
                                      focal_level=focal_level,
                                      focal_factor=focal_factor,
                                      use_postln_in_modulation=use_postln_in_modulation,
                                      normalize_modulator=normalize_modulator,
                                      full_kan=full_kan
                                      ), ]
                    else:
                        layer_list_dec = []
                else:
                    layer_list_dec = []
                layer_list_dec += [
                                      layer(16 * 3 * width_scale * 2 ** depth_index,
                                            16 * width_scale * 2 ** depth_index, 3,
                                            degree=degree, groups=groups, padding=1,
                                            stride=1, dilation=1, dropout=dropout, norm_layer=norm_layer,
                                            affine=affine),
                                  ] + [
                                      layer(16 * width_scale * 2 ** depth_index, 16 * width_scale * 2 ** depth_index, 3,
                                            degree=degree, groups=groups, padding=1,
                                            stride=1,
                                            dilation=1, dropout=dropout, norm_layer=norm_layer, affine=affine,

                                            ) for _ in
                                      range(unet_layers - 1)]

            self.decoder.append(nn.Sequential(*layer_list_dec))
            self.output = nn.Conv2d(16 * width_scale * 2 ** depth_index, num_classes, 1)

            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, **kwargs):

        skips = []
        for block_index, block in enumerate(self.encoder):
            x = block(x)
            if block_index < self.unet_depth - 1:
                skips.append(x)

        for block in self.decoder:
            skip_x = skips.pop(-1)
            x = self.upsample(x)
            x = torch.concatenate([x, skip_x], dim=1)
            x = block(x)

        x = self.output(x)
        return x
