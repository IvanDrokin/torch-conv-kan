from functools import partial
from typing import Callable, List, Optional, Type, Union

import torch.nn as nn
from torch import Tensor, flatten

from kan_convs import KALNConv2DLayer, KANConv2DLayer, KACNConv2DLayer, FastKANConv2DLayer, KAGNConv2DLayer, \
    BottleNeckKAGNConv2DLayer
from .model_utils import kan_conv1x1, kagn_conv1x1, kacn_conv1x1, kaln_conv1x1, fast_kan_conv1x1, \
    bottleneck_kagn_conv1x1
from .model_utils import kan_conv3x3, kagn_conv3x3, kacn_conv3x3, kaln_conv3x3, fast_kan_conv3x3, moe_kaln_conv3x3, \
    bottleneck_kagn_conv3x3
from .model_utils import moe_bottleneck_kagn_conv3x3


class BasicBlockTemplate(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            conv1x1x1_fun,
            conv3x3x3_fun,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super().__init__()
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3_fun(inplanes, planes, stride=stride, groups=groups)
        self.conv2 = conv1x1x1_fun(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out


class KANBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 spline_order: int = 3,
                 grid_size: int = 5, base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                 grid_range: List = [-1, 1],
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(kan_conv1x1, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kan_conv3x3, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(KANBasicBlock, self).__init__(conv1x1x1_fun,
                                            conv3x3x3_fun,
                                            inplanes=inplanes,
                                            planes=planes,
                                            stride=stride,
                                            downsample=downsample,
                                            groups=groups,
                                            base_width=base_width,
                                            dilation=dilation)


class FastKANBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 grid_size: int = 5,
                 base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
                 grid_range: List = [-1, 1],
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(fast_kan_conv1x1, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                l1_decay=l1_decay, dropout=dropout, **norm_kwargs)
        conv3x3x3_fun = partial(fast_kan_conv3x3, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range,
                                l1_decay=l1_decay, dropout=dropout, **norm_kwargs)

        super(FastKANBasicBlock, self).__init__(conv1x1x1_fun,
                                                conv3x3x3_fun,
                                                inplanes=inplanes,
                                                planes=planes,
                                                stride=stride,
                                                downsample=downsample,
                                                groups=groups,
                                                base_width=base_width,
                                                dilation=dilation)


class KALNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kaln_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(KALNBasicBlock, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class KAGNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kagn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay,
                                norm_layer=norm_layer, **norm_kwargs)
        conv3x3x3_fun = partial(kagn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay,
                                norm_layer=norm_layer, **norm_kwargs)

        super(KAGNBasicBlock, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class BottleneckKAGNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 **norm_kwargs):
        conv1x1x1_fun = partial(bottleneck_kagn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay,
                                norm_layer=norm_layer, **norm_kwargs)
        conv3x3x3_fun = partial(bottleneck_kagn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay,
                                norm_layer=norm_layer, **norm_kwargs)

        super(BottleneckKAGNBasicBlock, self).__init__(conv1x1x1_fun,
                                                       conv3x3x3_fun,
                                                       inplanes=inplanes,
                                                       planes=planes,
                                                       stride=stride,
                                                       downsample=downsample,
                                                       groups=groups,
                                                       base_width=base_width,
                                                       dilation=dilation)


class KACNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kacn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kacn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(KACNBasicBlock, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class BottleneckTemplate(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            conv1x1x1_fun,
            conv3x3x3_fun,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1_fun(inplanes, width)
        # self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3_fun(width, width, stride=stride, groups=groups, dilation=dilation)
        # self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1_fun(width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


class KANBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 spline_order: int = 3,
                 grid_size: int = 5, base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                 grid_range: List = [-1, 1],
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(kan_conv1x1, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range, dropout=dropout,
                                l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kan_conv3x3, spline_order=spline_order, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range, dropout=dropout,
                                l1_decay=l1_decay, **norm_kwargs)

        super(KANBottleneck, self).__init__(conv1x1x1_fun,
                                            conv3x3x3_fun,
                                            inplanes=inplanes,
                                            planes=planes,
                                            stride=stride,
                                            downsample=downsample,
                                            groups=groups,
                                            base_width=base_width,
                                            dilation=dilation)


class FastKANBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 grid_size: int = 5,
                 base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
                 grid_range: List = [-1, 1],
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(fast_kan_conv1x1, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range, dropout=dropout,
                                l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(fast_kan_conv3x3, grid_size=grid_size,
                                base_activation=base_activation, grid_range=grid_range, dropout=dropout,
                                l1_decay=l1_decay, **norm_kwargs)

        super(FastKANBottleneck, self).__init__(conv1x1x1_fun,
                                                conv3x3x3_fun,
                                                inplanes=inplanes,
                                                planes=planes,
                                                stride=stride,
                                                downsample=downsample,
                                                groups=groups,
                                                base_width=base_width,
                                                dilation=dilation)


class KALNBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kaln_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(KALNBottleneck, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class KAGNBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 **norm_kwargs):
        conv1x1_fun = partial(kagn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay,
                              norm_layer=norm_layer, **norm_kwargs)
        conv3x3_fun = partial(kagn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay,
                              norm_layer=norm_layer, **norm_kwargs)

        super(KAGNBottleneck, self).__init__(conv1x1_fun,
                                             conv3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class BottleneckKAGNBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 **norm_kwargs):
        conv1x1_fun = partial(bottleneck_kagn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay,
                              norm_layer=norm_layer, **norm_kwargs)
        conv3x3_fun = partial(bottleneck_kagn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay,
                              norm_layer=norm_layer, **norm_kwargs)

        super(BottleneckKAGNBottleneck, self).__init__(conv1x1_fun,
                                                       conv3x3_fun,
                                                       inplanes=inplanes,
                                                       planes=planes,
                                                       stride=stride,
                                                       downsample=downsample,
                                                       groups=groups,
                                                       base_width=base_width,
                                                       dilation=dilation)


class MoEKALNBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 num_experts: int = 8,
                 noisy_gating: bool = True,
                 k: int = 2,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs
                 ):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(moe_kaln_conv3x3, degree=degree, num_experts=num_experts,
                                k=k, noisy_gating=noisy_gating, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(MoEKALNBottleneck, self).__init__(conv1x1x1_fun,
                                                conv3x3x3_fun,
                                                inplanes=inplanes,
                                                planes=planes,
                                                stride=stride,
                                                downsample=downsample,
                                                groups=groups,
                                                base_width=base_width,
                                                dilation=dilation)

    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        identity = x

        out = self.conv1(x)

        out, moe_loss = self.conv2(out, train=train)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out, moe_loss


class MoEKALNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 num_experts: int = 8,
                 noisy_gating: bool = True,
                 k: int = 2,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(moe_kaln_conv3x3, degree=degree, num_experts=num_experts,
                                k=k, noisy_gating=noisy_gating, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(MoEKALNBasicBlock, self).__init__(conv1x1x1_fun,
                                                conv3x3x3_fun,
                                                inplanes=inplanes,
                                                planes=planes,
                                                stride=stride,
                                                downsample=downsample,
                                                groups=groups,
                                                base_width=base_width,
                                                dilation=dilation)

    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        identity = x

        out, moe_loss = self.conv1(x, train=train)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out, moe_loss


class KACNBottleneck(BottleneckTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kacn_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(kacn_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(KACNBottleneck, self).__init__(conv1x1x1_fun,
                                             conv3x3x3_fun,
                                             inplanes=inplanes,
                                             planes=planes,
                                             stride=stride,
                                             downsample=downsample,
                                             groups=groups,
                                             base_width=base_width,
                                             dilation=dilation)


class MoEBottleneckKAGNBasicBlock(BasicBlockTemplate):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 degree: int = 3,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 num_experts: int = 8,
                 noisy_gating: bool = True,
                 k: int = 2,
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 **norm_kwargs):
        conv1x1x1_fun = partial(kaln_conv1x1, degree=degree, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)
        conv3x3x3_fun = partial(moe_bottleneck_kagn_conv3x3, degree=degree, num_experts=num_experts,
                                k=k, noisy_gating=noisy_gating, dropout=dropout, l1_decay=l1_decay, **norm_kwargs)

        super(MoEBottleneckKAGNBasicBlock, self).__init__(conv1x1x1_fun,
                                                          conv3x3x3_fun,
                                                          inplanes=inplanes,
                                                          planes=planes,
                                                          stride=stride,
                                                          downsample=downsample,
                                                          groups=groups,
                                                          base_width=base_width,
                                                          dilation=dilation)

    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        identity = x

        out, moe_loss = self.conv1(x, train=train)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out, moe_loss


class ResKANet(nn.Module):
    def __init__(
            self,
            block: Type[Union[KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, KAGNBasicBlock,
                              KANBottleneck, FastKANBottleneck, KALNBottleneck, KACNBottleneck, KAGNBottleneck,
                              BottleneckKAGNBottleneck, BottleneckKAGNBasicBlock]],
            layers: List[int],
            input_channels: int = 3,
            use_first_maxpool: bool = True,
            mp_kernel_size: int = 3, mp_stride: int = 2, mp_padding: int = 1,
            fcnv_kernel_size: int = 7, fcnv_stride: int = 2, fcnv_padding: int = 3,
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            width_scale: int = 1,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            dropout_linear: float = 0.25,
            hidden_layer_dim: int = None,
            norm_layer: nn.Module = nn.BatchNorm2d,
            **kan_kwargs
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.inplanes = 8 * width_scale
        self.hidden_layer_dim = hidden_layer_dim
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
        self.use_first_maxpool = use_first_maxpool

        self.hidden_layer = None

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        kan_kwargs_clean.pop('groups', None)

        kan_kwargs_fc = kan_kwargs.copy()
        kan_kwargs_fc.pop('groups', None)
        kan_kwargs_fc.pop('dropout', None)
        kan_kwargs_fc['dropout'] = dropout_linear

        if hidden_layer_dim is not None:
            fc_layers = [64 * width_scale * block.expansion, hidden_layer_dim, num_classes]
        else:
            fc_layers = [64 * width_scale * block.expansion, num_classes]

        if block in (KANBasicBlock, KANBottleneck):
            self.conv1 = KANConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size, stride=fcnv_stride,
                                        padding=fcnv_padding, **kan_kwargs_clean)
            # self.fc = KAN(fc_layers, **kan_kwargs_fc)

        elif block in (FastKANBasicBlock, FastKANBottleneck):
            self.conv1 = FastKANConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                            stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
            # self.fc = FastKAN(fc_layers, **kan_kwargs_fc)

        elif block in (KALNBasicBlock, KALNBottleneck):
            self.conv1 = KALNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
            # self.fc = KALN(fc_layers, **kan_kwargs_fc)
        elif block in (KAGNBasicBlock, KAGNBottleneck):
            self.conv1 = KAGNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding,
                                         norm_layer=norm_layer, **kan_kwargs_clean)
            # self.fc = KAGN(fc_layers, **kan_kwargs_fc)
        elif block in (BottleneckKAGNBottleneck, BottleneckKAGNBasicBlock):
            self.conv1 = BottleNeckKAGNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                                   stride=fcnv_stride, padding=fcnv_padding,
                                                   norm_layer=norm_layer, **kan_kwargs_clean)
            # self.fc = KAGN(fc_layers, **kan_kwargs_fc)
        elif block in (KACNBasicBlock, KACNBottleneck):
            self.conv1 = KACNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
            # self.fc = KACN(fc_layers, **kan_kwargs_fc)
        else:
            raise TypeError(f"Block {type(block)} is not supported")
        self.maxpool = None
        if use_first_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=mp_padding)

        self.layer1 = self._make_layer(block, 8 * width_scale, layers[0],
                                       norm_layer=norm_layer, **kan_kwargs)
        self.layer2 = self._make_layer(block, 16 * width_scale, layers[1], stride=2, norm_layer=norm_layer,
                                       dilate=replace_stride_with_dilation[0],
                                       **kan_kwargs)
        self.layer3 = self._make_layer(block, 32 * width_scale, layers[2], stride=2, norm_layer=norm_layer,
                                       dilate=replace_stride_with_dilation[1],
                                       **kan_kwargs)
        self.layer4 = self._make_layer(block, 64 * width_scale, layers[3], stride=2, norm_layer=norm_layer,
                                       dilate=replace_stride_with_dilation[2],
                                       **kan_kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout_linear)
        self.fc = nn.Linear(64 * width_scale * block.expansion if self.hidden_layer is None else hidden_layer_dim,
                            num_classes)

    def _make_layer(
            self,
            block: Type[Union[
                KANBasicBlock, FastKANBasicBlock, KALNBasicBlock, KACNBasicBlock, BottleneckKAGNBasicBlock, BottleneckKAGNBottleneck,
                KANBottleneck, FastKANBottleneck, KALNBottleneck, KACNBottleneck]],
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
            elif block in (KAGNBasicBlock, KAGNBottleneck):
                conv1x1 = partial(kagn_conv1x1, **kan_kwargs)
            elif block in (KACNBasicBlock, KACNBottleneck):
                conv1x1 = partial(kacn_conv1x1, **kan_kwargs)
            elif block in (BottleneckKAGNBasicBlock, BottleneckKAGNBottleneck):
                conv1x1 = partial(bottleneck_kagn_conv1x1, **kan_kwargs)
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
                    **kan_kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        if self.use_first_maxpool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
        x = flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self._forward_impl(x)


class MoEResKANet(nn.Module):
    def __init__(
            self,
            block: Type[Union[MoEKALNBottleneck, MoEKALNBasicBlock, MoEBottleneckKAGNBasicBlock]],
            layers: List[int],
            input_channels: int = 3,
            use_first_maxpool: bool = True,
            mp_kernel_size: int = 3, mp_stride: int = 2, mp_padding: int = 1,
            fcnv_kernel_size: int = 7, fcnv_stride: int = 2, fcnv_padding: int = 3,
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            width_scale: int = 1,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            num_experts: int = 8,
            noisy_gating: bool = True,
            k: int = 2,
            hidden_layer_dim: int = None,
            dropout_linear: float = 0.0,
            **kan_kwargs
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.inplanes = 16 * width_scale
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
        self.use_first_maxpool = use_first_maxpool

        self.hidden_layer = None

        kan_kwargs_clean = kan_kwargs.copy()
        kan_kwargs_clean.pop('l1_decay', None)
        if block in (MoEKALNBottleneck, MoEKALNBasicBlock):
            self.conv1 = KALNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
            if hidden_layer_dim is not None:
                self.hidden_layer = kaln_conv1x1(64 * width_scale * block.expansion, hidden_layer_dim, **kan_kwargs)
        elif block in (MoEBottleneckKAGNBasicBlock, ):
            self.conv1 = BottleNeckKAGNConv2DLayer(input_channels, self.inplanes, kernel_size=fcnv_kernel_size,
                                         stride=fcnv_stride, padding=fcnv_padding, **kan_kwargs_clean)
            if hidden_layer_dim is not None:
                self.hidden_layer = bottleneck_kagn_conv1x1(64 * width_scale * block.expansion,
                                                            hidden_layer_dim, **kan_kwargs)
        else:
            raise TypeError(f"Block {type(block)} is not supported")
        self.maxpool = None
        if use_first_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=mp_padding)

        self.layer1 = self._make_layer(block, 8 * width_scale, layers[0],
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k, **kan_kwargs)
        self.layer2 = self._make_layer(block, 16 * width_scale, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k,
                                       **kan_kwargs)
        self.layer3 = self._make_layer(block, 32 * width_scale, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k,
                                       **kan_kwargs)
        self.layer4 = self._make_layer(block, 64 * width_scale, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       num_experts=num_experts, noisy_gating=noisy_gating, k=k,
                                       **kan_kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * width_scale * block.expansion if self.hidden_layer is None else hidden_layer_dim,
                            num_classes)
        self.drop = nn.Dropout(p=dropout_linear)

    def _make_layer(
            self,
            block: Type[Union[MoEKALNBottleneck,]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            num_experts: int = 8,
            noisy_gating: bool = True,
            k: int = 2,
            **kan_kwargs
    ) -> nn.Module:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block in (MoEKALNBottleneck, MoEKALNBasicBlock, MoEBottleneckKAGNBasicBlock):
                kan_kwargs.pop('num_experts', None)
                kan_kwargs.pop('noisy_gating', None)
                kan_kwargs.pop('k', None)
                conv1x1 = partial(kaln_conv1x1, **kan_kwargs)
            else:
                raise TypeError(f"Block {type(block)} is not supported")

            downsample = conv1x1(self.inplanes, planes * block.expansion, stride=stride)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                base_width=self.base_width, dilation=previous_dilation, num_experts=num_experts,
                noisy_gating=noisy_gating, k=k, **kan_kwargs
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
                    num_experts=num_experts,
                    noisy_gating=noisy_gating,
                    k=k,
                    **kan_kwargs
                )
            )

        return nn.ModuleList(layers)

    def _forward_layer(self, layer, x, train):
        moe_loss = 0
        for block in layer:
            x, _moe_loss = block(x, train)
            moe_loss += _moe_loss
        return x, moe_loss

    def _forward_impl(self, x: Tensor, train: bool = True) -> Tensor:
        x = self.conv1(x)
        if self.use_first_maxpool:
            x = self.maxpool(x)

        x, moe_loss1 = self._forward_layer(self.layer1, x, train)
        x, moe_loss2 = self._forward_layer(self.layer2, x, train)
        x, moe_loss3 = self._forward_layer(self.layer3, x, train)
        x, moe_loss4 = self._forward_layer(self.layer4, x, train)

        x = self.avgpool(x)
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
        x = flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)

        return x, (moe_loss1 + moe_loss2 + moe_loss3 + moe_loss4) / 4

    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        return self._forward_impl(x, train)


def reskanet_18x32p(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
                    base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                    grid_range: List = [-1, 1], hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                    dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KANBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    spline_order=spline_order, grid_size=grid_size, base_activation=base_activation,
                    grid_range=grid_range, hidden_layer_dim=hidden_layer_dim,
                    dropout_linear=dropout_linear,
                    dropout=dropout,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def fast_reskanet_18x32p(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                         base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                         grid_range: List = [-1, 1], hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                         dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(FastKANBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    grid_size=grid_size, base_activation=base_activation,
                    grid_range=grid_range, hidden_layer_dim=hidden_layer_dim,
                    dropout_linear=dropout_linear,
                    dropout=dropout,
                    l1_decay=l1_decay,
                    affine=affine)


def reskalnet_18x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                     hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                     dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KALNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def reskagnet_18x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                     hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                     dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KAGNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def reskagnet18(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                dropout_linear: float = 0.25, affine: bool = False, norm_layer: nn.Module = nn.InstanceNorm2d):
    return ResKANet(KAGNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=7, fcnv_stride=2, fcnv_padding=3,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    norm_layer=norm_layer,
                    affine=affine
                    )


def reskalnet_18x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                     hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                     dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KALNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def moe_reskalnet_18x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                         num_experts: int = 8, noisy_gating: bool = True, k: int = 2,
                         hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                         dropout_linear: float = 0.25, affine: bool = False):
    return MoEResKANet(MoEKALNBasicBlock, [2, 2, 2, 2],
                       input_channels=input_channels,
                       use_first_maxpool=True,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine)


def reskacnet_18x32p(input_channels, num_classes, groups: int = 1, degree: int = 3,
                     hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                     dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KACNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def reskalnet_50x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                     dropout: float = 0.15, dropout_linear: float = 0.25, l1_decay: float = 0.0,
                     hidden_layer_dim=None, affine: bool = False):
    return ResKANet(KALNBottleneck, [3, 4, 6, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    hidden_layer_dim=hidden_layer_dim,
                    affine=affine)


def reskagnet50(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                dropout: float = 0.15, dropout_linear: float = 0.25, l1_decay: float = 0.0,
                hidden_layer_dim=None, affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return ResKANet(KAGNBottleneck, [3, 4, 6, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=7, fcnv_stride=2, fcnv_padding=3,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    hidden_layer_dim=hidden_layer_dim,
                    norm_layer=norm_layer,
                    affine=affine)


def reskagnet_bn50(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                   dropout: float = 0.15, dropout_linear: float = 0.25, l1_decay: float = 0.0,
                   hidden_layer_dim=None, affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return ResKANet(BottleneckKAGNBottleneck, [3, 4, 6, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=7, fcnv_stride=2, fcnv_padding=3,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    hidden_layer_dim=hidden_layer_dim,
                    norm_layer=norm_layer,
                    affine=affine)


def reskagnet101(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                 dropout: float = 0.15, dropout_linear: float = 0.25, l1_decay: float = 0.0,
                 hidden_layer_dim=None, affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return ResKANet(KAGNBottleneck, [3, 4, 23, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=7, fcnv_stride=2, fcnv_padding=3,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    hidden_layer_dim=hidden_layer_dim,
                    norm_layer=norm_layer,
                    affine=affine)


def reskagnet152(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                 dropout: float = 0.15, dropout_linear: float = 0.25, l1_decay: float = 0.0,
                 hidden_layer_dim=None, affine: bool = True, norm_layer: nn.Module = nn.InstanceNorm2d):
    return ResKANet(KAGNBottleneck, [3, 8, 36, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=7, fcnv_stride=2, fcnv_padding=3,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    hidden_layer_dim=hidden_layer_dim,
                    norm_layer=norm_layer,
                    affine=affine)


def moe_reskalnet_50x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                         num_experts: int = 8, noisy_gating: bool = True, k: int = 2,
                         hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                         l1_decay: float = 0.0, affine: bool = False):
    return MoEResKANet(MoEKALNBottleneck, [3, 4, 6, 3],
                       input_channels=input_channels,
                       use_first_maxpool=True,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine
                       )


def reskalnet_101x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                      l1_decay: float = 0.0, affine: bool = False):
    return ResKANet(KALNBottleneck, [3, 4, 23, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def reskagnet_101x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                      l1_decay: float = 0.0, affine: bool = False):
    return ResKANet(KAGNBottleneck, [3, 4, 23, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def reskalnet_101x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                      l1_decay: float = 0.0, affine: bool = False):
    return ResKANet(KALNBottleneck, [3, 4, 23, 3],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def moe_reskalnet_101x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                          num_experts: int = 8, noisy_gating: bool = True, k: int = 2,
                          hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                          l1_decay: float = 0.0, affine: bool = False):
    return MoEResKANet(MoEKALNBottleneck, [3, 4, 23, 3],
                       input_channels=input_channels,
                       use_first_maxpool=True,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine
                       )


def reskalnet_152x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                      l1_decay: float = 0.0, affine: bool = False):
    return ResKANet(KALNBottleneck, [3, 8, 36, 3],
                    input_channels=input_channels,
                    use_first_maxpool=True,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def reskalnet_152x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                      hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                      l1_decay: float = 0.0, affine: bool = False):
    return ResKANet(KALNBottleneck, [3, 8, 36, 3],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine)


def moe_reskalnet_152x64p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                          num_experts: int = 8, noisy_gating: bool = True, k: int = 2,
                          hidden_layer_dim=None, dropout: float = 0.15, dropout_linear: float = 0.25,
                          l1_decay: float = 0.0, affine: bool = False):
    return MoEResKANet(MoEKALNBottleneck, [3, 8, 36, 3],
                       input_channels=input_channels,
                       use_first_maxpool=True,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine
                       )


def reskagnetbn_18x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                       hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                       dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(BottleneckKAGNBasicBlock, [2, 2, 2, 2],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def reskagnetbn_moe_18x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                           hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                           dropout_linear: float = 0.25, affine: bool = False,

                           num_experts: int = 8,
                           noisy_gating: bool = True,
                           k: int = 2
                           ):
    return MoEResKANet(MoEBottleneckKAGNBasicBlock, [2, 2, 2, 2],
                       input_channels=input_channels,
                       use_first_maxpool=False,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k
                       )




def reskagnetbn_34x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                       hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                       dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(BottleneckKAGNBasicBlock, [3, 4, 6, 3],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )


def reskagnetbn_moe_34x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                           hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                           dropout_linear: float = 0.25, affine: bool = False,

                           num_experts: int = 8,
                           noisy_gating: bool = True,
                           k: int = 2
                           ):
    return MoEResKANet(MoEBottleneckKAGNBasicBlock, [3, 4, 6, 3],
                       input_channels=input_channels,
                       use_first_maxpool=False,
                       fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                       num_classes=num_classes,
                       groups=groups,
                       width_per_group=64,
                       degree=degree,
                       width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                       dropout=dropout,
                       dropout_linear=dropout_linear,
                       l1_decay=l1_decay,
                       affine=affine,
                       num_experts=num_experts,
                       noisy_gating=noisy_gating,
                       k=k
                       )


def reskagnet_34x32p(input_channels, num_classes, groups: int = 1, degree: int = 3, width_scale: int = 1,
                     hidden_layer_dim=None, dropout: float = 0.0, l1_decay: float = 0.0,
                     dropout_linear: float = 0.25, affine: bool = False):
    return ResKANet(KAGNBasicBlock, [3, 4, 6, 3],
                    input_channels=input_channels,
                    use_first_maxpool=False,
                    fcnv_kernel_size=3, fcnv_stride=1, fcnv_padding=1,
                    num_classes=num_classes,
                    groups=groups,
                    width_per_group=64,
                    degree=degree,
                    width_scale=width_scale, hidden_layer_dim=hidden_layer_dim,
                    dropout=dropout,
                    dropout_linear=dropout_linear,
                    l1_decay=l1_decay,
                    affine=affine
                    )