# Based on this https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg16
from functools import partial
from math import prod
from typing import cast, Dict, List, Optional, Union, Tuple, Callable

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from kans import mlp_kan, mlp_fastkan, mlp_kacn, mlp_kagn, mlp_kaln, mlp_wav_kan
from .extra_layers import MatryoshkaHead
from .model_utils import kan_conv3x3, kaln_conv3x3, fast_kan_conv3x3, kacn_conv3x3, kagn_conv3x3, wav_kan_conv3x3
from .model_utils import moe_bottleneck_kagn_conv3x3
from .model_utils import moe_kagn_conv3x3, bottleneck_kagn_conv3x3, self_kagn_conv3x3, self_bottleneck_kagn_conv3x3

cfgs: Dict[str, List[Union[str, int]]] = {
    "VGG11": [16, "M", 32, "M", 64, 64, "M", 128, 128, "M", 128, 128],
    "VGG13": [16, 16, "M", 32, 32, "M", 64, 64, "M", 128, 128, "M", 128, 128],
    "VGG16": [16, 16, "M", 32, 32, "M", 64, 64, 64, "M", 128, 128, 128, "M", 128, 128, 128],
    "VGG19": [16, 16, "M", 32, 32, "M", 64, 64, 64, 64, "M", 128, 128, 128, 128, "M", 128, 128, 128, 128],

    "VGG11sa": [16, "M", 32, "M", 64, 64, "SAM", 128, 128, "SAM", 128, 128],
    "VGG13sa": [16, 16, "M", 32, 32, "M", 64, 64, "SAM", 128, 128, "SAM", 128, 128],
    "VGG16sa": [16, 16, "M", 32, 32, "M", 64, 64, 64, "SAM", 128, 128, 128, "SAM", 128, 128, 128],
    "VGG19sa": [16, 16, "M", 32, 32, "M", 64, 64, 64, 64, "SAM", 128, 128, 128, 128, "SAM", 128, 128, 128, 128],

    "VGG11v2": [16, "M", 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M", 128, 128],
    "VGG13v2": [16, 16, "M", 32, 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M", 128, 128],
    "VGG16v2": [16, 16, "M", 32, 32, "M", 64, 64, 64, "M", 128, 128, 128, "M", 128, 128, 128, "M", 128, 128],
    "VGG19v2": [16, 16, "M", 32, 32, "M", 64, 64, 64, 64, "M", 128, 128, 128, 128, "M", 128, 128, 128, 128, "M", 128,
                128],
    "VGG11v3": [16, "M", 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M", 128, 256],
    "VGG13v3": [16, 16, "M", 32, 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M", 128, 256],
    "VGG16v3": [16, 16, "M", 32, 32, "M", 64, 64, 64, "M", 128, 128, 128, "M", 128, 128, 128, "M", 128, 256],
    "VGG19v3": [16, 16, "M", 32, 32, "M", 64, 64, 64, 64, "M", 128, 128, 128, 128, "M", 128, 128, 128, 128, "M", 128,
                256],
    "VGG11v4": [16, "M", 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M", 256, 256],
    "VGG13v4": [16, 16, "M", 32, 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M", 256, 256],
    "VGG16v4": [16, 16, "M", 32, 32, "M", 64, 64, 64, "M", 128, 128, 128, "M", 128, 128, 128, "M", 256, 256],
    "VGG19v4": [16, 16, "M", 32, 32, "M", 64, 64, 64, 64, "M", 128, 128, 128, 128, "M", 128, 128, 128, 128, "M", 256,
                256],
    "VGG11v4sa": [16, "M", 32, "M", 64, 64, "M", 128, 128, "SAM", 128, 128, "SAM", 256, 256],
    "VGG13v4sa": [16, 16, "M", 32, 32, "M", 64, 64, "M", 128, 128, "SAM", 128, 128, "SAM", 256, 256],
    "VGG16v4sa": [16, 16, "M", 32, 32, "M", 64, 64, 64, "M", 128, 128, 128, "SAM", 128, 128, 128, "SAM", 256, 256],
    "VGG19v4sa": [16, 16, "M", 32, 32, "M", 64, 64, 64, 64, "M", 128, 128, 128, 128, "SAM", 128, 128, 128, 128, "SAM", 256,
                256]
}


class VGG(nn.Module):
    def __init__(
            self, features: nn.ModuleList, classifier: nn.Module, expected_feature_shape: Tuple = (7, 7)
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(expected_feature_shape)
        self.classifier = classifier
        self.expected_feature_shape = expected_feature_shape

    @staticmethod
    def make_layers(cfg: List[Union[str, int]],
                    head_type,
                    conv_fun,
                    conv_fun_first,
                    kan_fun,
                    kan_att_fun=None,
                    expected_feature_shape: Tuple = (7, 7),
                    num_input_features: int = 3,
                    num_classes: int = 1000,
                    width_scale: int = 1,
                    head_dropout: float = 0.5,
                    last_attention: bool = False
                    ):
        layers: List[nn.Module] = []
        in_channels = num_input_features
        for l_index, v in enumerate(cfg):
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'SAM':
                layers += [kan_att_fun(in_channels), nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                if l_index == 0:
                    conv2d = conv_fun_first(in_channels, v * width_scale)
                elif l_index == len(cfg) - 1 and last_attention:
                    conv2d = kan_att_fun(in_channels)
                else:
                    conv2d = conv_fun(in_channels, v * width_scale)
                layers.append(conv2d)
                in_channels = v * width_scale
        if head_type == 'KAN':
            classification = kan_fun([in_channels * prod(expected_feature_shape), num_classes])
        elif head_type == 'HiddenKAN':
            classification = nn.Sequential(
                kan_fun([in_channels * prod(expected_feature_shape), 512]),
                nn.Dropout(p=head_dropout),
                nn.Linear(512, num_classes),
            )
        elif head_type == 'VGG':
            classification = nn.Sequential(
                nn.Linear(in_channels * prod(expected_feature_shape), 1024),
                nn.ReLU(True),
                nn.Dropout(p=head_dropout),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Dropout(p=head_dropout),
                nn.Linear(1024, num_classes),
            )
        elif head_type == 'Linear':
            classification = nn.Sequential(
                nn.Dropout(p=head_dropout),
                nn.Linear(in_channels * prod(expected_feature_shape), num_classes),
            )
        elif head_type == 'Matryoshka':
            classification = nn.Sequential(
                nn.Linear(in_channels * prod(expected_feature_shape), 1024),
                nn.ReLU(True),
                nn.Dropout(p=head_dropout),
                MatryoshkaHead([2 ** n for n in range(6, 11)], num_classes, efficient=True),
            )
        elif head_type == 'KANtryoshka':
            classification = nn.Sequential(
                kan_fun([in_channels * prod(expected_feature_shape), 512]),
                nn.Dropout(p=head_dropout),
                MatryoshkaHead([2 ** n for n in range(5, 10)], num_classes, efficient=True),
            )
        else:
            classification = None

        return nn.ModuleList(layers), classification

    def forward_features(self, x):
        moe_loss = None
        for layer in self.features:
            x = layer(x)
            if isinstance(x, tuple):
                x, _moe = x
                if moe_loss is None:
                    moe_loss = _moe
                else:
                    moe_loss = moe_loss + _moe
        if moe_loss is not None:
            return x, moe_loss
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        moe_loss = None
        x = self.forward_features(x)
        if isinstance(x, tuple):
            x, moe_loss = x
        if self.classifier is None:
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if moe_loss is not None:
            return x, moe_loss
        return x


class VGGKAN(VGG, PyTorchModelHubMixin):
    def __init__(self, input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
                 base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                 grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                 dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
                 expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
                 norm_layer: nn.Module = nn.InstanceNorm2d):
        conv_fun = partial(kan_conv3x3, spline_order=spline_order, grid_size=grid_size,
                           base_activation=base_activation, grid_range=grid_range,
                           dropout=dropout, l1_decay=l1_decay, groups=groups, affine=affine,
                           norm_layer=norm_layer)
        conv_fun_first = partial(kan_conv3x3, spline_order=spline_order, grid_size=grid_size,
                                 base_activation=base_activation, grid_range=grid_range, l1_decay=l1_decay,
                                 affine=affine, norm_layer=norm_layer)
        kan_fun = partial(mlp_kan, spline_order=spline_order, grid_size=grid_size,
                          base_activation=base_activation, grid_range=grid_range,
                          dropout=dropout_linear, l1_decay=l1_decay)

        features, head = self.make_layers(cfgs[vgg_type], head_type, conv_fun, conv_fun_first, kan_fun,
                                          expected_feature_shape=expected_feature_shape,
                                          num_input_features=input_channels, num_classes=num_classes,
                                          width_scale=width_scale,
                                          head_dropout=dropout_linear)
        super().__init__(features, head, expected_feature_shape)


def vggkan(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
           base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
           grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
           dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
           expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
           norm_layer: nn.Module = nn.InstanceNorm2d):
    return VGGKAN(input_channels, num_classes, groups=groups, spline_order=spline_order, grid_size=grid_size,
                  base_activation=base_activation,
                  grid_range=grid_range, dropout=dropout, l1_decay=l1_decay,
                  dropout_linear=dropout_linear, vgg_type=vgg_type, head_type=head_type,
                  expected_feature_shape=expected_feature_shape, width_scale=width_scale, affine=affine,
                  norm_layer=norm_layer)


class FastVGGKAN(VGG, PyTorchModelHubMixin):
    def __init__(self, input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                 base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                 grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                 dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
                 expected_feature_shape: Tuple = (7, 7),
                 width_scale: int = 1, affine: bool = False,
                 norm_layer: nn.Module = nn.InstanceNorm2d):
        conv_fun = partial(fast_kan_conv3x3, grid_size=grid_size,
                           base_activation=base_activation, grid_range=grid_range,
                           dropout=dropout, l1_decay=l1_decay, groups=groups, affine=affine,
                           norm_layer=norm_layer)
        conv_fun_first = partial(fast_kan_conv3x3, grid_size=grid_size,
                                 base_activation=base_activation, grid_range=grid_range, l1_decay=l1_decay,
                                 affine=affine, norm_layer=norm_layer)
        kan_fun = partial(mlp_fastkan, grid_size=grid_size,
                          base_activation=base_activation, grid_range=grid_range,
                          dropout=dropout_linear, l1_decay=l1_decay)

        features, head = self.make_layers(cfgs[vgg_type], head_type, conv_fun, conv_fun_first, kan_fun,
                                          expected_feature_shape=expected_feature_shape,
                                          num_input_features=input_channels, num_classes=num_classes,
                                          width_scale=width_scale,
                                          head_dropout=dropout_linear)
        super().__init__(features, head, expected_feature_shape)


def fast_vggkan(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
                expected_feature_shape: Tuple = (7, 7),
                width_scale: int = 1, affine: bool = False,
                norm_layer: nn.Module = nn.InstanceNorm2d):
    return FastVGGKAN(input_channels, num_classes, groups=groups, grid_size=grid_size,
                      base_activation=base_activation,
                      grid_range=grid_range, dropout=dropout, l1_decay=l1_decay,
                      dropout_linear=dropout_linear, vgg_type=vgg_type, head_type=head_type,
                      expected_feature_shape=expected_feature_shape,
                      width_scale=width_scale, affine=affine, norm_layer=norm_layer)


class VGGKALN(VGG, PyTorchModelHubMixin):
    def __init__(self, input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
                 expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
                 norm_layer: nn.Module = nn.InstanceNorm2d):
        conv_fun = partial(kaln_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay, groups=groups,
                           affine=affine, norm_layer=norm_layer)
        conv_fun_first = partial(kaln_conv3x3, degree=degree, l1_decay=l1_decay, affine=affine, norm_layer=norm_layer)
        kan_fun = partial(mlp_kaln, degree=degree, dropout=dropout_linear, l1_decay=l1_decay)

        features, head = self.make_layers(cfgs[vgg_type], head_type, conv_fun, conv_fun_first, kan_fun,
                                          expected_feature_shape=expected_feature_shape,
                                          num_input_features=input_channels, num_classes=num_classes,
                                          width_scale=width_scale,
                                          head_dropout=dropout_linear)
        super().__init__(features, head, expected_feature_shape)


def vggkaln(input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0, l1_decay: float = 0.0,
            dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
            expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
            norm_layer: nn.Module = nn.InstanceNorm2d):
    return VGGKALN(input_channels, num_classes, groups=groups, degree=degree, dropout=dropout, l1_decay=l1_decay,
                   dropout_linear=dropout_linear, vgg_type=vgg_type, head_type=head_type,
                   expected_feature_shape=expected_feature_shape, width_scale=width_scale,
                   affine=affine, norm_layer=norm_layer)


class VGGKAGN(VGG, PyTorchModelHubMixin):
    def __init__(self, input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0,
                 l1_decay: float = 0.0, dropout_linear: float = 0.25, vgg_type: str = 'VGG11',
                 head_type: str = 'Linear',
                 expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 last_attention: bool = False, sa_inner_projection: int = None):
        conv_fun = partial(kagn_conv3x3, degree=degree,
                           dropout=dropout, l1_decay=l1_decay, groups=groups, affine=affine, norm_layer=norm_layer)
        conv_fun_first = partial(kagn_conv3x3, degree=degree, l1_decay=l1_decay, affine=affine, norm_layer=norm_layer)
        kan_fun = partial(mlp_kagn, degree=degree,
                          dropout=dropout_linear, l1_decay=l1_decay)
        kan_att_fun = partial(self_kagn_conv3x3, degree=degree, inner_projection=sa_inner_projection,
                              dropout=dropout, groups=groups, affine=affine, norm_layer=norm_layer)

        features, head = self.make_layers(cfgs[vgg_type], head_type, conv_fun, conv_fun_first, kan_fun,
                                          expected_feature_shape=expected_feature_shape,
                                          kan_att_fun=kan_att_fun,
                                          num_input_features=input_channels, num_classes=num_classes,
                                          width_scale=width_scale,
                                          head_dropout=dropout_linear, last_attention=last_attention)
        super().__init__(features, head, expected_feature_shape)


def vggkagn(input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0, l1_decay: float = 0.0,
            dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
            expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
            norm_layer: nn.Module = nn.InstanceNorm2d, last_attention: bool = False, sa_inner_projection: int = None):
    return VGGKAGN(input_channels, num_classes, groups=groups, degree=degree, dropout=dropout, l1_decay=l1_decay,
                   dropout_linear=dropout_linear, vgg_type=vgg_type, head_type=head_type,
                   expected_feature_shape=expected_feature_shape, width_scale=width_scale,
                   affine=affine, norm_layer=norm_layer,
                   last_attention=last_attention, sa_inner_projection=sa_inner_projection)


class VGGKAGN_BN(VGG, PyTorchModelHubMixin):
    def __init__(self, input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0,
                 l1_decay: float = 0.0, dropout_linear: float = 0.25, vgg_type: str = 'VGG11',
                 head_type: str = 'Linear',
                 expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 last_attention: bool = False, sa_inner_projection: int = None):
        conv_fun = partial(bottleneck_kagn_conv3x3, degree=degree,
                           dropout=dropout, l1_decay=l1_decay, groups=groups, affine=affine, norm_layer=norm_layer)
        conv_fun_first = partial(bottleneck_kagn_conv3x3, degree=degree, l1_decay=l1_decay,
                                 affine=affine, norm_layer=norm_layer)
        kan_fun = partial(mlp_kagn, degree=degree,
                          dropout=dropout_linear, l1_decay=l1_decay)
        kan_att_fun = partial(self_bottleneck_kagn_conv3x3, degree=degree, inner_projection=sa_inner_projection,
                              dropout=dropout, groups=groups, affine=affine, norm_layer=norm_layer)

        features, head = self.make_layers(cfgs[vgg_type], head_type, conv_fun, conv_fun_first, kan_fun,
                                          expected_feature_shape=expected_feature_shape,
                                          num_input_features=input_channels, num_classes=num_classes,
                                          width_scale=width_scale,
                                          head_dropout=dropout_linear,
                                          kan_att_fun=kan_att_fun,
                                          last_attention=last_attention)
        super().__init__(features, head, expected_feature_shape)


def vggkagn_bn(input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0,
               l1_decay: float = 0.0,
               dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
               expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
               norm_layer: nn.Module = nn.InstanceNorm2d,
               last_attention: bool = False, sa_inner_projection: int = None):
    return VGGKAGN_BN(input_channels, num_classes, groups=groups, degree=degree, dropout=dropout, l1_decay=l1_decay,
                      dropout_linear=dropout_linear, vgg_type=vgg_type, head_type=head_type,
                      expected_feature_shape=expected_feature_shape, width_scale=width_scale,
                      affine=affine, norm_layer=norm_layer, last_attention=last_attention,
                      sa_inner_projection=sa_inner_projection)


class MoE_VGGKAGN_BN(VGG, PyTorchModelHubMixin):
    def __init__(self, input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0,
                 l1_decay: float = 0.0, dropout_linear: float = 0.25, vgg_type: str = 'VGG11',
                 head_type: str = 'Linear',
                 expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
                 norm_layer: nn.Module = nn.InstanceNorm2d,
                 num_experts: int = 8, noisy_gating: bool = True, k: int = 2,
                 last_attention: bool = False, sa_inner_projection: int = None):
        conv_fun = partial(moe_bottleneck_kagn_conv3x3, degree=degree,
                           dropout=dropout, l1_decay=l1_decay, groups=groups, affine=affine, norm_layer=norm_layer,
                           num_experts=num_experts, k=k, noisy_gating=noisy_gating)
        conv_fun_first = partial(bottleneck_kagn_conv3x3, degree=degree, l1_decay=l1_decay,
                                 affine=affine, norm_layer=norm_layer)
        kan_fun = partial(mlp_kagn, degree=degree,
                          dropout=dropout_linear, l1_decay=l1_decay)
        kan_att_fun = partial(self_bottleneck_kagn_conv3x3, degree=degree, inner_projection=sa_inner_projection,
                              dropout=dropout, groups=groups, affine=affine, norm_layer=norm_layer)

        features, head = self.make_layers(cfgs[vgg_type], head_type, conv_fun, conv_fun_first, kan_fun,
                                          expected_feature_shape=expected_feature_shape,
                                          num_input_features=input_channels, num_classes=num_classes,
                                          width_scale=width_scale,
                                          head_dropout=dropout_linear,
                                          kan_att_fun=kan_att_fun,
                                          last_attention=last_attention)
        super().__init__(features, head, expected_feature_shape)


def moe_vggkagn_bn(input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0,
                   l1_decay: float = 0.0,
                   dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
                   expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
                   norm_layer: nn.Module = nn.InstanceNorm2d,
                   last_attention: bool = False, sa_inner_projection: int = None,
                   num_experts: int = 8, noisy_gating: bool = True, k: int = 2,):
    return MoE_VGGKAGN_BN(input_channels, num_classes, groups=groups, degree=degree, dropout=dropout, l1_decay=l1_decay,
                          dropout_linear=dropout_linear, vgg_type=vgg_type, head_type=head_type,
                          expected_feature_shape=expected_feature_shape, width_scale=width_scale,
                          affine=affine, norm_layer=norm_layer, last_attention=last_attention,
                          sa_inner_projection=sa_inner_projection, num_experts=num_experts, noisy_gating=noisy_gating,
                          k=k)


class VGGKACN(VGG, PyTorchModelHubMixin):
    def __init__(self, input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
                 expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
                 norm_layer: nn.Module = nn.InstanceNorm2d):
        conv_fun = partial(kacn_conv3x3, degree=degree,
                           dropout=dropout, l1_decay=l1_decay, groups=groups, affine=affine, norm_layer=norm_layer)
        conv_fun_first = partial(kacn_conv3x3, degree=degree, l1_decay=l1_decay, affine=affine, norm_layer=norm_layer)
        kan_fun = partial(mlp_kacn, degree=degree,
                          dropout=dropout_linear, l1_decay=l1_decay)

        features, head = self.make_layers(cfgs[vgg_type], head_type, conv_fun, conv_fun_first, kan_fun,
                                          expected_feature_shape=expected_feature_shape,
                                          num_input_features=input_channels, num_classes=num_classes,
                                          width_scale=width_scale,
                                          head_dropout=dropout_linear)
        super().__init__(features, head, expected_feature_shape)


def vggkacn(input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0, l1_decay: float = 0.0,
            dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
            expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
            norm_layer: nn.Module = nn.InstanceNorm2d):
    return VGGKACN(input_channels, num_classes, groups=groups, degree=degree, dropout=dropout, l1_decay=l1_decay,
                   dropout_linear=dropout_linear, vgg_type=vgg_type, head_type=head_type,
                   expected_feature_shape=expected_feature_shape, width_scale=width_scale,
                   affine=affine, norm_layer=norm_layer)


class MoEVGGKAGN(VGG, PyTorchModelHubMixin):
    def __init__(self, input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
                 expected_feature_shape: Tuple = (7, 7), width_scale: int = 1, affine: bool = False,
                 num_experts: int = 8, noisy_gating: bool = True, k: int = 2, ):
        conv_fun = partial(moe_kagn_conv3x3, degree=degree,
                           dropout=dropout, l1_decay=l1_decay, groups=groups,
                           num_experts=num_experts, noisy_gating=noisy_gating, k=k, affine=affine)
        conv_fun_first = partial(kagn_conv3x3, degree=degree, l1_decay=l1_decay, affine=affine)
        kan_fun = partial(mlp_kagn, degree=degree,
                          dropout=dropout_linear, l1_decay=l1_decay)
        features, head = self.make_layers(cfgs[vgg_type], head_type, conv_fun, conv_fun_first, kan_fun,
                                          expected_feature_shape=expected_feature_shape,
                                          num_input_features=input_channels, num_classes=num_classes,
                                          width_scale=width_scale,
                                          head_dropout=dropout_linear)
        super().__init__(features, head, expected_feature_shape)


def moe_vggkagn(input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0,
                l1_decay: float = 0.0,
                dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
                expected_feature_shape: Tuple = (7, 7), width_scale: int = 1,
                num_experts: int = 8, noisy_gating: bool = True, k: int = 2, affine: bool = False):
    return MoEVGGKAGN(input_channels, num_classes, groups=groups, degree=degree, dropout=dropout, l1_decay=l1_decay,
                      dropout_linear=dropout_linear, vgg_type=vgg_type, head_type=head_type,
                      expected_feature_shape=expected_feature_shape, width_scale=width_scale, affine=affine,
                      num_experts=num_experts, noisy_gating=noisy_gating, k=k)


class WavVVGKAN(VGG, PyTorchModelHubMixin):
    def __init__(self, input_channels, num_classes, groups: int = 1,
                 wavelet_type: str = 'mexican_hat', wav_version: str = 'fast',
                 dropout: float = 0.0, l1_decay: float = 0.0,
                 dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
                 expected_feature_shape: Tuple = (7, 7), width_scale: int = 1,
                 norm_layer: nn.Module = nn.InstanceNorm2d, affine: bool = False):
        conv_fun = partial(wav_kan_conv3x3, wavelet_type=wavelet_type, wav_version=wav_version,
                           dropout=dropout, l1_decay=l1_decay, groups=groups, norm_layer=norm_layer, affine=affine)
        conv_fun_first = partial(wav_kan_conv3x3, wavelet_type=wavelet_type, wav_version=wav_version,
                                 l1_decay=l1_decay, norm_layer=norm_layer, affine=affine)
        kan_fun = partial(mlp_wav_kan, wavelet_type=wavelet_type,
                          dropout=dropout_linear, l1_decay=l1_decay)

        features, head = self.make_layers(cfgs[vgg_type], head_type, conv_fun, conv_fun_first, kan_fun,
                                          expected_feature_shape=expected_feature_shape,
                                          num_input_features=input_channels, num_classes=num_classes,
                                          width_scale=width_scale,
                                          head_dropout=dropout_linear)
        super().__init__(features, head, expected_feature_shape)


def vgg_wav_kan(input_channels, num_classes, groups: int = 1,
                wavelet_type: str = 'mexican_hat', wav_version: str = 'fast',
                dropout: float = 0.0, l1_decay: float = 0.0,
                dropout_linear: float = 0.25, vgg_type: str = 'VGG11', head_type: str = 'Linear',
                expected_feature_shape: Tuple = (7, 7), width_scale: int = 1,
                norm_layer: nn.Module = nn.InstanceNorm2d, affine: bool = False):
    return WavVVGKAN(input_channels, num_classes, groups=groups,
                     wavelet_type=wavelet_type, wav_version=wav_version,
                     dropout=dropout, l1_decay=l1_decay,
                     dropout_linear=dropout_linear, vgg_type=vgg_type, head_type=head_type,
                     expected_feature_shape=expected_feature_shape, width_scale=width_scale,
                     norm_layer=norm_layer, affine=affine)
