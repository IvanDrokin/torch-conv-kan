# Based on this https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg16
from functools import partial
from math import prod
from typing import cast, Dict, List, Optional, Union, Tuple, Callable

import torch
import torch.nn as nn

from kans import mlp_kan, mlp_fastkan, mlp_kacn, mlp_kagn, mlp_kaln
from .reskanet import kan_conv3x3, kaln_conv3x3, fast_kan_conv3x3, kacn_conv3x3, kagn_conv3x3


class VGG(nn.Module):
    def __init__(
            self, features: nn.Module, classifier: nn.Module, expected_feature_shape: Tuple = (7, 7)
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(expected_feature_shape)
        self.classifier = classifier

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


cfgs: Dict[str, List[Union[str, int]]] = {
    "VGG11": [16, "M", 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M"],
    "VGG13": [16, 16, "M", 32, 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M"],
    "VGG16": [16, 16, "M", 32, 32, "M", 64, 64, 64, "M", 128, 128, 128, "M", 128, 128, 128, "M"],
    "VGG19": [16, 16, "M", 32, 32, "M", 64, 64, 64, 64, "M", 128, 128, 128, 128, "M", 128, 128, 128, 128, "M"],
}


def make_layers(cfg: List[Union[str, int]],
                conv_fun,
                conv_fun_first,
                kan_fun,
                expected_feature_shape: Tuple = (7, 7),
                num_input_features: int = 3,
                num_classes: int = 1000,
                width_scale: int = 1,
                ):
    layers: List[nn.Module] = []
    in_channels = num_input_features
    for l_index, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            if l_index == 0:
                conv2d = conv_fun_first(in_channels, v * width_scale)
            else:
                conv2d = conv_fun(in_channels, v * width_scale)
            layers.append(conv2d)
            in_channels = v * width_scale

    classification = kan_fun([128 * width_scale * prod(expected_feature_shape), num_classes])

    return nn.Sequential(*layers), classification


def vggkan(input_channels, num_classes, groups: int = 1, spline_order: int = 3, grid_size: int = 5,
           base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
           grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
           dropout_linear: float = 0.25, vgg_type: str = 'VGG11', expected_feature_shape: Tuple = (7, 7),
           width_scale: int = 1):
    conv_fun = partial(kan_conv3x3, spline_order=spline_order, grid_size=grid_size,
                       base_activation=base_activation, grid_range=grid_range,
                       dropout=dropout, l1_decay=l1_decay, groups=groups)
    conv_fun_first = partial(kan_conv3x3, spline_order=spline_order, grid_size=grid_size,
                             base_activation=base_activation, grid_range=grid_range, l1_decay=l1_decay)
    kan_fun = partial(mlp_kan, spline_order=spline_order, grid_size=grid_size,
                      base_activation=base_activation, grid_range=grid_range,
                      dropout=dropout_linear, l1_decay=l1_decay)

    features, head = make_layers(cfgs[vgg_type], conv_fun, conv_fun_first, kan_fun,
                                 expected_feature_shape=expected_feature_shape,
                                 num_input_features=input_channels, num_classes=num_classes, width_scale=width_scale)
    return VGG(features, head, expected_feature_shape)


def fast_vggkan(input_channels, num_classes, groups: int = 1, grid_size: int = 5,
                base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
                grid_range: List = [-1, 1], dropout: float = 0.0, l1_decay: float = 0.0,
                dropout_linear: float = 0.25, vgg_type: str = 'VGG11', expected_feature_shape: Tuple = (7, 7),
                width_scale: int = 1):
    conv_fun = partial(fast_kan_conv3x3, grid_size=grid_size,
                       base_activation=base_activation, grid_range=grid_range,
                       dropout=dropout, l1_decay=l1_decay, groups=groups)
    conv_fun_first = partial(fast_kan_conv3x3, grid_size=grid_size,
                             base_activation=base_activation, grid_range=grid_range, l1_decay=l1_decay)
    kan_fun = partial(mlp_fastkan, grid_size=grid_size,
                      base_activation=base_activation, grid_range=grid_range,
                      dropout=dropout_linear, l1_decay=l1_decay)

    features, head = make_layers(cfgs[vgg_type], conv_fun, conv_fun_first, kan_fun,
                                 expected_feature_shape=expected_feature_shape,
                                 num_input_features=input_channels, num_classes=num_classes, width_scale=width_scale)
    return VGG(features, head, expected_feature_shape)


def vggkaln(input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0, l1_decay: float = 0.0,
            dropout_linear: float = 0.25, vgg_type: str = 'VGG11', expected_feature_shape: Tuple = (7, 7),
            width_scale: int = 1):
    conv_fun = partial(kaln_conv3x3, degree=degree, dropout=dropout, l1_decay=l1_decay, groups=groups)
    conv_fun_first = partial(kaln_conv3x3, degree=degree, l1_decay=l1_decay)
    kan_fun = partial(mlp_kaln, degree=degree, dropout=dropout_linear, l1_decay=l1_decay)

    features, head = make_layers(cfgs[vgg_type], conv_fun, conv_fun_first, kan_fun,
                                 expected_feature_shape=expected_feature_shape,
                                 num_input_features=input_channels, num_classes=num_classes, width_scale=width_scale)
    return VGG(features, head, expected_feature_shape)


def vggkagn(input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0, l1_decay: float = 0.0,
            dropout_linear: float = 0.25, vgg_type: str = 'VGG11', expected_feature_shape: Tuple = (7, 7),
            width_scale: int = 1):
    conv_fun = partial(kagn_conv3x3, degree=degree,
                       dropout=dropout, l1_decay=l1_decay, groups=groups)
    conv_fun_first = partial(kagn_conv3x3, degree=degree, l1_decay=l1_decay)
    kan_fun = partial(mlp_kagn, degree=degree,
                      dropout=dropout_linear, l1_decay=l1_decay)

    features, head = make_layers(cfgs[vgg_type], conv_fun, conv_fun_first, kan_fun,
                                 expected_feature_shape=expected_feature_shape,
                                 num_input_features=input_channels, num_classes=num_classes, width_scale=width_scale)
    return VGG(features, head, expected_feature_shape)


def vggkacn(input_channels, num_classes, groups: int = 1, degree: int = 3, dropout: float = 0.0, l1_decay: float = 0.0,
            dropout_linear: float = 0.25, vgg_type: str = 'VGG11', expected_feature_shape: Tuple = (7, 7),
            width_scale: int = 1):
    conv_fun = partial(kacn_conv3x3, degree=degree,
                       dropout=dropout, l1_decay=l1_decay, groups=groups)
    conv_fun_first = partial(kacn_conv3x3, degree=degree, l1_decay=l1_decay)
    kan_fun = partial(mlp_kacn, degree=degree,
                      dropout=dropout_linear, l1_decay=l1_decay)

    features, head = make_layers(cfgs[vgg_type], conv_fun, conv_fun_first, kan_fun,
                                 expected_feature_shape=expected_feature_shape,
                                 num_input_features=input_channels, num_classes=num_classes, width_scale=width_scale)
    return VGG(features, head, expected_feature_shape)
