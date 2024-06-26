from typing import Union

import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .densekanet import densekalnet161, densekalnet169, densekalnet201, densekagnet161, densekagnet169, densekagnet201
from .densekanet import densekanet121, densekagnet121, densekacnet121, fast_densekanet121, densekalnet121
from .densekanet import tiny_densekagnet_bn, tiny_densekagnet_moebn
from .reskanet import reskagnet18, reskagnet50, reskagnet101, reskagnet152
from .u2kanet import u2kagnet, u2kacnet, u2kalnet, u2kanet, fast_u2kanet
from .u2kanet import u2kagnet_small, u2kacnet_small, u2kalnet_small, u2kanet_small, fast_u2kanet_small
from .ukanet import ukanet_18, ukalnet_18, fast_ukanet_18, ukacnet_18, ukagnet_18
from .vggkan import fast_vggkan, vggkan, vggkaln, vggkacn, vggkagn, vgg_wav_kan, vggkagn_bn, moe_vggkagn_bn


def get_activation_from_string(activation_string):
    if activation_string == 'gelu':
        return nn.GELU
    if activation_string == 'silu':
        return nn.SiLU
    if activation_string == 'relu':
        return nn.ReLU
    if activation_string == 'prelu':
        return nn.PReLU


def get_norm_layer_from_string(norm_layer_string) -> Union[(nn.InstanceNorm2d, nn.BatchNorm2d)]:
    if norm_layer_string == 'instance2d':
        return nn.InstanceNorm2d
    if norm_layer_string == 'batchnorm2d':
        return nn.BatchNorm2d
    return nn.InstanceNorm2d


class AutoKAN(PyTorchModelHubMixin):
    def __init__(self, **kwargs):

        super(AutoKAN).__init__()

        self.model = None
        self.config = kwargs
        model_name = kwargs.pop('model_name')
        input_channels = kwargs.pop('input_channels')
        num_classes = kwargs.pop('num_classes')

        # UNET18 ===========
        if model_name == "ukanet18":
            self.model = ukanet_18(input_channels,
                                   num_classes,
                                   width_scale=kwargs.pop('width_scale', 1),
                                   groups=kwargs.pop("groups", 1),
                                   spline_order=kwargs.pop("spline_order", 1),
                                   grid_size=kwargs.pop("grid_size", 1),
                                   base_activation=get_activation_from_string(kwargs.pop("base_activation", 'gelu')),
                                   grid_range=kwargs.pop("grid_range", [-1, 1]),
                                   affine=kwargs.pop('affine', True),
                                   norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))

        elif model_name in ["ukalnet18", "ukagnet18", "ukacnet18"]:
            if model_name == 'ukalnet18':
                func = ukalnet_18
            elif model_name == 'ukagnet18':
                func = ukagnet_18
            else:
                func = ukacnet_18
            self.model = func(input_channels,
                              num_classes,
                              width_scale=kwargs.pop('width_scale', 1),
                              groups=kwargs.pop("groups", 1),
                              affine=kwargs.pop('affine', True),
                              degree=kwargs.pop('degree', 3),
                              norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        elif model_name == "fast_ukanet_18":
            self.model = fast_ukanet_18(input_channels,
                                        num_classes,
                                        width_scale=kwargs.pop('width_scale', 1),
                                        groups=kwargs.pop("groups", 1),
                                        grid_size=kwargs.pop("grid_size", 1),
                                        grid_range=kwargs.pop("grid_range", [-1, 1]),
                                        base_activation=get_activation_from_string(
                                            kwargs.pop("base_activation", 'gelu')),
                                        affine=kwargs.pop('affine', True),
                                        norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        # U2NET ===========
        elif model_name == "u2kanet":
            self.model = u2kanet(input_channels,
                                 num_classes,
                                 width_scale=kwargs.pop('width_scale', 1),
                                 groups=kwargs.pop("groups", 1),
                                 spline_order=kwargs.pop("spline_order", 1),
                                 grid_size=kwargs.pop("grid_size", 1),
                                 base_activation=get_activation_from_string(kwargs.pop("base_activation", 'gelu')),
                                 grid_range=kwargs.pop("grid_range", [-1, 1]),
                                 affine=kwargs.pop('affine', True),
                                 norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        elif model_name in ["u2kagnet", "u2kalnet", "u2kacnet"]:
            if model_name == 'u2kalnet':
                func = u2kalnet
            elif model_name == 'u2kagnet':
                func = u2kagnet
            else:
                func = u2kacnet
            self.model = func(input_channels,
                              num_classes,
                              width_scale=kwargs.pop('width_scale', 1),
                              groups=kwargs.pop("groups", 1),
                              affine=kwargs.pop('affine', True),
                              degree=kwargs.pop('degree', 3),
                              norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        elif model_name == "fast_u2kanet":
            self.model = fast_u2kanet(input_channels,
                                      num_classes,
                                      width_scale=kwargs.pop('width_scale', 1),
                                      groups=kwargs.pop("groups", 1),
                                      grid_size=kwargs.pop("grid_size", 1),
                                      grid_range=kwargs.pop("grid_range", [-1, 1]),
                                      base_activation=get_activation_from_string(kwargs.pop("base_activation", 'gelu')),
                                      affine=kwargs.pop('affine', True),
                                      norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        # U2NET Small ===========
        elif model_name == "u2kanet_small":
            self.model = u2kanet_small(input_channels,
                                       num_classes,
                                       width_scale=kwargs.pop('width_scale', 1),
                                       groups=kwargs.pop("groups", 1),
                                       spline_order=kwargs.pop("spline_order", 1),
                                       grid_size=kwargs.pop("grid_size", 1),
                                       base_activation=get_activation_from_string(
                                           kwargs.pop("base_activation", 'gelu')),
                                       grid_range=kwargs.pop("grid_range", [-1, 1]),
                                       affine=kwargs.pop('affine', True),
                                       norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        elif model_name in ["u2kagnet_small", "u2kalnet_small", "u2kacnet_small"]:
            if model_name == 'u2kalnet_small':
                func = u2kalnet_small
            elif model_name == 'u2kagnet_small':
                func = u2kagnet_small
            else:
                func = u2kacnet_small
            self.model = func(input_channels,
                              num_classes,
                              width_scale=kwargs.pop('width_scale', 1),
                              groups=kwargs.pop("groups", 1),
                              affine=kwargs.pop('affine', True),
                              degree=kwargs.pop('degree', 3),
                              norm_layer=kwargs.pop('norm_layer', 'instance2d'))
        elif model_name == "fast_u2kanet_small":
            self.model = fast_u2kanet_small(input_channels,
                                            num_classes,
                                            width_scale=kwargs.pop('width_scale', 1),
                                            groups=kwargs.pop("groups", 1),
                                            grid_size=kwargs.pop("grid_size", 1),
                                            grid_range=kwargs.pop("grid_range", [-1, 1]),
                                            base_activation=get_activation_from_string(
                                                kwargs.pop("base_activation", 'gelu')),
                                            affine=kwargs.pop('affine', True),
                                            norm_layer=get_norm_layer_from_string(
                                                kwargs.pop('norm_layer', 'instance2d')))
        # DenseNets ===========

        elif model_name == "densekanet121":
            self.model = densekanet121(input_channels,
                                       num_classes,
                                       groups=kwargs.pop("groups", 1),
                                       spline_order=kwargs.pop("spline_order", 1),
                                       grid_size=kwargs.pop("grid_size", 1),
                                       base_activation=get_activation_from_string(kwargs.pop("base_activation",
                                                                                             'gelu')),
                                       dropout=kwargs.pop("dropout", 0.0),
                                       l1_decay=kwargs.pop('l1_decay', 0.0),
                                       use_first_maxpool=kwargs.pop('use_first_maxpool', True),
                                       growth_rate=kwargs.pop('growth_rate', 32),
                                       num_init_features=kwargs.pop('num_init_features', 64),
                                       grid_range=kwargs.pop("grid_range", [-1, 1]),
                                       affine=kwargs.pop('affine', True),
                                       norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        elif model_name == "fast_densekanet121":
            self.model = fast_densekanet121(input_channels,
                                            num_classes,
                                            groups=kwargs.pop("groups", 1),
                                            grid_size=kwargs.pop("grid_size", 1),
                                            base_activation=get_activation_from_string(kwargs.pop("base_activation",
                                                                                                  'gelu')),
                                            dropout=kwargs.pop("dropout", 0.0),
                                            l1_decay=kwargs.pop('l1_decay', 0.0),
                                            use_first_maxpool=kwargs.pop('use_first_maxpool', True),
                                            growth_rate=kwargs.pop('growth_rate', 32),
                                            num_init_features=kwargs.pop('num_init_features', 64),
                                            grid_range=kwargs.pop("grid_range", [-1, 1]),
                                            affine=kwargs.pop('affine', True),
                                            norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer',
                                                                                             'instance2d')))
        elif model_name in ["densekagnet121", "densekalnet121", "densekacnet121", "densekalnet161", "densekalnet169",
                            "densekalnet201", "densekagnet161", "densekagnet169", "densekagnet201"]:
            if model_name == 'densekagnet121':
                func = densekagnet121
            elif model_name == 'densekalnet121':
                func = densekalnet121
            elif model_name == 'densekalnet161':
                func = densekalnet161
            elif model_name == '"densekalnet169"':
                func = densekalnet169
            elif model_name == '"densekalnet201"':
                func = densekalnet201
            elif model_name == '"densekagnet161"':
                func = densekagnet161
            elif model_name == '"densekagnet169"':
                func = densekagnet169
            elif model_name == '"densekagnet201"':
                func = densekagnet201
            else:
                func = densekacnet121
            self.model = func(input_channels,
                              num_classes,
                              dropout=kwargs.pop("dropout", 0.0),
                              l1_decay=kwargs.pop('l1_decay', 0.0),
                              use_first_maxpool=kwargs.pop('use_first_maxpool', True),
                              growth_rate=kwargs.pop('growth_rate', 32),
                              num_init_features=kwargs.pop('num_init_features', 64),
                              groups=kwargs.pop("groups", 1),
                              affine=kwargs.pop('affine', True),
                              degree=kwargs.pop('degree', 3),
                              norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        # VGG ===========
        elif model_name == "vggkan":
            self.model = vggkan(input_channels,
                                num_classes,
                                groups=kwargs.pop("groups", 1),
                                spline_order=kwargs.pop("spline_order", 1),
                                grid_size=kwargs.pop("grid_size", 1),
                                base_activation=get_activation_from_string(kwargs.pop("base_activation",
                                                                                      'gelu')),
                                grid_range=kwargs.pop("grid_range", [-1, 1]),
                                dropout=kwargs.pop("dropout", 0.0),
                                l1_decay=kwargs.pop('l1_decay', 0.0),
                                dropout_linear=kwargs.pop("dropout_linear", 0.0),
                                vgg_type=kwargs.pop("vgg_type", 'VGG11'),
                                head_type=kwargs.pop("head_type", 'Linear'),
                                expected_feature_shape=kwargs.pop('expected_feature_shape', (7, 7)),
                                width_scale=kwargs.pop('width_scale', 1),
                                affine=kwargs.pop('affine', True),
                                norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        elif model_name == "fast_vggkan":
            self.model = fast_vggkan(input_channels,
                                     num_classes,
                                     groups=kwargs.pop("groups", 1),
                                     grid_size=kwargs.pop("grid_size", 1),
                                     base_activation=get_activation_from_string(kwargs.pop("base_activation",
                                                                                           'gelu')),
                                     grid_range=kwargs.pop("grid_range", [-1, 1]),
                                     dropout=kwargs.pop("dropout", 0.0),
                                     l1_decay=kwargs.pop('l1_decay', 0.0),
                                     dropout_linear=kwargs.pop("dropout_linear", 0.0),
                                     vgg_type=kwargs.pop("vgg_type", 'VGG11'),
                                     head_type=kwargs.pop("head_type", 'Linear'),
                                     expected_feature_shape=kwargs.pop('expected_feature_shape', (7, 7)),
                                     width_scale=kwargs.pop('width_scale', 1),
                                     affine=kwargs.pop('affine', True),
                                     norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        elif model_name == "vgg_wav_kan":
            self.model = vgg_wav_kan(input_channels,
                                     num_classes,
                                     groups=kwargs.pop("groups", 1),
                                     wavelet_type=kwargs.pop('wavelet_type', 'mexican_hat'),
                                     wav_version=kwargs.pop('wav_version', 'fast'),
                                     dropout=kwargs.pop("dropout", 0.0),
                                     l1_decay=kwargs.pop('l1_decay', 0.0),
                                     dropout_linear=kwargs.pop("dropout_linear", 0.0),
                                     vgg_type=kwargs.pop("vgg_type", 'VGG11'),
                                     head_type=kwargs.pop("head_type", 'Linear'),
                                     expected_feature_shape=kwargs.pop('expected_feature_shape', (7, 7)),
                                     width_scale=kwargs.pop('width_scale', 1),
                                     affine=kwargs.pop('affine', True),
                                     norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        elif model_name in ['vggkaln', 'vggkacn', 'vggkagn']:
            if model_name == 'vggkaln':
                func = vggkaln
            elif model_name == 'vggkacn':
                func = vggkacn
            else:
                func = vggkagn
            self.model = func(input_channels,
                              num_classes,
                              groups=kwargs.pop("groups", 1),
                              degree=kwargs.pop("degree", 3),
                              dropout=kwargs.pop("dropout", 0.0),
                              l1_decay=kwargs.pop('l1_decay', 0.0),
                              dropout_linear=kwargs.pop("dropout_linear", 0.0),
                              vgg_type=kwargs.pop("vgg_type", 'VGG11'),
                              head_type=kwargs.pop("head_type", 'Linear'),
                              expected_feature_shape=kwargs.pop('expected_feature_shape', (7, 7)),
                              width_scale=kwargs.pop('width_scale', 1),
                              affine=kwargs.pop('affine', True),
                              norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))
        elif model_name in ["reskagnet18", "reskagnet50", "reskagnet101", "reskagnet152"]:
            if model_name == 'reskagnet18':
                func = reskagnet18
            elif model_name == 'reskagnet50':
                func = reskagnet50
            elif model_name == 'reskagnet101':
                func = reskagnet101
            else:
                func = reskagnet152
            self.model = func(input_channels,
                              num_classes,
                              groups=kwargs.pop("groups", 1),
                              degree=kwargs.pop("degree", 3),
                              dropout=kwargs.pop("dropout", 0.0),
                              l1_decay=kwargs.pop('l1_decay', 0.0),
                              dropout_linear=kwargs.pop("dropout_linear", 0.0),
                              hidden_layer_dim=kwargs.pop('hidden_layer_dim', None),
                              width_scale=kwargs.pop('width_scale', 1),
                              affine=kwargs.pop('affine', True),
                              norm_layer=get_norm_layer_from_string(kwargs.pop('norm_layer', 'instance2d')))

    def __call__(self, *args, **kwargs):
        return self.model.__call__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def modules(self, *args, **kwargs):
        return self.model.modules(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.model.eval(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)


class TinyAutoKAGN(PyTorchModelHubMixin):
    """
    Easy interface for a set of BottleNeckKAGNConv Nets for TinyImageNet experiments
    """

    def __init__(self, input_channels, num_classes, model_type,
                 is_moe: bool = False, groups: int = 1, degree: int = 3, dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 dropout_linear: float = 0.25, width_scale: int = 1,
                 norm_layer: nn.Module = nn.BatchNorm2d, affine: bool = True,
                 num_experts: int = 8, noisy_gating: bool = True, k: int = 2, **kwargs):

        super(TinyAutoKAGN).__init__()

        self.model = None
        self.config = kwargs

        if model_type == 'VGG':
            vgg_type = kwargs.pop('vgg_type', 'VGG11')
            head_type = kwargs.pop('head_type', 'Linear')
            last_attention = kwargs.pop('last_attention', False)
            sa_inner_projection = kwargs.pop('sa_inner_projection', None)
            expected_feature_shape = kwargs.pop('expected_feature_shape', (1, 1))
            if is_moe:
                self.model = moe_vggkagn_bn(input_channels, num_classes, groups=groups, degree=degree, dropout=dropout,
                                            l1_decay=l1_decay, norm_layer=norm_layer,
                                            dropout_linear=dropout_linear, vgg_type=vgg_type, head_type=head_type,
                                            expected_feature_shape=expected_feature_shape, width_scale=width_scale,
                                            num_experts=num_experts, noisy_gating=noisy_gating, k=k, affine=affine,
                                            last_attention=last_attention, sa_inner_projection=sa_inner_projection)
            else:
                self.model = vggkagn_bn(input_channels, num_classes, groups=groups, degree=degree, dropout=dropout,
                                        l1_decay=l1_decay, dropout_linear=dropout_linear, vgg_type=vgg_type,
                                        head_type=head_type,
                                        expected_feature_shape=expected_feature_shape, width_scale=width_scale,
                                        affine=affine, last_attention=last_attention,
                                        sa_inner_projection=sa_inner_projection)

        if model_type == 'ResNet':
            pass

        if model_type == 'DenseNet':
            growth_rate = kwargs.pop('growth_rate', 32)
            num_init_features = kwargs.pop('num_init_features', 64)
            if is_moe:
                self.model = tiny_densekagnet_moebn(input_channels, num_classes,
                                                    groups=groups, degree=degree, dropout=dropout,
                                                    dropout_linear=dropout_linear, l1_decay=l1_decay,
                                                    growth_rate=growth_rate, num_init_features=num_init_features,
                                                    affine=affine,
                                                    norm_layer=norm_layer,
                                                    num_experts=num_experts, noisy_gating=noisy_gating, k=k)
            else:
                self.model = tiny_densekagnet_bn(input_channels, num_classes,
                                                 groups=groups, degree=degree, dropout=dropout,
                                                 dropout_linear=dropout_linear, l1_decay=l1_decay,
                                                 growth_rate=growth_rate, num_init_features=num_init_features,
                                                 affine=affine,
                                                 norm_layer=norm_layer)

    def __call__(self, *args, **kwargs):
        return self.model.__call__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def modules(self, *args, **kwargs):
        return self.model.modules(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.model.eval(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)

    def children(self, *args, **kwargs):
        return self.model.children(*args, **kwargs)

    def register_forward_pre_hook(self, *args, **kwargs):
        return self.model.register_forward_pre_hook(*args, **kwargs)

    def register_forward_hook(self, *args, **kwargs):
        return self.model.register_forward_hook(*args, **kwargs)
