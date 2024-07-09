from functools import lru_cache
from typing import List

import torch
import torch.nn as nn

from kan_convs.kagn_conv import KAGNConvNDLayer, KAGNConv1DLayer, KAGNConv2DLayer, KAGNConv3DLayer
from models.vggkan import VGG


class PEFTKAGNConvNDLayer(KAGNConvNDLayer):
    def __init__(self, pretrained_weights,
                 conv_class, norm_class, conv_w_fun,
                 input_dim, output_dim, degree, kernel_size,
                 trainable_degrees: List = None, extra_degrees=1, finetune_base: bool = False,
                 groups: int = 1, padding: int = 0, stride: int = 1,
                 dilation: int = 1, dropout: float = 0.0, ndim: int = 2,
                 **norm_kwargs
                 ):
        super(PEFTKAGNConvNDLayer, self).__init__(conv_class, norm_class, conv_w_fun, input_dim,
                                                  output_dim, degree, kernel_size,
                                                  groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                  dropout=dropout, ndim=ndim, **norm_kwargs)

        self.conv_class = conv_class
        self.norm_class = norm_class
        self.finetune_base = finetune_base

        self.extra_degrees = extra_degrees
        self.new_degree = self.degree + extra_degrees
        if trainable_degrees is not None:
            assert len(trainable_degrees) <= degree + 1
            assert min(trainable_degrees) >= 0
            assert max(trainable_degrees) <= degree
            self.trainable_degrees = trainable_degrees
            self.delta_mapper = {d: i for i, d in enumerate(trainable_degrees)}
        else:
            self.trainable_degrees = []

        # load weights
        self.load_state_dict(pretrained_weights)

        # New degrees weights:
        if extra_degrees > 0:
            poly_shape = (groups, output_dim // groups, (input_dim // groups) * extra_degrees) + tuple(
                kernel_size for _ in range(ndim))

            self.poly_weights_extra = nn.Parameter(torch.zeros(*poly_shape))
            self.beta_weights_extra = nn.Parameter(torch.zeros(extra_degrees, dtype=torch.float32))

        # Trainable degrees
        if len(self.trainable_degrees) > 0:
            poly_shape = (groups, output_dim // groups, (input_dim // groups) * len(self.trainable_degrees)) + tuple(
                kernel_size for _ in range(ndim))

            self.poly_weights_delta = nn.Parameter(torch.zeros(*poly_shape))
            self.beta_weights_delta = nn.Parameter(torch.zeros(len(self.trainable_degrees), dtype=torch.float32))

            list_of_delta_index = []
            for i in self.trainable_degrees:
                list_of_delta_index += list(range((input_dim // groups) * i, (input_dim // groups) * (i + 1)))
            self.delta_indices = nn.Parameter(torch.tensor(list_of_delta_index, dtype=torch.int32), requires_grad=False)
        # Set base parameters to non-trainable
        if not self.finetune_base:
            for layer in self.base_conv:
                for p in layer.parameters():
                    p.requires_grad = False
        self.poly_weights.requires_grad = False
        self.beta_weights.requires_grad = False

    @staticmethod
    def wrap_layer(layer, trainable_degrees: List = None, extra_degrees: int = 1, finetune_base: bool = False):
        assert isinstance(layer, (KAGNConv1DLayer, KAGNConv2DLayer, KAGNConv3DLayer))
        return PEFTKAGNConvNDLayer(
            layer.state_dict(),
            layer.base_conv[0].__class__, layer.layer_norm[0].__class__, layer.conv_w_fun,
            layer.inputdim, layer.outdim, layer.degree, layer.kernel_size,
            trainable_degrees=trainable_degrees, extra_degrees=extra_degrees, finetune_base=finetune_base,
            groups=layer.groups, padding=layer.padding, stride=layer.stride,
            dilation=layer.dilation, dropout=layer.p_dropout, ndim=layer.ndim,
            **layer.norm_kwargs
        )

    def fuse_layer(self):

        if self.ndim == 1:
            output_class = KAGNConv1DLayer
        elif self.ndim == 2:
            output_class = KAGNConv2DLayer
        elif self.ndim == 3:
            output_class = KAGNConv3DLayer
        else:
            output_class = None

        fused_layer = output_class(
            self.inputdim, self.outdim, self.kernel_size, degree=self.new_degree,
            norm_layer=self.layer_norm[0].__class__,
            groups=self.groups, padding=self.padding, stride=self.stride,
            dilation=self.dilation, dropout=self.p_dropout,
            **self.norm_kwargs
        )

        fused_layer.base_conv.load_state_dict(self.base_conv.state_dict())
        fused_layer.layer_norm.load_state_dict(self.layer_norm.state_dict())

        poly_shape = (self.groups, self.outdim // self.groups, (self.inputdim // self.groups) * (self.new_degree + 1))\
                     + tuple(self.kernel_size for _ in range(self.ndim))
        with torch.no_grad():
            fused_poly_weights = nn.Parameter(torch.zeros(*poly_shape))
            fused_beta_weights = nn.Parameter(torch.zeros(self.new_degree + 1, dtype=torch.float32))

            # Add initial weight
            fused_poly_weights[:, :, :(self.inputdim // self.groups) * (self.degree + 1)] += self.poly_weights
            fused_beta_weights[:self.degree + 1] += self.beta_weights

            # Add trainable part
            for _index, _degree in enumerate(self.trainable_degrees):
                fused_poly_weights[:, :, (self.inputdim // self.groups) * _degree:(self.inputdim // self.groups) * (
                            _degree + 1)] += self.poly_weights_delta[:, :,
                                             (self.inputdim // self.groups) * _index:(self.inputdim // self.groups) * (
                                                         _index + 1)]
                fused_beta_weights[_degree] += self.beta_weights_delta[_index]

            # Add new degrees

            fused_poly_weights[:, :, (self.inputdim // self.groups) * (self.degree + 1):] += self.poly_weights_extra
            fused_beta_weights[(self.degree + 1):] += self.beta_weights_extra

            fused_layer.poly_weights = fused_poly_weights
            fused_layer.beta_weights = fused_beta_weights

        return fused_layer

    def beta(self, n, m):
        _x = ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))

        if n + 1 in self.trainable_degrees:
            _x = _x * (self.beta_weights[n] + self.beta_weights_delta[self.delta_mapper[n + 1]])
        elif n >= self.degree:
            _x = _x * self.beta_weights_extra[n - self.degree - 1]
        else:
            _x = _x * self.beta_weights[n]
        return _x

    def forward_kag(self, x, group_index):
        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x = torch.tanh(x).contiguous()

        if self.dropout is not None:
            x = self.dropout(x)

        grams_basis = self.base_activation(self.gram_poly(x, self.new_degree))

        # Original
        y = self.conv_w_fun(grams_basis[:, :(self.degree + 1) * (self.inputdim // self.groups)],
                            self.poly_weights[group_index],
                            stride=self.stride, dilation=self.dilation,
                            padding=self.padding, groups=1)
        # Delta
        if len(self.trainable_degrees) > 0:
            y = y + self.conv_w_fun(torch.index_select(grams_basis, 1, self.delta_indices),
                                    self.poly_weights_delta[group_index],
                                    stride=self.stride, dilation=self.dilation,
                                    padding=self.padding, groups=1)

        # New
        if self.extra_degrees > 0:
            y = y + self.conv_w_fun(grams_basis[:, (self.degree + 1)*(self.inputdim // self.groups):],
                                    self.poly_weights_extra[group_index],
                                    stride=self.stride, dilation=self.dilation,
                                    padding=self.padding, groups=1)

        y = self.base_activation(self.layer_norm[group_index](y + basis))

        return y


class PEFTVGGKAGN(VGG):
    def __init__(self, pretrained_vgg: VGG,
                 trainable_degrees: List = None, extra_degrees=1, finetune_base: bool = False,):

        features = nn.ModuleList()
        for layer in pretrained_vgg.features:
            if isinstance(layer, (KAGNConv1DLayer, KAGNConv2DLayer, KAGNConv3DLayer)):
                features.append(PEFTKAGNConvNDLayer.wrap_layer(layer,
                                                               trainable_degrees=trainable_degrees,
                                                               extra_degrees=extra_degrees,
                                                               finetune_base=finetune_base))
            else:
                features.append(layer)

        super().__init__(features, pretrained_vgg.classifier, pretrained_vgg.expected_feature_shape)

    def merge(self):
        features = nn.ModuleList()
        for layer in self.features:
            if isinstance(layer, PEFTKAGNConvNDLayer):
                features.append(layer.fuse_layer())
            else:
                features.append(layer)
        return VGG(features, self.classifier, self.expected_feature_shape)
