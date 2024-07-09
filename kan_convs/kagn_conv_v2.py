# Based on this: https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py
from functools import lru_cache

import torch
import torch.nn as nn
from torch.nn.functional import conv3d, conv2d, conv1d


class KAGNConvNDLayerV2(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2.,
                 **norm_kwargs):
        super(KAGNConvNDLayerV2, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = nn.SiLU()
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        self.p_dropout = dropout
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = conv_class(input_dim,
                                    output_dim,
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    groups=groups,
                                    bias=False)

        self.layer_norm = norm_class(output_dim, **norm_kwargs)

        # poly_shape = (groups, output_dim // groups, (input_dim // groups) * (degree + 1)) + tuple(
        #     kernel_size for _ in range(ndim))
        self.poly_conv = conv_class(input_dim * (degree + 1),
                                    output_dim,
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    groups=groups,
                                    bias=False)

        # self.poly_weights = nn.Parameter(torch.randn(*poly_shape))
        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

        # Initialize weights using Kaiming uniform distribution for better training start
        # for conv_layer in self.base_conv:
        nn.init.kaiming_uniform_(self.base_conv.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.poly_conv.weight, nonlinearity='linear')

        # nn.init.kaiming_uniform_(self.poly_weights, nonlinearity='linear')
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / ((kernel_size ** ndim) * self.inputdim * (self.degree + 1.0)),
        )

    def beta(self, n, m):
        return (
                       ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
               ) * self.beta_weights[n]

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Gram polynomials
    def gram_poly(self, x, degree):
        p0 = x.new_ones(x.size())

        if degree == 0:
            return p0.unsqueeze(-1)

        p1 = x
        grams_basis = [p0, p1]

        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, i) * p0
            grams_basis.append(p2)
            p0, p1 = p1, p2

        indexes = [i*(degree + 1) + j for i in range(x.shape[1]) for j in range(degree + 1)]

        grams_basis = torch.concatenate(grams_basis, dim=1)
        grams_basis = grams_basis[:, indexes]
        return grams_basis

    def forward_kag(self, x):
        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv(self.base_activation(x))

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x = torch.tanh(x).contiguous()

        if self.dropout is not None:
            x = self.dropout(x)

        grams_basis = self.base_activation(self.gram_poly(x, self.degree))

        # y = self.conv_w_fun(grams_basis, self.poly_weights[group_index],
        #                     stride=self.stride, dilation=self.dilation,
        #                     padding=self.padding, groups=1)
        y = self.poly_conv(grams_basis)

        y = self.base_activation(self.layer_norm(y + basis))

        return y

    def forward(self, x):

        return self.forward_kag(x)


class KAGNConv3DLayerV2(KAGNConvNDLayerV2):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(KAGNConv3DLayerV2, self).__init__(nn.Conv3d, norm_layer, conv3d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=3, dropout=dropout, **norm_kwargs)


class KAGNConv2DLayerV2(KAGNConvNDLayerV2):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(KAGNConv2DLayerV2, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout=dropout, **norm_kwargs)


class KAGNConv1DLayerV2(KAGNConvNDLayerV2):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(KAGNConv1DLayerV2, self).__init__(nn.Conv1d, norm_layer, conv1d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=1, dropout=dropout, **norm_kwargs)
