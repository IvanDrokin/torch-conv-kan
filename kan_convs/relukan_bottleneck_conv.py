# Based on this: https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py

import torch
import torch.nn as nn
from torch.nn.functional import conv3d, conv2d, conv1d


class BottleNeckReLUConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, kernel_size, g: int = 5, k: int = 3,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2., train_ab: bool = True,
                 dim_reduction: float = 4, min_internal: int = 16,
                 **norm_kwargs):
        super(BottleNeckReLUConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.g = g
        self.k = k
        self.r = 4 * g * g / ((k + 1) * (k + 1))
        self.train_ab = train_ab
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
        self.dim_reduction = dim_reduction
        self.min_internal = min_internal

        inner_dim = int(max((input_dim // groups) / dim_reduction,
                            (output_dim // groups) / dim_reduction))
        if inner_dim < min_internal:
            self.inner_dim = min(min_internal, input_dim // groups, output_dim // groups)
        else:
            self.inner_dim = inner_dim

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

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.relukan_conv = nn.ModuleList([conv_class((self.g + self.k) * self.inner_dim,
                                                      self.inner_dim,
                                                      kernel_size,
                                                      stride,
                                                      padding,
                                                      dilation,
                                                      groups=1,
                                                      bias=False) for _ in range(groups)])

        self.inner_proj = nn.ModuleList([conv_class(input_dim // groups,
                                                    self.inner_dim,
                                                    1,
                                                    1,
                                                    0,
                                                    1,
                                                    groups=1,
                                                    bias=False) for _ in range(groups)])
        self.out_proj = nn.ModuleList([conv_class(self.inner_dim,
                                                  output_dim // groups,
                                                  1,
                                                  1,
                                                  0,
                                                  1,
                                                  groups=1,
                                                  bias=False) for _ in range(groups)])

        phase_low = torch.arange(-k, g) / g
        phase_high = phase_low + (k + 1) / g

        phase_dims = (1, self.inner_dim, k + g) + (1,) * ndim

        self.phase_low = nn.Parameter((phase_low[None, :].expand(self.inner_dim, -1)).view(*phase_dims),
                                      requires_grad=train_ab)

        self.phase_high = nn.Parameter(
            (phase_high[None, :].expand(self.inner_dim, -1)).view(*phase_dims),
            requires_grad=train_ab)

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.relukan_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.inner_proj:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.out_proj:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_relukan(self, x, group_index):
        if self.dropout:
            x = self.dropout(x)
        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))
        x = self.inner_proj[group_index](x)
        x = x.unsqueeze(dim=2)
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        x = x1 * x2 * self.r
        x = x * x
        x = torch.flatten(x, 1, 2)

        y = self.relukan_conv[group_index](x)
        y = self.out_proj[group_index](y)

        y = self.base_activation(self.layer_norm[group_index](y + basis))

        return y

    def forward(self, x):

        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_relukan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class BottleNeckReLUKANConv3DLayer(BottleNeckReLUConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, g=5, k=3, train_ab=True, groups=1, padding=0, stride=1,
                 dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(BottleNeckReLUKANConv3DLayer, self).__init__(nn.Conv3d, norm_layer, conv3d,
                                                           input_dim, output_dim,
                                                           kernel_size, g=g, k=k, train_ab=train_ab,
                                                           groups=groups, padding=padding, stride=stride,
                                                           dilation=dilation,
                                                           ndim=3, dropout=dropout, **norm_kwargs)


class BottleNeckReLUKANConv2DLayer(BottleNeckReLUConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, g=5, k=3, train_ab=True, groups=1, padding=0, stride=1,
                 dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(BottleNeckReLUKANConv2DLayer, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                                           input_dim, output_dim,
                                                           kernel_size, g=g, k=k, train_ab=train_ab,
                                                           groups=groups, padding=padding, stride=stride,
                                                           dilation=dilation,
                                                           ndim=2, dropout=dropout, **norm_kwargs)


class BottleNeckReLUKANConv1DLayer(BottleNeckReLUConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, g=5, k=3, train_ab=True, groups=1, padding=0, stride=1,
                 dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(BottleNeckReLUKANConv1DLayer, self).__init__(nn.Conv1d, norm_layer, conv1d,
                                                           input_dim, output_dim,
                                                           kernel_size, g=g, k=k, train_ab=train_ab,
                                                           groups=groups, padding=padding, stride=stride,
                                                           dilation=dilation,
                                                           ndim=1, dropout=dropout, **norm_kwargs)
