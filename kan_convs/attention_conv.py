from math import prod

import torch
import torch.nn as nn

from .fast_kan_conv import FastKANConv1DLayer, FastKANConv2DLayer, FastKANConv3DLayer
from .kabn_conv import KABNConv1DLayer, KABNConv2DLayer, KABNConv3DLayer
from .kacn_conv import KACNConv1DLayer, KACNConv2DLayer, KACNConv3DLayer
from .kagn_bottleneck_conv import BottleNeckKAGNConv1DLayer, BottleNeckKAGNConv2DLayer, BottleNeckKAGNConv3DLayer
from .kagn_bottleneck_conv import MoEBottleNeckKAGNConv1DLayer, MoEBottleNeckKAGNConv2DLayer, \
    MoEBottleNeckKAGNConv3DLayer
from .kagn_conv import KAGNConv1DLayer, KAGNConv2DLayer, KAGNConv3DLayer
from .kajn_conv import KAJNConv1DLayer, KAJNConv2DLayer, KAJNConv3DLayer
from .kaln_conv import KALNConv1DLayer, KALNConv2DLayer, KALNConv3DLayer
from .kan_conv import KANConv1DLayer, KANConv2DLayer, KANConv3DLayer
from .relukan_bottleneck_conv import BottleNeckReLUKANConv1DLayer, BottleNeckReLUKANConv2DLayer, \
    BottleNeckReLUKANConv3DLayer
from .relukan_conv import ReLUKANConv1DLayer, ReLUKANConv2DLayer, ReLUKANConv3DLayer
from .wav_kan import WavKANConv1DLayer, WavKANConv2DLayer, WavKANConv3DLayer


class SelfKANtentionND(nn.Module):
    def __init__(self, input_dim, conv_kan_layer, inner_projection=None, **kankwargs):
        super(SelfKANtentionND, self).__init__()

        self.input_dim = input_dim
        self.ndim = None
        self.norm_layer = None

        kernel_size = kankwargs.pop('kernel_size')

        if conv_kan_layer in [FastKANConv1DLayer, KANConv1DLayer, KALNConv1DLayer, KACNConv1DLayer, KAGNConv1DLayer,
                              WavKANConv1DLayer, KAJNConv1DLayer, KABNConv1DLayer, BottleNeckKAGNConv1DLayer,
                              MoEBottleNeckKAGNConv1DLayer, ReLUKANConv1DLayer, BottleNeckReLUKANConv1DLayer]:
            self.ndim = 1
            self.norm_layer = nn.BatchNorm1d(input_dim)
        elif conv_kan_layer in [FastKANConv2DLayer, KANConv2DLayer, KALNConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer,
                                WavKANConv2DLayer, KAJNConv2DLayer, KABNConv2DLayer, BottleNeckKAGNConv2DLayer,
                                MoEBottleNeckKAGNConv2DLayer, ReLUKANConv2DLayer, BottleNeckReLUKANConv2DLayer]:
            self.ndim = 2
            self.norm_layer = nn.BatchNorm2d(input_dim)
        elif conv_kan_layer in [FastKANConv3DLayer, KANConv3DLayer, KALNConv3DLayer, KACNConv3DLayer, KAGNConv3DLayer,
                                WavKANConv3DLayer, KAJNConv3DLayer, KABNConv3DLayer, BottleNeckKAGNConv3DLayer,
                                MoEBottleNeckKAGNConv3DLayer, ReLUKANConv3DLayer, BottleNeckReLUKANConv3DLayer]:
            self.ndim = 3
            self.norm_layer = nn.BatchNorm3d(input_dim)
        assert self.ndim is not None, "Unsupported conv kan layer"

        self.inner_proj = None
        self.outer_proj = None
        if inner_projection is not None:
            if self.ndim == 1:
                self.inner_proj = nn.Conv1d(input_dim, inner_projection, 1)
                self.outer_proj = nn.Conv1d(inner_projection, input_dim, 1)
            if self.ndim == 2:
                self.inner_proj = nn.Conv2d(input_dim, inner_projection, 1)
                self.outer_proj = nn.Conv2d(inner_projection, input_dim, 1)
            if self.ndim == 3:
                self.inner_proj = nn.Conv3d(input_dim, inner_projection, 1)
                self.outer_proj = nn.Conv3d(inner_projection, input_dim, 1)

        dims = input_dim if inner_projection is None else inner_projection

        self.proj_k = conv_kan_layer(dims, dims, kernel_size, **kankwargs)
        self.proj_q = conv_kan_layer(dims, dims, kernel_size, **kankwargs)
        self.proj_v = conv_kan_layer(dims, dims, kernel_size, **kankwargs)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def attention(self, q, k, v):

        input_shape = v.size()
        m_batchsize = input_shape[0]
        total_pixels = prod(input_shape[2:])

        proj_query = q.view(m_batchsize, -1, total_pixels).permute(0, 2, 1)  # B X CX(N)
        proj_key = k.view(m_batchsize, -1, total_pixels)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = v.view(m_batchsize, -1, total_pixels)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(*input_shape)

        return out

    def forward(self, x):

        if self.inner_proj is not None:
            att = self.inner_proj(x)
        else:
            att = x

        q = self.proj_q(att)
        k = self.proj_k(att)
        v = self.proj_v(att)

        att = self.attention(q, k, v)

        if self.inner_proj is not None:
            att = self.outer_proj(att)

        return self.norm_layer(self.gamma * x + att)


class SelfKAGNtention1D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, degree=3, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm1d, **norm_kwargs):
        super(SelfKAGNtention1D, self).__init__(input_dim, KAGNConv1DLayer, inner_projection=inner_projection,
                                                kernel_size=kernel_size, degree=degree, groups=groups, padding=padding,
                                                stride=stride, dilation=dilation, dropout=dropout,
                                                norm_layer=norm_layer, **norm_kwargs)


class SelfKAGNtention2D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, degree=3, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm2d, **norm_kwargs):
        super(SelfKAGNtention2D, self).__init__(input_dim, KAGNConv2DLayer, inner_projection=inner_projection,
                                                kernel_size=kernel_size, degree=degree, groups=groups, padding=padding,
                                                stride=stride, dilation=dilation, dropout=dropout,
                                                norm_layer=norm_layer, **norm_kwargs)


class SelfKAGNtention3D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, degree=3, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm3d, **norm_kwargs):
        super(SelfKAGNtention3D, self).__init__(input_dim, KAGNConv3DLayer, inner_projection=inner_projection,
                                                kernel_size=kernel_size, degree=degree, groups=groups, padding=padding,
                                                stride=stride, dilation=dilation, dropout=dropout,
                                                norm_layer=norm_layer, **norm_kwargs)


class BottleNeckSelfKAGNtention1D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, degree=3, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm1d, **norm_kwargs):
        super(BottleNeckSelfKAGNtention1D, self).__init__(input_dim, BottleNeckKAGNConv1DLayer,
                                                          inner_projection=inner_projection,
                                                          kernel_size=kernel_size, degree=degree, groups=groups,
                                                          padding=padding,
                                                          stride=stride, dilation=dilation, dropout=dropout,
                                                          norm_layer=norm_layer, **norm_kwargs)


class BottleNeckSelfKAGNtention2D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, degree=3, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm2d, **norm_kwargs):
        super(BottleNeckSelfKAGNtention2D, self).__init__(input_dim, BottleNeckKAGNConv2DLayer,
                                                          inner_projection=inner_projection,
                                                          kernel_size=kernel_size, degree=degree, groups=groups,
                                                          padding=padding,
                                                          stride=stride, dilation=dilation, dropout=dropout,
                                                          norm_layer=norm_layer, **norm_kwargs)


class BottleNeckSelfKAGNtention3D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, degree=3, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm3d, **norm_kwargs):
        super(BottleNeckSelfKAGNtention3D, self).__init__(input_dim, BottleNeckKAGNConv3DLayer,
                                                          inner_projection=inner_projection,
                                                          kernel_size=kernel_size, degree=degree, groups=groups,
                                                          padding=padding,
                                                          stride=stride, dilation=dilation, dropout=dropout,
                                                          norm_layer=norm_layer, **norm_kwargs)


class SelfReLUKANtention1D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, g=5, k=3, train_ab=True, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm1d, **norm_kwargs):
        super(SelfReLUKANtention1D, self).__init__(input_dim, ReLUKANConv1DLayer, inner_projection=inner_projection,
                                                   kernel_size=kernel_size, g=g, k=k, train_ab=train_ab, groups=groups,
                                                   padding=padding,
                                                   stride=stride, dilation=dilation, dropout=dropout,
                                                   norm_layer=norm_layer, **norm_kwargs)


class SelfReLUKANtention2D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, g=5, k=3, train_ab=True, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm2d, **norm_kwargs):
        super(SelfReLUKANtention2D, self).__init__(input_dim, ReLUKANConv2DLayer, inner_projection=inner_projection,
                                                   kernel_size=kernel_size, g=g, k=k, train_ab=train_ab, groups=groups,
                                                   padding=padding,
                                                   stride=stride, dilation=dilation, dropout=dropout,
                                                   norm_layer=norm_layer, **norm_kwargs)


class SelfReLUKANtention3D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, g=5, k=3, train_ab=True, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm3d, **norm_kwargs):
        super(SelfReLUKANtention3D, self).__init__(input_dim, ReLUKANConv3DLayer, inner_projection=inner_projection,
                                                   kernel_size=kernel_size, g=g, k=k, train_ab=train_ab, groups=groups,
                                                   padding=padding,
                                                   stride=stride, dilation=dilation, dropout=dropout,
                                                   norm_layer=norm_layer, **norm_kwargs)


class BottleNeckSelfReLUKANtention1D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, g=5, k=3, train_ab=True, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm1d, **norm_kwargs):
        super(BottleNeckSelfReLUKANtention1D, self).__init__(input_dim, BottleNeckReLUKANConv1DLayer,
                                                             inner_projection=inner_projection,
                                                             kernel_size=kernel_size, g=g, k=k, train_ab=train_ab,
                                                             groups=groups,
                                                             padding=padding,
                                                             stride=stride, dilation=dilation, dropout=dropout,
                                                             norm_layer=norm_layer, **norm_kwargs)


class BottleNeckSelfReLUKANtention2D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, g=5, k=3, train_ab=True, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm2d, **norm_kwargs):
        super(BottleNeckSelfReLUKANtention2D, self).__init__(input_dim, BottleNeckReLUKANConv2DLayer,
                                                             inner_projection=inner_projection,
                                                             kernel_size=kernel_size, g=g, k=k, train_ab=train_ab,
                                                             groups=groups,
                                                             padding=padding,
                                                             stride=stride, dilation=dilation, dropout=dropout,
                                                             norm_layer=norm_layer, **norm_kwargs)


class BottleNeckSelfReLUKANtention3D(SelfKANtentionND):
    def __init__(self, input_dim, inner_projection=None, kernel_size=3, g=5, k=3, train_ab=True, groups=1,
                 padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 norm_layer=nn.BatchNorm3d, **norm_kwargs):
        super(BottleNeckSelfReLUKANtention3D, self).__init__(input_dim, BottleNeckReLUKANConv2DLayer,
                                                             inner_projection=inner_projection,
                                                             kernel_size=kernel_size, g=g, k=k, train_ab=train_ab,
                                                             groups=groups,
                                                             padding=padding,
                                                             stride=stride, dilation=dilation, dropout=dropout,
                                                             norm_layer=norm_layer, **norm_kwargs)
