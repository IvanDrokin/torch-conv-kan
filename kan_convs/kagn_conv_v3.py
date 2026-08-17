import torch
from torch.nn.functional import conv3d, conv2d, conv1d
import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (stochastic depth) per sample, applied in the main path
    of residual blocks. No-op at inference or when drop_prob == 0."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # one bernoulli mask per sample, broadcast over all remaining dims,
    # so the same code works for 1D/2D/3D convs and sequence tensors
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)  # rescale to keep the expected value unchanged
    return x * random_tensor


class DropPath(nn.Module):
    """Stochastic depth: randomly zeroes the whole residual branch per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class GRNNd(nn.Module):
    """Global Response Normalization for channels_first tensors (N, C, *spatial).

    Aggregates an L2 norm over spatial dims, normalizes it across channels
    (divisive normalization), and applies a learnable zero-init calibration,
    so the layer starts as an identity mapping.
    """

    def __init__(self, dim, ndim: int = 2, eps: float = 1e-6):
        super().__init__()
        shape = (1, dim) + (1,) * ndim
        self.gamma = nn.Parameter(torch.zeros(*shape))
        self.beta = nn.Parameter(torch.zeros(*shape))
        self.spatial_dims = tuple(range(2, 2 + ndim))
        self.eps = eps

    def forward(self, x):
        # (N, C, 1...) - per-channel spatial energy
        gx = torch.norm(x, p=2, dim=self.spatial_dims, keepdim=True)
        # contrast across channels
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class KAGNConvNDLayerV3(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2,
                 use_grn: bool = False, **norm_kwargs):
        super(KAGNConvNDLayerV3, self).__init__()
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

        self.poly_conv = conv_class(input_dim * (degree + 1),
                                    output_dim,
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    groups=groups,
                                    bias=False)

        # GRN on the expanded polynomial basis, analogous to GRN on the
        # 4x-expanded hidden features in ConvNeXtV2's MLP
        self.grn = GRNNd(input_dim * (degree + 1), ndim=ndim) if use_grn else None

        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.base_conv.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.poly_conv.weight, nonlinearity='linear')

        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / ((kernel_size ** ndim) * self.inputdim * (self.degree + 1.0)),
        )

    def beta(self, n, m):
        return (
                       ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
               ) * self.beta_weights[n]

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

        indexes = [i * (degree + 1) + j for i in range(x.shape[1]) for j in range(degree + 1)]

        grams_basis = torch.concatenate(grams_basis, dim=1)
        grams_basis = grams_basis[:, indexes]
        return grams_basis

    def forward_kag(self, x):
        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv(self.base_activation(x))

        # Normalize x to the range [-1, 1] for stable polynomial computation
        x = torch.tanh(x).contiguous()

        if self.dropout is not None:
            x = self.dropout(x)

        grams_basis = self.base_activation(self.gram_poly(x, self.degree))

        # De-correlate channels of the expanded basis before projecting back
        if self.grn is not None:
            grams_basis = self.grn(grams_basis)

        y = self.poly_conv(grams_basis)

        y = self.base_activation(self.layer_norm(y + basis))

        return y

    def forward(self, x):
        return self.forward_kag(x)


class KAGNConv3DLayerV3(KAGNConvNDLayerV3):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, use_grn: bool = False, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(KAGNConv3DLayerV3, self).__init__(nn.Conv3d, norm_layer, conv3d,
                                                input_dim, output_dim,
                                                degree, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=3, dropout=dropout, use_grn=use_grn, **norm_kwargs)


class KAGNConv2DLayerV3(KAGNConvNDLayerV3):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, use_grn: bool = False, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(KAGNConv2DLayerV3, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                                input_dim, output_dim,
                                                degree, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=2, dropout=dropout, use_grn=use_grn, **norm_kwargs)


class KAGNConv1DLayerV3(KAGNConvNDLayerV3):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, use_grn: bool = False, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(KAGNConv1DLayerV3, self).__init__(nn.Conv1d, norm_layer, conv1d,
                                                input_dim, output_dim,
                                                degree, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=1, dropout=dropout, use_grn=use_grn, **norm_kwargs)


class ConvNeXtKAGNBlock(nn.Module):
    """ConvNeXt V2-style block with KAGN (Gram KAN) convolutions.

    The block layout (7x7 depthwise conv -> LayerNorm -> pointwise mixing ->
    residual) and the Global Response Normalization (GRN) layer follow:

        Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with
        Masked Autoencoders", CVPR 2023.
        https://arxiv.org/abs/2301.00808

    Here the pointwise MLP (pwconv1 -> GELU -> GRN -> pwconv2) is replaced with a
    single 1x1 KAGN convolution; with `use_grn=True`, GRN is applied inside the
    KAGN layer on the (degree + 1)-expanded Gram polynomial basis, mirroring the
    role it plays on the 4x-expanded hidden features in the original block.

    KAGN layer based on GRAM KAN:
        https://github.com/Khochawongwat/GRAMKAN
    """

    def __init__(self, dim, degree=3, use_grn: bool = True, drop_path: float = 0.0,
                 dropout: float = 0.0, norm_layer=nn.BatchNorm2d, **norm_kwargs):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.kagn = KAGNConv2DLayerV3(dim, dim, kernel_size=1, degree=degree,
                                      use_grn=use_grn, dropout=dropout,
                                      norm_layer=norm_layer, **norm_kwargs)

        if drop_path > 0.0 and DropPath is None:
            warnings.warn(
                f"drop_path={drop_path} requested but timm is not installed; "
                "using nn.Identity instead (no stochastic depth).",
                RuntimeWarning,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.kagn(x)
        return shortcut + self.drop_path(x)