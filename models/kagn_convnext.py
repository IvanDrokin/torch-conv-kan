# ConvNeXt V2-style model built from KAGN blocks.
#
# Macro design (stem, stages, downsampling, head, size configs, init) follows:
#     Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with
#     Masked Autoencoders", CVPR 2023. https://arxiv.org/abs/2301.00808
# Reference implementation: https://github.com/facebookresearch/ConvNeXt-V2
from typing import Dict, Tuple

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .conv_kagn_layers import (ConvNeXtKAGNBlock, KAGNConv1DLayerV3,
                               KAGNConv2DLayerV3, KAGNConv3DLayerV3)  # adjust to actual module path

# Original ConvNeXt V2 size configs: (depths, dims)
cfgs: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {
    "Atto":  ((2, 2, 6, 2),  (40, 80, 160, 320)),
    "Femto": ((2, 2, 6, 2),  (48, 96, 192, 384)),
    "Pico":  ((2, 2, 6, 2),  (64, 128, 256, 512)),
    "Nano":  ((2, 2, 8, 2),  (80, 160, 320, 640)),
    "Tiny":  ((3, 3, 9, 3),  (96, 192, 384, 768)),
    "Base":  ((3, 3, 27, 3), (128, 256, 512, 1024)),
    "Large": ((3, 3, 27, 3), (192, 384, 768, 1536)),
    "Huge":  ((3, 3, 27, 3), (352, 704, 1408, 2816)),
}

_KAGN_LAYER_TYPES = (KAGNConv1DLayerV3, KAGNConv2DLayerV3, KAGNConv3DLayerV3)


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim for channels_first (N, C, H, W) tensors."""

    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # direct channels_first layer norm, avoids permute round-trips
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtKAGN(nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_channels: int = 3, num_classes: int = 1000,
                 convnext_type: str = 'Tiny',
                 degree: int = 3, use_grn: bool = True,
                 dropout: float = 0.0, drop_path_rate: float = 0.0,
                 head_init_scale: float = 1.0,
                 norm_layer: nn.Module = nn.BatchNorm2d, affine: bool = True):
        super().__init__()
        depths, dims = cfgs[convnext_type]
        self.depths = depths
        self.dims = dims

        norm_kwargs = {'affine': affine} if norm_layer in (nn.InstanceNorm2d, nn.BatchNorm2d) else {}

        # stem ("patchify") + 3 intermediate downsampling convs
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        # stages: plain stacks of blocks with a linear drop_path schedule over depth
        self.stages = nn.ModuleList()
        dp_rates = [r.item() for r in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                ConvNeXtKAGNBlock(dims[i], degree=degree, use_grn=use_grn,
                                  drop_path=dp_rates[cur + j], dropout=dropout,
                                  norm_layer=norm_layer, **norm_kwargs)
                for j in range(depths[i])
            ]))
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else None

        self._init_weights(head_init_scale)

    def _init_weights(self, head_init_scale: float):
        # Collect submodules owned by KAGN layers: they carry their own
        # initialization (Kaiming for convs + custom beta_weights), which a
        # blanket trunc_normal_ pass would silently destroy.
        kagn_owned = set()
        for m in self.modules():
            if isinstance(m, _KAGN_LAYER_TYPES):
                kagn_owned.update(id(sm) for sm in m.modules())

        for m in self.modules():
            if id(m) in kagn_owned:
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.head is not None and head_init_scale != 1.0:
            # used by the original repo when fine-tuning large models
            with torch.no_grad():
                self.head.weight.mul_(head_init_scale)
                self.head.bias.mul_(head_init_scale)

    def forward_features(self, x):
        """Returns feature maps of all 4 stages (useful for dense prediction)."""
        features = []
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)
            features.append(x)
        return features

    def forward(self, x):
        x = self.forward_features(x)[-1]
        x = self.norm(x.mean(dim=(2, 3)))  # global average pooling -> (N, C)
        if self.head is None:
            return x
        return self.head(x)


def _convnext_kagn(convnext_type: str, input_channels: int, num_classes: int, **kwargs) -> ConvNeXtKAGN:
    return ConvNeXtKAGN(input_channels, num_classes, convnext_type=convnext_type, **kwargs)


def convnext_kagn_atto(input_channels, num_classes, **kwargs):
    return _convnext_kagn('Atto', input_channels, num_classes, **kwargs)


def convnext_kagn_femto(input_channels, num_classes, **kwargs):
    return _convnext_kagn('Femto', input_channels, num_classes, **kwargs)


def convnext_kagn_pico(input_channels, num_classes, **kwargs):
    return _convnext_kagn('Pico', input_channels, num_classes, **kwargs)


def convnext_kagn_nano(input_channels, num_classes, **kwargs):
    return _convnext_kagn('Nano', input_channels, num_classes, **kwargs)


def convnext_kagn_tiny(input_channels, num_classes, **kwargs):
    return _convnext_kagn('Tiny', input_channels, num_classes, **kwargs)


def convnext_kagn_base(input_channels, num_classes, **kwargs):
    return _convnext_kagn('Base', input_channels, num_classes, **kwargs)


def convnext_kagn_large(input_channels, num_classes, **kwargs):
    return _convnext_kagn('Large', input_channels, num_classes, **kwargs)


def convnext_kagn_huge(input_channels, num_classes, **kwargs):
    return _convnext_kagn('Huge', input_channels, num_classes, **kwargs)