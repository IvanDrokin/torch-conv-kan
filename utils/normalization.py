import torch
import torch.nn as nn


class SelfSpatialNorm(nn.Module):

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 16,
        affine: bool = True
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=num_channels, num_groups=num_groups, eps=1e-6, affine=affine)
        self.conv_y = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(f) + self.conv_b(f)
        return new_f


class SpatialNorm(nn.Module):

    def __init__(
        self,
        num_channels: int,
        num_channels_cond: int,
        num_groups: int = 32,
        affine: bool = True
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=num_channels, num_groups=num_groups, eps=1e-6, affine=affine)
        self.conv_y = nn.Conv2d(num_channels_cond, num_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(num_channels_cond, num_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(c) + self.conv_b(c)
        return new_f
