# taken from and based on https://github.com/1ssb/torchkan/blob/main/torchkan.py
# and https://github.com/1ssb/torchkan/blob/main/KALnet.py
# and https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py
# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# and https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
# and https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py
# and https://github.com/zavareh1/Wav-KAN
# and https://github.com/SpaceLearner/JacobiKAN/blob/main/JacobiKANLayer.py
import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from .utils import RadialBasisFunction


class KANLayer(nn.Module):
    def __init__(self, input_features, output_features, grid_size=5, spline_order=3, base_activation=nn.GELU,
                 grid_range=[-1, 1]):
        super(KANLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # The number of points in the grid for the spline interpolation.
        self.grid_size = grid_size
        # The order of the spline used in the interpolation.
        self.spline_order = spline_order
        # Activation function used for the initial transformation of the input.
        self.base_activation = base_activation()
        # The range of values over which the grid for spline interpolation is defined.
        self.grid_range = grid_range

        # Initialize the base weights with random values for the linear transformation.
        self.base_weight = nn.Parameter(torch.randn(output_features, input_features))
        # Initialize the spline weights with random values for the spline transformation.
        self.spline_weight = nn.Parameter(torch.randn(output_features, input_features, grid_size + spline_order))
        # Add a layer normalization for stabilizing the output of this layer.
        self.layer_norm = nn.LayerNorm(output_features)
        # Add a PReLU activation for this layer to provide a learnable non-linearity.
        self.prelu = nn.PReLU()

        # Compute the grid values based on the specified range and grid size.
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        ).expand(input_features, -1).contiguous()

        # Initialize the weights using Kaiming uniform distribution for better initial values.
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.spline_weight, nonlinearity='linear')

    def forward(self, x):
        # Process each layer using the defined base weights, spline weights, norms, and activations.
        grid = self.grid.to(x.device)
        # Move the input tensor to the device where the weights are located.

        # Perform the base linear transformation followed by the activation function.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype).to(x.device)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[:, :-(k + 1)]
            right_intervals = grid[:, k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x_uns) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        bases = bases.contiguous()

        # Compute the spline transformation and combine it with the base transformation.
        spline_output = F.linear(bases.view(x.size(0), -1), self.spline_weight.view(self.spline_weight.size(0), -1))
        # Apply layer normalization and PReLU activation to the combined output.
        x = self.prelu(self.layer_norm(base_output + spline_output))

        return x


class KALNLayer(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, input_features, output_features, degree=3, base_activation=nn.SiLU):
        super(KALNLayer, self).__init__()  # Initialize the parent nn.Module class

        self.input_features = input_features
        self.output_features = output_features
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # base_activation: Activation function used after each layer's computation
        self.base_activation = base_activation()

        # Base weight for linear transformation in each layer
        self.base_weight = nn.Parameter(torch.randn(output_features, input_features))
        # Polynomial weight for handling Legendre polynomial expansions
        self.poly_weight = nn.Parameter(torch.randn(output_features, input_features * (degree + 1)))
        # Layer normalization to stabilize learning and outputs
        self.layer_norm = nn.LayerNorm(output_features)

        # Initialize weights using Kaiming uniform distribution for better training start
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.poly_weight, nonlinearity='linear')

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Legendre polynomials
    def compute_legendre_polynomials(self, x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        legendre_polys = [P0, P1]

        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)

        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x):

        # Apply base activation to input and then linear transform with base weights
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        # Compute Legendre polynomials for the normalized x
        legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order)
        # Reshape legendre_basis to match the expected input dimensions for linear transformation
        legendre_basis = legendre_basis.view(x.size(0), -1)

        # Compute polynomial output using polynomial weights
        poly_output = F.linear(legendre_basis, self.poly_weight)
        # Combine base and polynomial outputs, normalize, and activate
        x = self.base_activation(self.layer_norm(base_output + poly_output))

        return x


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=nn.SiLU,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation()
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class GRAMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, degree=3, act=nn.SiLU):
        super(GRAMLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degrees = degree

        self.act = act()

        self.norm = nn.LayerNorm(out_channels, dtype=torch.float32)

        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

        self.grams_basis_weights = nn.Parameter(
            torch.zeros(in_channels, out_channels, degree + 1, dtype=torch.float32)
        )

        self.base_weights = nn.Parameter(
            torch.zeros(out_channels, in_channels, dtype=torch.float32)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (self.in_channels * (self.degrees + 1.0)),
        )

        nn.init.xavier_uniform_(self.grams_basis_weights)

        nn.init.xavier_uniform_(self.base_weights)

    def beta(self, n, m):
        return (
                       ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
               ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
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

        return torch.stack(grams_basis, dim=-1)

    def forward(self, x):

        basis = F.linear(self.act(x), self.base_weights)

        x = torch.tanh(x).contiguous()

        grams_basis = self.act(self.gram_poly(x, self.degrees))

        y = einsum(
            grams_basis,
            self.grams_basis_weights,
            "b l d, l o d -> b o",
        )

        y = self.act(self.norm(y + basis))

        y = y.view(-1, self.out_channels)

        return y


class WavKANLayer(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WavKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Linear weights for combining outputs
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features,
                                                 in_features))  # not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real

        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype,
                                          device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window

            # You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")
        wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
        wavelet_output = wavelet_weighted.sum(dim=2)
        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        base_output = F.linear(self.base_activation(x), self.weight1)

        # base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output + base_output

        # Apply batch normalization
        return self.bn(combined_output)


class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0, act=nn.SiLU):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.act = act()
        self.norm = nn.LayerNorm(output_dim, dtype=torch.float32)

        self.base_weights = nn.Parameter(
            torch.zeros(output_dim, input_dim, dtype=torch.float32)
        )

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        nn.init.xavier_uniform_(self.base_weights)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)

        basis = F.linear(self.act(x), self.base_weights)

        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:  ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            theta_k1 = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (
                    2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            theta_k2 = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (
                    i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :,
                                                                                                  i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)

        y = self.act(self.norm(y + basis))
        return y


class BernsteinKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, act=nn.SiLU):
        super(BernsteinKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.norm = nn.LayerNorm(output_dim, dtype=torch.float32)

        self.base_weights = nn.Parameter(
            torch.zeros(output_dim, input_dim, dtype=torch.float32)
        )

        self.bernstein_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        self.act = act()

        nn.init.normal_(self.bernstein_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        nn.init.xavier_uniform_(self.base_weights)

    @lru_cache(maxsize=128)
    def bernstein_poly(self, x, degree):

        bernsteins = torch.ones(x.shape + (self.degree + 1,), dtype=x.dtype, device=x.device)
        for j in range(1, degree + 1):
            for k in range(degree + 1 - j):
                bernsteins[..., k] = bernsteins[..., k] * (1 - x) + bernsteins[..., k + 1] * x
        return bernsteins

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)

        basis = F.linear(self.act(x), self.base_weights)

        # Since Bernstein polynomial is defined in [0, 1]
        # We need to normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        bernsteins = self.bernstein_poly(x, self.degree)
        y = torch.einsum('bid,iod->bo', bernsteins, self.bernstein_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)

        y = self.act(self.norm(y + basis))
        return y


class ReLUKANLayer(nn.Module):

    def __init__(self,
                 input_size: int,
                 g: int,
                 k: int,
                 output_size: int,
                 train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4 * g * g / ((k + 1) * (k + 1))
        self.input_size, self.output_size = input_size, output_size
        # modification here
        phase_low = torch.arange(-k, g) / g
        phase_high = phase_low + (k + 1) / g
        # modification here
        self.phase_low = nn.Parameter(phase_low[None, :].expand(input_size, -1),
                                      requires_grad=train_ab)
        # modification here, and: `phase_height` to `phase_high`
        self.phase_high = nn.Parameter(phase_high[None, :].expand(input_size, -1),
                                       requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g + k, input_size))

    def forward(self, x):
        x = x[..., None]
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        x = x.reshape((len(x), self.output_size))
        return x


class BottleNeckGRAMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, degree=3, act=nn.SiLU,
                 dim_reduction: float = 8, min_internal: int = 16):
        super(BottleNeckGRAMLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degrees = degree

        self.dim_reduction = dim_reduction
        self.min_internal = min_internal

        inner_dim = int(max(in_channels / dim_reduction,
                            out_channels / dim_reduction))
        if inner_dim < min_internal:
            self.inner_dim = min(min_internal, in_channels, out_channels)
        else:
            self.inner_dim = inner_dim

        self.act = act()

        self.inner_proj = nn.Linear(in_channels, self.inner_dim)
        self.outer_proj = nn.Linear(self.inner_dim, out_channels)

        self.norm = nn.LayerNorm(out_channels, dtype=torch.float32)

        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

        self.grams_basis_weights = nn.Parameter(
            torch.zeros(self.inner_dim, self.inner_dim, degree + 1, dtype=torch.float32)
        )

        self.base_weights = nn.Parameter(
            torch.zeros(out_channels, in_channels, dtype=torch.float32)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (self.in_channels * (self.degrees + 1.0)),
        )

        nn.init.xavier_uniform_(self.grams_basis_weights)

        nn.init.xavier_uniform_(self.base_weights)
        nn.init.xavier_uniform_(self.inner_proj.weight)
        nn.init.xavier_uniform_(self.outer_proj.weight)

    def beta(self, n, m):
        return (
                       ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
               ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
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

        return torch.stack(grams_basis, dim=-1)

    def forward(self, x):

        basis = F.linear(self.act(x), self.base_weights)

        x = self.inner_proj(x)

        x = torch.tanh(x).contiguous()

        grams_basis = self.act(self.gram_poly(x, self.degrees))

        y = einsum(
            grams_basis,
            self.grams_basis_weights,
            "b l d, l o d -> b o",
        )

        y = self.outer_proj(y)

        y = self.act(self.norm(y + basis))

        y = y.view(-1, self.out_channels)

        return y
