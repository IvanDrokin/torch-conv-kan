'''
Based on https://github.com/zavareh1/Wav-KAN
This is a sample code for the simulations of the paper:
Bozorgasl, Zavareh and Chen, Hao, Wav-KAN: Wavelet Kolmogorov-Arnold Networks (May, 2024)

https://arxiv.org/abs/2405.12832
and also available at:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4835325
We used efficient KAN notation and some part of the code:https://github.com/Blealtan/efficient-kan

'''
import math

import torch
import torch.nn as nn


class WaveletConvND(nn.Module):
    def __init__(self, conv_class, input_dim, output_dim, kernel_size,
                 padding=0, stride=1, dilation=1,
                 ndim: int = 2, wavelet_type='mexican_hat'):
        super(WaveletConvND, self).__init__()

        _shapes = (1, output_dim, input_dim) + tuple(1 for _ in range(ndim))

        self.scale = nn.Parameter(torch.ones(*_shapes))
        self.translation = nn.Parameter(torch.zeros(*_shapes))

        self.ndim = ndim
        self.wavelet_type = wavelet_type

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.wavelet_weights = nn.ModuleList([conv_class(input_dim,
                                                         1,
                                                         kernel_size,
                                                         stride,
                                                         padding,
                                                         dilation,
                                                         groups=1,
                                                         bias=False) for _ in range(output_dim)])

        self.wavelet_out = conv_class(output_dim, output_dim, 1, 1, 0, dilation, groups=1, bias=False)

        for conv_layer in self.wavelet_weights:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.wavelet_out.weight, nonlinearity='linear')

    @staticmethod
    def _forward_mexican_hat(x):
        term1 = ((x ** 2) - 1)
        term2 = torch.exp(-0.5 * x ** 2)
        wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
        return wavelet

    @staticmethod
    def _forward_morlet(x):
        omega0 = 5.0  # Central frequency
        real = torch.cos(omega0 * x)
        envelope = torch.exp(-0.5 * x ** 2)
        wavelet = envelope * real
        return wavelet

    @staticmethod
    def _forward_dog(x):
        return -x * torch.exp(-0.5 * x ** 2)

    @staticmethod
    def _forward_meyer(x):
        v = torch.abs(x)
        pi = math.pi

        def meyer_aux(v):
            return torch.where(v <= 1 / 2, torch.ones_like(v),
                               torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

        def nu(t):
            return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

        # Meyer wavelet calculation using the auxiliary function
        wavelet = torch.sin(pi * v) * meyer_aux(v)
        return wavelet

    def _forward_shannon(self, x):
        pi = math.pi
        sinc = torch.sinc(x / pi)  # sinc(x) = sin(pi*x) / (pi*x)

        _shape = (1, 1, x.size(2)) + tuple(1 for _ in range(self.ndim))
        # Applying a Hamming window to limit the infinite support of the sinc function
        window = torch.hamming_window(x.size(2), periodic=False, dtype=x.dtype,
                                      device=x.device).view(*_shape)
        # Shannon wavelet is the product of the sinc function and the window
        wavelet = sinc * window
        return wavelet

    def forward(self, x):
        x_expanded = x.unsqueeze(1)

        x_scaled = (x_expanded - self.translation) / self.scale

        if self.wavelet_type == 'mexican_hat':
            wavelet = self._forward_mexican_hat(x_scaled)
        elif self.wavelet_type == 'morlet':
            wavelet = self._forward_morlet(x_scaled)
        elif self.wavelet_type == 'dog':
            wavelet = self._forward_dog(x_scaled)
        elif self.wavelet_type == 'meyer':
            wavelet = self._forward_meyer(x_scaled)
        elif self.wavelet_type == 'shannon':
            wavelet = self._forward_shannon(x_scaled)
        else:
            raise ValueError("Unsupported wavelet type")

        wavelet_x = torch.split(wavelet, 1, dim=1)
        output = []
        for group_ind, _x in enumerate(wavelet_x):
            y = self.wavelet_weights[group_ind](_x.squeeze(1))
            # output.append(y.clone())
            output.append(y)
        y = torch.cat(output, dim=1)
        y = self.wavelet_out(y)
        return y


class WaveletConvNDFastPlusOne(WaveletConvND):
    def __init__(self, conv_class, conv_class_d_plus_one, input_dim, output_dim, kernel_size,
                 padding=0, stride=1, dilation=1,
                 ndim: int = 2, wavelet_type='mexican_hat'):
        super(WaveletConvND, self).__init__()

        assert ndim < 3, "fast_plus_one version suppoerts only 1D and 2D convs"

        _shapes = (1, output_dim, input_dim) + tuple(1 for _ in range(ndim))

        self.scale = nn.Parameter(torch.ones(*_shapes))
        self.translation = nn.Parameter(torch.zeros(*_shapes))

        self.ndim = ndim
        self.wavelet_type = wavelet_type

        self.input_dim = input_dim
        self.output_dim = output_dim

        kernel_size_plus = (input_dim,) + kernel_size if isinstance(kernel_size, tuple) else (input_dim,) + (
        kernel_size,) * ndim
        stride_plus = (1,) + stride if isinstance(stride, tuple) else (1,) + (stride,) * ndim
        padding_plus = (0,) + padding if isinstance(padding, tuple) else (0,) + (padding,) * ndim
        dilation_plus = (1,) + dilation if isinstance(dilation, tuple) else (1,) + (dilation,) * ndim

        self.wavelet_weights = conv_class_d_plus_one(output_dim,
                                                     output_dim,
                                                     kernel_size_plus,
                                                     stride_plus,
                                                     padding_plus,
                                                     dilation_plus,
                                                     groups=output_dim,
                                                     bias=False)

        self.wavelet_out = conv_class(output_dim, output_dim, 1, 1, 0, dilation, groups=1, bias=False)

        nn.init.kaiming_uniform_(self.wavelet_weights.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.wavelet_out.weight, nonlinearity='linear')

    def forward(self, x):
        x_expanded = x.unsqueeze(1)

        x_scaled = (x_expanded - self.translation) / self.scale

        if self.wavelet_type == 'mexican_hat':
            wavelet = self._forward_mexican_hat(x_scaled)
        elif self.wavelet_type == 'morlet':
            wavelet = self._forward_morlet(x_scaled)
        elif self.wavelet_type == 'dog':
            wavelet = self._forward_dog(x_scaled)
        elif self.wavelet_type == 'meyer':
            wavelet = self._forward_meyer(x_scaled)
        elif self.wavelet_type == 'shannon':
            wavelet = self._forward_shannon(x_scaled)
        else:
            raise ValueError("Unsupported wavelet type")
        # wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
        # wavelet_output = wavelet_weighted.sum(dim=2)

        y = self.wavelet_weights(wavelet).squeeze(2)
        y = self.wavelet_out(y)
        return y


class WaveletConvNDFast(WaveletConvND):
    def __init__(self, conv_class, input_dim, output_dim, kernel_size,
                 padding=0, stride=1, dilation=1,
                 ndim: int = 2, wavelet_type='mexican_hat'):
        super(WaveletConvND, self).__init__()

        _shapes = (1, output_dim, input_dim) + tuple(1 for _ in range(ndim))

        self.scale = nn.Parameter(torch.ones(*_shapes))
        self.translation = nn.Parameter(torch.zeros(*_shapes))

        self.ndim = ndim
        self.wavelet_type = wavelet_type

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.wavelet_weights = conv_class(output_dim * input_dim,
                                          output_dim,
                                          kernel_size,
                                          stride,
                                          padding,
                                          dilation,
                                          groups=output_dim,
                                          bias=False)

        self.wavelet_out = conv_class(output_dim, output_dim, 1, 1, 0, dilation, groups=1, bias=False)

        nn.init.kaiming_uniform_(self.wavelet_weights.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.wavelet_out.weight, nonlinearity='linear')

    def forward(self, x):
        x_expanded = x.unsqueeze(1)

        x_scaled = (x_expanded - self.translation) / self.scale

        if self.wavelet_type == 'mexican_hat':
            wavelet = self._forward_mexican_hat(x_scaled)
        elif self.wavelet_type == 'morlet':
            wavelet = self._forward_morlet(x_scaled)
        elif self.wavelet_type == 'dog':
            wavelet = self._forward_dog(x_scaled)
        elif self.wavelet_type == 'meyer':
            wavelet = self._forward_meyer(x_scaled)
        elif self.wavelet_type == 'shannon':
            wavelet = self._forward_shannon(x_scaled)
        else:
            raise ValueError("Unsupported wavelet type")
        # wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
        # wavelet_output = wavelet_weighted.sum(dim=2)

        y = self.wavelet_weights(wavelet.flatten(1, 2))
        y = self.wavelet_out(y)
        return y


class WavKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, conv_class_plus1, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, wav_version: str = 'base',
                 ndim: int = 2, dropout=0.0, wavelet_type='mexican_hat', **norm_kwargs):
        super(WavKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.norm_kwargs = norm_kwargs
        assert wavelet_type in ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'], \
            ValueError(f"Unsupported wavelet type: {wavelet_type}")
        self.wavelet_type = wavelet_type

        self.dropout = None
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
        if wav_version == 'base':
            self.wavelet_conv = nn.ModuleList(
                [
                    WaveletConvND(
                        conv_class,
                        input_dim // groups,
                        output_dim // groups,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        ndim=ndim, wavelet_type=wavelet_type
                    ) for _ in range(groups)
                ]
            )
        elif wav_version == 'fast':
            self.wavelet_conv = nn.ModuleList(
                [
                    WaveletConvNDFast(
                        conv_class,
                        input_dim // groups,
                        output_dim // groups,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        ndim=ndim, wavelet_type=wavelet_type
                    ) for _ in range(groups)
                ]
            )
        elif wav_version == 'fast_plus_one':

            self.wavelet_conv = nn.ModuleList(
                [
                    WaveletConvNDFastPlusOne(
                        conv_class, conv_class_plus1,
                        input_dim // groups,
                        output_dim // groups,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        ndim=ndim, wavelet_type=wavelet_type
                    ) for _ in range(groups)
                ]
            )

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.base_activation = nn.SiLU()

    def forward_wavkan(self, x, group_ind):
        # You may like test the cases like Spl-KAN
        base_output = self.base_conv[group_ind](self.base_activation(x))

        if self.dropout is not None:
            x = self.dropout(x)

        wavelet_output = self.wavelet_conv[group_ind](x)

        combined_output = wavelet_output + base_output

        # Apply batch normalization
        return self.layer_norm[group_ind](combined_output)

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_wavkan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class WavKANConv3DLayer(WavKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, wavelet_type='mexican_hat', norm_layer=nn.BatchNorm3d,
                 wav_version: str = 'fast', **norm_kwargs):
        super(WavKANConv3DLayer, self).__init__(nn.Conv3d, None, norm_layer, input_dim, output_dim, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=3, dropout=dropout, wavelet_type=wavelet_type,
                                                wav_version=wav_version, **norm_kwargs)


class WavKANConv2DLayer(WavKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, wavelet_type='mexican_hat', norm_layer=nn.BatchNorm2d,
                 wav_version: str = 'fast', **norm_kwargs):
        super(WavKANConv2DLayer, self).__init__(nn.Conv2d, nn.Conv3d, norm_layer, input_dim, output_dim, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=2, dropout=dropout, wavelet_type=wavelet_type,
                                                wav_version=wav_version, **norm_kwargs)


class WavKANConv1DLayer(WavKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, wavelet_type='mexican_hat', norm_layer=nn.BatchNorm1d,
                 wav_version: str = 'fast', **norm_kwargs):
        super(WavKANConv1DLayer, self).__init__(nn.Conv1d, nn.Conv2d, norm_layer, input_dim, output_dim, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=1, dropout=dropout, wavelet_type=wavelet_type,
                                                wav_version=wav_version, **norm_kwargs)
