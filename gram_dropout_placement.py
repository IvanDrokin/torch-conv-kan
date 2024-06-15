import os
from functools import lru_cache

import torch
import torch.nn as nn
from torch.nn.functional import conv2d

from kans import KAGN
from mnist_conv import train_and_validate
from utils import L1, NoiseInjection
from models import SimpleMoEConvKAGNBN


class KAGNConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, ndim: int = 2.,
                 dropout_poly: float = 0.0,
                 dropout_full: float = 0.0,
                 dropout_degree: float = 0.0,
                 drop_type='regular',
                 **norm_kwargs):
        super(KAGNConvNDLayer, self).__init__()
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
        self.norm_kwargs = norm_kwargs

        self.dropout_poly = None
        if dropout_poly > 0:
            if ndim == 1:
                self.dropout_poly = nn.Dropout1d(p=dropout_poly) if drop_type == 'regular' else NoiseInjection(
                    p=dropout_poly)
            if ndim == 2:
                self.dropout_poly = nn.Dropout2d(p=dropout_poly) if drop_type == 'regular' else NoiseInjection(
                    p=dropout_poly)
            if ndim == 3:
                self.dropout_poly = nn.Dropout3d(p=dropout_poly) if drop_type == 'regular' else NoiseInjection(
                    p=dropout_poly)

        self.dropout_full = None
        if dropout_full > 0:
            if ndim == 1:
                self.dropout_full = nn.Dropout1d(p=dropout_full) if drop_type == 'regular' else NoiseInjection(
                    p=dropout_full)
            if ndim == 2:
                self.dropout_full = nn.Dropout2d(p=dropout_full) if drop_type == 'regular' else NoiseInjection(
                    p=dropout_full)
            if ndim == 3:
                self.dropout_full = nn.Dropout3d(p=dropout_full) if drop_type == 'regular' else NoiseInjection(
                    p=dropout_full)

        self.dropout_degree = None
        if dropout_degree > 0:
            if ndim == 1:
                self.dropout_degree = nn.Dropout1d(p=dropout_degree) if drop_type == 'regular' else NoiseInjection(
                    p=dropout_degree)
            if ndim == 2:
                self.dropout_degree = nn.Dropout2d(p=dropout_degree) if drop_type == 'regular' else NoiseInjection(
                    p=dropout_degree)
            if ndim == 3:
                self.dropout_degree = nn.Dropout3d(p=dropout_degree) if drop_type == 'regular' else NoiseInjection(
                    p=dropout_degree)

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

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        poly_shape = (groups, output_dim // groups, (input_dim // groups) * (degree + 1)) + tuple(
            kernel_size for _ in range(ndim))

        self.poly_weights = nn.Parameter(torch.randn(*poly_shape))
        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        nn.init.kaiming_uniform_(self.poly_weights, nonlinearity='linear')
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

        return torch.concatenate(grams_basis, dim=1)

    def forward_kag(self, x, group_index):
        # Apply base activation to input and then linear transform with base weights

        if self.dropout_full is not None:
            x = self.dropout_full(x)
        basis = self.base_conv[group_index](self.base_activation(x))

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x = torch.tanh(x).contiguous()

        if self.dropout_poly is not None:
            x = self.dropout_poly(x)

        grams_basis = self.base_activation(self.gram_poly(x, self.degree))

        if self.dropout_degree is not None:
            grams_basis = self.dropout_degree(grams_basis)

        y = self.conv_w_fun(grams_basis, self.poly_weights[group_index],
                            stride=self.stride, dilation=self.dilation,
                            padding=self.padding, groups=1)

        y = self.base_activation(self.layer_norm[group_index](y + basis))

        return y

    def forward(self, x):

        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kag(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class KAGNConv2DLayer(KAGNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,

                 dropout_poly: float = 0.0,
                 dropout_full: float = 0.0,
                 dropout_degree: float = 0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(KAGNConv2DLayer, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout_poly=dropout_poly, dropout_full=dropout_full,
                                              dropout_degree=dropout_degree,
                                              **norm_kwargs)


class EightSimpleConvKAGN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout_poly: float = 0.0,
            dropout_full: float = 0.0,
            dropout_degree: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d,
            drop_type='regular'
    ):
        super(EightSimpleConvKAGN, self).__init__()

        self.layers = nn.Sequential(
            KAGNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KAGNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout_poly=dropout_poly, dropout_full=dropout_full,
                               dropout_degree=dropout_degree, affine=affine, norm_layer=norm_layer,
                               drop_type=drop_type),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout_poly=dropout_poly, dropout_full=dropout_full,
                               dropout_degree=dropout_degree, affine=affine, norm_layer=norm_layer,
                               drop_type=drop_type),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout_poly=dropout_poly, dropout_full=dropout_full,
                               dropout_degree=dropout_degree, affine=affine, norm_layer=norm_layer,
                               drop_type=drop_type),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout_poly=dropout_poly, dropout_full=dropout_full,
                               dropout_degree=dropout_degree, affine=affine, norm_layer=norm_layer,
                               drop_type=drop_type),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout_poly=dropout_poly, dropout_full=dropout_full,
                               dropout_degree=dropout_degree, affine=affine, norm_layer=norm_layer,
                               drop_type=drop_type),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout_poly=dropout_poly, dropout_full=dropout_full,
                               dropout_degree=dropout_degree, affine=affine, norm_layer=norm_layer,
                               drop_type=drop_type),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout_poly=dropout_poly, dropout_full=dropout_full,
                               dropout_degree=dropout_degree, affine=affine, norm_layer=norm_layer,
                               drop_type=drop_type),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAGN([layer_sizes[7], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)

    def forward(self, x, **kwargs):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class SixteenSimpleConvKAGN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            stride_indexes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout_poly: float = 0.0,
            dropout_full: float = 0.0,
            dropout_degree: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d,
            drop_type='regular'
    ):
        super(SixteenSimpleConvKAGN, self).__init__()

        self.layers = nn.Sequential()

        channels = [input_channels, ] + layer_sizes

        for index, (inp, out) in enumerate(zip(channels[:-1], channels[1:])):
            if index == 0:
                self.layers.append(
                    KAGNConv2DLayer(
                        inp, out, kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                        dilation=1, affine=affine, norm_layer=norm_layer
                    )
                )
            else:
                stride = 1
                if index + 1 in stride_indexes:
                    stride = 2

                self.layers.append(
                    L1(
                        KAGNConv2DLayer(
                            inp, out, kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=stride, dilation=1, dropout_poly=dropout_poly,
                            dropout_full=dropout_full,
                            dropout_degree=dropout_degree, affine=affine,
                            norm_layer=norm_layer, drop_type=drop_type
                        ),
                        l1_penalty
                    )
                )
        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[-1], num_classes))
        else:
            self.output = KAGN([layer_sizes[-1], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)

    def forward(self, x, **kwargs):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


def get_params_to_test():
    configs_to_test = []

    # degree_default = 3
    # degree_out_default = 1
    dropout_linear_default = 0.
    dropout_poly_default = 0.
    dropout_full_default = 0.
    dropout_degree_default = 0.
    l1_penalty_default = 0.
    default_drop_type = 'regular'
    l1_act_penalty_default = 0.0
    l2_act_penalty_default = 0.0

    # degree without_norm
    for degree, degree_out in [(3, 1), ]:
        configs_to_test.append(
            {
                'degree': degree,
                'degree_out': degree_out,
                'dropout_linear': dropout_linear_default,
                'l1_penalty': l1_penalty_default,
                'dropout_poly': dropout_poly_default,
                'dropout_full': dropout_full_default,
                'dropout_degree': dropout_degree_default,
                'run_name': f"no_reg_degree_{degree}_degree_out_{degree_out}",
                "drop_type": default_drop_type,
                'l1_activation_penalty': l1_act_penalty_default,
                'l2_act_penalty_default': l2_act_penalty_default
            }
        )
        for d in [0.05, 0.15, 0.25]:
            configs_to_test.append(
                {
                    'degree': degree,
                    'degree_out': degree_out,
                    'dropout_linear': dropout_linear_default,
                    'l1_penalty': l1_penalty_default,
                    'dropout_poly': d,
                    'dropout_full': dropout_full_default,
                    'dropout_degree': dropout_degree_default,
                    'run_name': f"dropout_poly_{d}_degree_{degree}_degree_out_{degree_out}",
                    "drop_type": default_drop_type,
                    'l1_activation_penalty': l1_act_penalty_default,
                    'l2_act_penalty_default': l2_act_penalty_default
                }
            )
            configs_to_test.append(
                {
                    'degree': degree,
                    'degree_out': degree_out,
                    'dropout_linear': dropout_linear_default,
                    'l1_penalty': l1_penalty_default,
                    'dropout_poly': dropout_poly_default,
                    'dropout_full': d,
                    'dropout_degree': dropout_degree_default,
                    'run_name': f"dropout_full{d}_degree_{degree}_degree_out_{degree_out}",
                    "drop_type": default_drop_type,
                    'l1_activation_penalty': l1_act_penalty_default,
                    'l2_act_penalty_default': l2_act_penalty_default
                }
            )
            configs_to_test.append(
                {
                    'degree': degree,
                    'degree_out': degree_out,
                    'dropout_linear': dropout_linear_default,
                    'l1_penalty': l1_penalty_default,
                    'dropout_poly': dropout_poly_default,
                    'dropout_full': dropout_full_default,
                    'dropout_degree': d,
                    'run_name': f"dropout_degree{d}_degree_{degree}_degree_out_{degree_out}",
                    "drop_type": default_drop_type,
                    'l1_activation_penalty': l1_act_penalty_default,
                    'l2_act_penalty_default': l2_act_penalty_default
                }
            )
        for reg in [1e-08, 1e-07, 1e-06]:
            configs_to_test.append(
                {
                    'degree': degree,
                    'degree_out': degree_out,
                    'dropout_linear': dropout_linear_default,
                    'l1_penalty': reg,
                    'dropout_poly': dropout_poly_default,
                    'dropout_full': dropout_full_default,
                    'dropout_degree': dropout_degree_default,
                    'run_name': f"l1_{reg}_degree_{degree}_degree_out_{degree_out}",
                    "drop_type": default_drop_type,
                    'l1_activation_penalty': l1_act_penalty_default,
                    'l2_act_penalty_default': l2_act_penalty_default
                }
            )

        for reg in [1e-08, 1e-07, 1e-06]:
            configs_to_test.append(
                {
                    'degree': degree,
                    'degree_out': degree_out,
                    'dropout_linear': dropout_linear_default,
                    'l1_penalty': l1_penalty_default,
                    'dropout_poly': dropout_poly_default,
                    'dropout_full': dropout_full_default,
                    'dropout_degree': dropout_degree_default,
                    'run_name': f"l1_act_{reg}_degree_{degree}_degree_out_{degree_out}",
                    "drop_type": default_drop_type,
                    'l1_activation_penalty': reg,
                    'l2_act_penalty_default': l2_act_penalty_default
                }
            )
            configs_to_test.append(
                {
                    'degree': degree,
                    'degree_out': degree_out,
                    'dropout_linear': dropout_linear_default,
                    'l1_penalty': l1_penalty_default,
                    'dropout_poly': dropout_poly_default,
                    'dropout_full': dropout_full_default,
                    'dropout_degree': dropout_degree_default,
                    'run_name': f"l2_act_{reg}_degree_{degree}_degree_out_{degree_out}",
                    "drop_type": default_drop_type,
                    'l1_activation_penalty': l1_act_penalty_default,
                    'l2_act_penalty_default': reg
                }
            )
        #
        for d in [0.05, 0.15, 0.25]:
            configs_to_test.append(
                {
                    'degree': degree,
                    'degree_out': degree_out,
                    'dropout_linear': dropout_linear_default,
                    'l1_penalty': l1_penalty_default,
                    'dropout_poly': d,
                    'dropout_full': dropout_full_default,
                    'dropout_degree': dropout_degree_default,
                    'run_name': f"noisy_dropout_poly_{d}_degree_{degree}_degree_out_{degree_out}",
                    "drop_type": 'noisy',
                    'l1_activation_penalty': l1_act_penalty_default,
                    'l2_act_penalty_default': l2_act_penalty_default
                }
            )
            configs_to_test.append(
                {
                    'degree': degree,
                    'degree_out': degree_out,
                    'dropout_linear': dropout_linear_default,
                    'l1_penalty': l1_penalty_default,
                    'dropout_poly': dropout_poly_default,
                    'dropout_full': d,
                    'dropout_degree': dropout_degree_default,
                    'run_name': f"noisy_dropout_full_{d}_degree_{degree}_degree_out_{degree_out}",
                    "drop_type": 'noisy',
                    'l1_activation_penalty': l1_act_penalty_default,
                    'l2_act_penalty_default': l2_act_penalty_default
                }
            )
            configs_to_test.append(
                {
                    'degree': degree,
                    'degree_out': degree_out,
                    'dropout_linear': dropout_linear_default,
                    'l1_penalty': l1_penalty_default,
                    'dropout_poly': dropout_poly_default,
                    'dropout_full': dropout_full_default,
                    'dropout_degree': d,
                    'run_name': f"noisy_dropout_degree_{d}_degree_{degree}_degree_out_{degree_out}",
                    "drop_type": 'noisy',
                    'l1_activation_penalty': l1_act_penalty_default,
                    'l2_act_penalty_default': l2_act_penalty_default
                }
            )

    return configs_to_test


def get_params_to_test_scales():
    configs_to_test = []

    # degree_default = 3
    # degree_out_default = 1
    dropout_full_default = 0.05

    # degree without_norm
    for degree, degree_out in [(7, 1), (5, 1), (3, 1)]:
        for ws in [4, 2, 1]:
            for model in ['deep', 'shallow', 'moe']:
                configs_to_test.append(
                    {
                        'degree': degree,
                        'degree_out': degree_out,
                        "width_scale": ws,
                        'model': model,
                        'dropout_linear': 0.05,
                        'l1_penalty': 0,
                        'dropout_poly': 0,
                        'dropout_full': dropout_full_default,
                        'dropout_degree': 0,
                        'run_name': f"{model}_width_{ws}_degree_{degree}_degree_out_{degree_out}",
                        "drop_type": 'noisy',
                        'l1_activation_penalty': 0,
                        'l2_act_penalty_default': 0
                    }
                )
    return configs_to_test


if __name__ == '__main__':
    list_of_params_to_test = get_params_to_test_scales()
    print(f"Total experiments to run: {len(list_of_params_to_test)}")
    dataset_name = 'CIFAR100'
    for model_params in list_of_params_to_test:
        print(f'Running {model_params["run_name"]} experiment')
        folder_to_save = os.path.join('experiments_scale', model_params['run_name'])
        if os.path.exists(folder_to_save):
            continue
        num_classes = 100 if dataset_name == 'CIFAR100' else 10
        input_channels = 1 if dataset_name == 'MNIST' else 3
        bs = 224
        epochs = 100
        _model_type = model_params.pop('model', 'shallow')
        ws = model_params.pop('width_scale', 1)

        is_moe = False
        if _model_type == 'shallow':
            kan_model = EightSimpleConvKAGN([16 * ws, 32 * ws, 64 * ws, 128 * ws,
                                             256 * ws, 256 * ws, 512 * ws, 512 * ws],
                                            num_classes=num_classes,
                                            input_channels=input_channels,
                                            degree=model_params['degree'],
                                            degree_out=model_params['degree_out'],
                                            dropout_linear=model_params['dropout_linear'],
                                            l1_penalty=model_params['l1_penalty'],
                                            dropout_poly=model_params['dropout_poly'],
                                            dropout_full=model_params['dropout_full'],
                                            dropout_degree=model_params['dropout_degree'],
                                            )
        elif _model_type == 'moe':
            kan_model = SimpleMoEConvKAGNBN([16 * ws, 32 * ws, 64 * ws, 128 * ws,
                                             256 * ws, 256 * ws, 512 * ws, 512 * ws],
                                            num_classes=num_classes,
                                            input_channels=input_channels,
                                            degree=model_params['degree'],
                                            degree_out=model_params['degree_out'],
                                            dropout_linear=model_params['dropout_linear'],
                                            l1_penalty=model_params['l1_penalty'],
                                            dropout=model_params['dropout_full'],
                                            num_experts=8,
                                            k=2
                                            )
            is_moe = True
        else:

            kan_model = SixteenSimpleConvKAGN([8 * ws, 8 * ws, 16 * ws, 16 * ws, 32 * ws, 32 * ws, 64 * ws, 64 * ws,
                                               128 * ws, 128 * ws, 128 * ws, 128 * ws, 256 * ws, 256 * ws, 256 * ws,
                                               256 * ws],
                                              [2, 4, 8],
                                              num_classes=num_classes,
                                              input_channels=input_channels,
                                              degree=model_params['degree'],
                                              degree_out=model_params['degree_out'],
                                              dropout_linear=model_params['dropout_linear'],
                                              l1_penalty=model_params['l1_penalty'],
                                              dropout_poly=model_params['dropout_poly'],
                                              dropout_full=model_params['dropout_full'],
                                              dropout_degree=model_params['dropout_degree'],
                                              )

        train_and_validate(kan_model, bs, epochs=epochs,
                           dataset_name=dataset_name,
                           model_save_dir=folder_to_save, is_moe=is_moe)  # Call the function to train and evaluate the model
