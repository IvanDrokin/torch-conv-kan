# Based on this: https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py
from functools import lru_cache

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn.functional import conv3d, conv2d, conv1d

from utils import NoiseInjection
from .moe_utils import SparseDispatcher
from kans import GRAMLayer


class BottleNeckKAGNConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2.,
                 dim_reduction: float = 4, min_internal: int = 16,
                 **norm_kwargs):
        super(BottleNeckKAGNConvNDLayer, self).__init__()
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

        inner_dim = int(max((input_dim // groups) / dim_reduction,
                            (output_dim // groups) / dim_reduction))
        if inner_dim < min_internal:
            self.inner_dim = min(min_internal, input_dim // groups, output_dim // groups)
        else:
            self.inner_dim = inner_dim
        if dropout > 0:
            self.dropout = NoiseInjection(p=dropout, alpha=0.05)

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

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        poly_shape = (groups, self.inner_dim, self.inner_dim * (degree + 1)) + tuple(
            kernel_size for _ in range(ndim))

        self.poly_weights = nn.Parameter(torch.randn(*poly_shape))
        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.inner_proj:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.out_proj:
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

        if self.dropout is not None:
            x = self.dropout(x)

        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x = self.inner_proj[group_index](x)
        x = torch.tanh(x).contiguous()

        grams_basis = self.base_activation(self.gram_poly(x, self.degree))

        y = self.conv_w_fun(grams_basis, self.poly_weights[group_index],
                            stride=self.stride, dilation=self.dilation,
                            padding=self.padding, groups=1)
        y = self.out_proj[group_index](y)

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


class BottleNeckKAGNConv3DLayer(BottleNeckKAGNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm3d, dim_reduction: float = 4, **norm_kwargs):
        super(BottleNeckKAGNConv3DLayer, self).__init__(nn.Conv3d, norm_layer, conv3d,
                                                        input_dim, output_dim,
                                                        degree, kernel_size, dim_reduction=dim_reduction,
                                                        groups=groups, padding=padding, stride=stride,
                                                        dilation=dilation,
                                                        ndim=3, dropout=dropout, **norm_kwargs)


class BottleNeckKAGNConv2DLayer(BottleNeckKAGNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, dim_reduction: float = 4, **norm_kwargs):
        super(BottleNeckKAGNConv2DLayer, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                                        input_dim, output_dim,
                                                        degree, kernel_size, dim_reduction=dim_reduction,
                                                        groups=groups, padding=padding, stride=stride,
                                                        dilation=dilation,
                                                        ndim=2, dropout=dropout, **norm_kwargs)


class BottleNeckKAGNConv1DLayer(BottleNeckKAGNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm1d, dim_reduction: float = 4, **norm_kwargs):
        super(BottleNeckKAGNConv1DLayer, self).__init__(nn.Conv1d, norm_layer, conv1d,
                                                        input_dim, output_dim,
                                                        degree, kernel_size, dim_reduction=dim_reduction,
                                                        groups=groups, padding=padding, stride=stride,
                                                        dilation=dilation,
                                                        ndim=1, dropout=dropout, **norm_kwargs)


class KAGNExpert(nn.Module):
    def __init__(self, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2.):
        super(KAGNExpert, self).__init__()
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
        self.p_dropout = dropout

        if dropout > 0:
            self.dropout = NoiseInjection(p=dropout, alpha=0.05)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        poly_shape = (groups, self.outdim // groups, self.inputdim * (degree + 1) // groups) + tuple(
            kernel_size for _ in range(ndim))

        self.poly_weights = nn.Parameter(torch.randn(*poly_shape))
        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

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

        x = torch.tanh(x).contiguous()

        grams_basis = self.base_activation(self.gram_poly(x, self.degree))

        if self.dropout is not None:
            grams_basis = self.dropout(grams_basis)

        y = self.conv_w_fun(grams_basis, self.poly_weights[group_index],
                            stride=self.stride, dilation=self.dilation,
                            padding=self.padding, groups=1)

        return y

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kag(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class KAGNMoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, num_experts, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2.,
                 noisy_gating=True, k=4, pregate: bool = False):
        super(KAGNMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_dim
        self.input_size = input_dim
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([KAGNExpert(conv_w_fun, input_dim, output_dim, degree, kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 dropout=dropout, ndim=ndim)
                                      for _ in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, num_experts), requires_grad=True)

        self.pre_gate = None
        if pregate:
            self.pre_gate = GRAMLayer(input_dim, output_dim, degree=degree)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        if ndim == 1:
            self.avgpool = nn.AdaptiveAvgPool1d((1,))
            self.conv_dims = 1
        elif ndim == 2:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.conv_dims = 2
        elif ndim == 3:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.conv_dims = 3
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gate_x = torch.flatten(self.avgpool(x), 1)
        if self.pre_gate:
            gate_x = self.pre_gate(gate_x)
        gates, load = self.noisy_top_k_gating(gate_x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs, self.conv_dims)
        return y, loss


class MoEBottleNeckKAGNConvND(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, conv_class, conv_w_fun, norm_class, input_dim, output_dim, num_experts=16,
                 noisy_gating=True, k=4, kernel_size=3, stride=1, padding=1, degree=3, groups=1, dilation=1,
                 dropout: float = 0.0, ndim: int = 2., dim_reduction: float = 4, min_internal: int = 16,
                 pregate: bool = False, **norm_kwargs):
        super(MoEBottleNeckKAGNConvND, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_dim
        self.input_size = input_dim
        self.k = k
        self.groups = groups
        self.base_activation = nn.SiLU()
        # instantiate experts
        inner_dim = int(max((input_dim // groups) / dim_reduction,
                            (output_dim // groups) / dim_reduction))
        if inner_dim < min_internal:
            self.inner_dim = min(min_internal, input_dim // groups, output_dim // groups)
        else:
            self.inner_dim = inner_dim

        self.experts = KAGNMoE(num_experts, conv_w_fun, self.inner_dim * groups, self.inner_dim * groups,
                               degree, kernel_size=kernel_size,
                               groups=groups, padding=padding, stride=stride, dilation=dilation,
                               dropout=dropout, ndim=ndim, noisy_gating=noisy_gating, k=k, pregate=pregate)
        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
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

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.inner_proj:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.out_proj:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        self.w_gate = nn.Parameter(torch.zeros(self.inner_dim * groups, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.inner_dim * groups, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        if ndim == 1:
            self.avgpool = nn.AdaptiveAvgPool1d((1,))
            self.conv_dims = 1
        elif ndim == 2:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.conv_dims = 2
        elif ndim == 3:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.conv_dims = 3

        assert (self.k <= self.num_experts)

    def forward_moe_base(self, x, group_index):

        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))

        return basis

    def forward_moe_inner(self, x, group_index):

        y = self.inner_proj[group_index](x)

        return y

    def forward_moe_outer(self, x, basis, group_index):

        y = self.out_proj[group_index](x)

        y = self.base_activation(self.layer_norm[group_index](y + basis))

        return y

    def forward(self, x, train=True, loss_coef=1e-2):

        split_x = torch.split(x, self.input_size // self.groups, dim=1)
        output = []
        bases = []
        for group_ind, _x in enumerate(split_x):
            base = self.forward_moe_base(_x, group_ind)
            bases.append(base)
            x = self.forward_moe_inner(_x, group_ind)
            output.append(x.clone())

        y, loss = self.experts.forward(torch.cat(output, dim=1), loss_coef=loss_coef)
        output = []
        for group_ind, (_xb, _xe) in enumerate(zip(bases, torch.split(y, self.inner_dim, dim=1))):
            x = self.forward_moe_outer(_xe, _xb, group_index=group_ind)
            output.append(x.clone())
        y = torch.cat(output, dim=1)
        return y, loss


class MoEBottleNeckKAGNConv3DLayer(MoEBottleNeckKAGNConvND):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm3d, dim_reduction: float = 4, num_experts=16,
                 noisy_gating=True, k=4, pregate: bool = False, **norm_kwargs):
        super(MoEBottleNeckKAGNConv3DLayer, self).__init__(nn.Conv3d, conv3d, norm_layer,
                                                           input_dim, output_dim,
                                                           degree=degree, kernel_size=kernel_size,
                                                           dim_reduction=dim_reduction,
                                                           groups=groups, padding=padding, stride=stride,
                                                           dilation=dilation, num_experts=num_experts,
                                                           noisy_gating=noisy_gating, k=k, pregate=pregate,
                                                           ndim=3, dropout=dropout, **norm_kwargs)


class MoEBottleNeckKAGNConv2DLayer(MoEBottleNeckKAGNConvND):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, num_experts=8,
                 noisy_gating=True, k=2, dim_reduction: float = 4, pregate: bool = False, **norm_kwargs):
        super(MoEBottleNeckKAGNConv2DLayer, self).__init__(nn.Conv2d, conv2d, norm_layer,
                                                           input_dim, output_dim,
                                                           degree=degree, kernel_size=kernel_size,
                                                           dim_reduction=dim_reduction,
                                                           groups=groups, padding=padding, stride=stride,
                                                           dilation=dilation, num_experts=num_experts,
                                                           noisy_gating=noisy_gating, k=k, pregate=pregate,
                                                           ndim=2, dropout=dropout, **norm_kwargs)


class MoEBottleNeckKAGNConv1DLayer(MoEBottleNeckKAGNConvND):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm1d, num_experts=16,
                 noisy_gating=True, k=4, dim_reduction: float = 4, pregate: bool = False, **norm_kwargs):
        super(MoEBottleNeckKAGNConv1DLayer, self).__init__(nn.Conv1d, conv1d, norm_layer,
                                                           input_dim, output_dim,
                                                           degree=degree, kernel_size=kernel_size,
                                                           dim_reduction=dim_reduction,
                                                           groups=groups, padding=padding, stride=stride,
                                                           dilation=dilation, num_experts=num_experts,
                                                           noisy_gating=noisy_gating, k=k, pregate=pregate,
                                                           ndim=1, dropout=dropout, **norm_kwargs)
