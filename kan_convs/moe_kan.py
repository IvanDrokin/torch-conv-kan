# The code is based on the Yeonwoo Sung's implementation:
# https://github.com/YeonwooSung/Pytorch_mixture-of-experts/blob/main/moe.py
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .fast_kan_conv import FastKANConv1DLayer, FastKANConv2DLayer, FastKANConv3DLayer
from .kacn_conv import KACNConv1DLayer, KACNConv2DLayer, KACNConv3DLayer
from .kagn_conv import KAGNConv1DLayer, KAGNConv2DLayer, KAGNConv3DLayer
from .kaln_conv import KALNConv1DLayer, KALNConv2DLayer, KALNConv3DLayer
from .kan_conv import KANConv1DLayer, KANConv2DLayer, KANConv3DLayer
from .wav_kan import WavKANConv1DLayer, WavKANConv2DLayer, WavKANConv3DLayer
from .kagn_bottleneck_conv import BottleNeckKAGNConv1DLayer, BottleNeckKAGNConv2DLayer, BottleNeckKAGNConv3DLayer

from .moe_utils import SparseDispatcher


class MoEKANConvBase(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, conv_class, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKANConvBase, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([conv_class(input_size, self.output_size, kernel_size=kernel_size,
                                                 stride=stride, padding=padding, **kan_kwargs) for _ in
                                      range(num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        if conv_class in [KANConv1DLayer, FastKANConv1DLayer, KALNConv1DLayer, KACNConv1DLayer, KAGNConv1DLayer,
                          WavKANConv1DLayer, BottleNeckKAGNConv1DLayer]:
            self.avgpool = nn.AdaptiveAvgPool1d((1,))
            self.conv_dims = 1
        elif conv_class in [KANConv2DLayer, FastKANConv2DLayer, KALNConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer,
                            WavKANConv2DLayer, BottleNeckKAGNConv2DLayer]:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.conv_dims = 2
        elif conv_class in [KANConv3DLayer, FastKANConv3DLayer, KALNConv3DLayer, KACNConv3DLayer, KAGNConv3DLayer,
                            WavKANConv3DLayer, BottleNeckKAGNConv3DLayer]:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.conv_dims = 3

        for i in range(1, num_experts):
            self.experts[i].load_state_dict(self.experts[0].state_dict())

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

    def forward(self, x, train=True, loss_coef=1e-02):
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


class MoEKALNConv2DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKALNConv2DLayer, self).__init__(KALNConv2DLayer, input_size, output_size, num_experts=num_experts,
                                                 noisy_gating=noisy_gating, k=k,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEKALNConv1DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKALNConv1DLayer, self).__init__(KALNConv1DLayer, input_size, output_size, num_experts=num_experts,
                                                 noisy_gating=noisy_gating, k=k,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEKALNConv3DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKALNConv3DLayer, self).__init__(KALNConv3DLayer, input_size, output_size, num_experts=num_experts,
                                                 noisy_gating=noisy_gating, k=k,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEKANConv2DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKANConv2DLayer, self).__init__(KANConv2DLayer, input_size, output_size, num_experts=num_experts,
                                                noisy_gating=noisy_gating, k=k,
                                                kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEKAGNConv1DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKAGNConv1DLayer, self).__init__(KAGNConv1DLayer, input_size, output_size, num_experts=num_experts,
                                                 noisy_gating=noisy_gating, k=k,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEKAGNConv3DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKAGNConv3DLayer, self).__init__(KAGNConv3DLayer, input_size, output_size, num_experts=num_experts,
                                                 noisy_gating=noisy_gating, k=k,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEKAGNConv2DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKAGNConv2DLayer, self).__init__(KAGNConv2DLayer, input_size, output_size, num_experts=num_experts,
                                                 noisy_gating=noisy_gating, k=k,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEKANConv1DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKANConv1DLayer, self).__init__(KANConv1DLayer, input_size, output_size, num_experts=num_experts,
                                                noisy_gating=noisy_gating, k=k,
                                                kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEKANConv3DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKANConv3DLayer, self).__init__(KANConv3DLayer, input_size, output_size, num_experts=num_experts,
                                                noisy_gating=noisy_gating, k=k,
                                                kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEFastKANConv2DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEFastKANConv2DLayer, self).__init__(FastKANConv2DLayer, input_size, output_size,
                                                    num_experts=num_experts,
                                                    noisy_gating=noisy_gating, k=k,
                                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                                    **kan_kwargs)


class MoEFastKANConv1DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEFastKANConv1DLayer, self).__init__(FastKANConv1DLayer, input_size, output_size,
                                                    num_experts=num_experts,
                                                    noisy_gating=noisy_gating, k=k,
                                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                                    **kan_kwargs)


class MoEFastKANConv3DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEFastKANConv3DLayer, self).__init__(FastKANConv3DLayer, input_size, output_size,
                                                    num_experts=num_experts,
                                                    noisy_gating=noisy_gating, k=k,
                                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                                    **kan_kwargs)


class MoEKACNConv2DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKACNConv2DLayer, self).__init__(KACNConv2DLayer, input_size, output_size, num_experts=num_experts,
                                                 noisy_gating=noisy_gating, k=k,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEKACNConv1DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKACNConv1DLayer, self).__init__(KACNConv1DLayer, input_size, output_size, num_experts=num_experts,
                                                 noisy_gating=noisy_gating, k=k,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEKACNConv3DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEKACNConv3DLayer, self).__init__(KACNConv3DLayer, input_size, output_size, num_experts=num_experts,
                                                 noisy_gating=noisy_gating, k=k,
                                                 kernel_size=kernel_size, stride=stride, padding=padding, **kan_kwargs)


class MoEWavKANConv2DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEWavKANConv2DLayer, self).__init__(WavKANConv2DLayer, input_size, output_size, num_experts=num_experts,
                                                   noisy_gating=noisy_gating, k=k,
                                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                                   **kan_kwargs)


class MoEWavKANConv1DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEWavKANConv1DLayer, self).__init__(WavKANConv1DLayer, input_size, output_size, num_experts=num_experts,
                                                   noisy_gating=noisy_gating, k=k,
                                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                                   **kan_kwargs)


class MoEWavKANConv3DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEWavKANConv3DLayer, self).__init__(WavKANConv3DLayer, input_size, output_size, num_experts=num_experts,
                                                   noisy_gating=noisy_gating, k=k,
                                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                                   **kan_kwargs)


class MoEFullBottleneckKAGNConv1DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEFullBottleneckKAGNConv1DLayer, self).__init__(BottleNeckKAGNConv1DLayer, input_size, output_size,
                                                               num_experts=num_experts,
                                                               noisy_gating=noisy_gating, k=k,
                                                               kernel_size=kernel_size, stride=stride, padding=padding,
                                                               **kan_kwargs)


class MoEFullBottleneckKAGNConv3DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEFullBottleneckKAGNConv3DLayer, self).__init__(BottleNeckKAGNConv3DLayer, input_size, output_size,
                                                               num_experts=num_experts,
                                                               noisy_gating=noisy_gating, k=k,
                                                               kernel_size=kernel_size, stride=stride, padding=padding,
                                                               **kan_kwargs)


class MoEFullBottleneckKAGNConv2DLayer(MoEKANConvBase):
    def __init__(self, input_size, output_size, num_experts=16, noisy_gating=True, k=4,
                 kernel_size=3, stride=1, padding=1, **kan_kwargs):
        super(MoEFullBottleneckKAGNConv2DLayer, self).__init__(BottleNeckKAGNConv2DLayer, input_size, output_size,
                                                               num_experts=num_experts,
                                                               noisy_gating=noisy_gating, k=k,
                                                               kernel_size=kernel_size, stride=stride, padding=padding,
                                                               **kan_kwargs)
