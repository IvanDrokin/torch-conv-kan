# focal loss taken from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 label_smoothing: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100,
                 num_classes: int = 1000):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    @staticmethod
    def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
        """
        if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method

        """
        assert 0 <= smoothing < 1
        confidence = 1.0 - smoothing
        label_shape = torch.Size((true_labels.size(0), classes))
        with torch.no_grad():
            true_dist = torch.empty(size=label_shape, device=true_labels.device)
            true_dist.fill_(smoothing / (classes - 1))
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
        return true_dist

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]

        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        if self.label_smoothing > 0:
            y = self.smooth_one_hot(y, self.num_classes, self.label_smoothing)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class Dice(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super(Dice, self).__init__()
        self.smooth = smooth

    def _channel_with_dice(self, inputs, targets):
        dice = 0.
        for i in range(inputs.shape[1]):
            # flatten label and prediction tensors

            c_inp = inputs[:, i].view(inputs.shape[0], -1)
            c_tgt = targets[:, i].view(inputs.shape[0], -1)

            intersection = torch.sum(c_inp * c_tgt, dim=1)
            dice += (2. * intersection + self.smooth) / (c_inp.sum(dim=1) + c_tgt.sum(dim=1) + self.smooth)
        dice = torch.mean(dice) / float(inputs.shape[1])
        return dice

    def forward(self, inputs, targets):
        if inputs.shape[1] > 1:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = F.sigmoid(inputs)
        dice = self._channel_with_dice(inputs, targets)
        return dice


class DiceLoss(Dice):
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__(smooth=smooth)

    def forward(self, inputs, targets):
        return 1. - super(DiceLoss, self).forward(inputs, targets)


class DiceLossWithBCE(DiceLoss):
    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5):
        super(DiceLossWithBCE, self).__init__(smooth=smooth)
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer

        dice = super(DiceLossWithBCE, self).forward(inputs, targets)
        inputs = F.sigmoid(inputs)

        dice = self.bce_weight * F.binary_cross_entropy(inputs, targets) + dice
        return dice


class DiceLossWithFocal(DiceLoss):
    def __init__(self,
                 focal_weight: float = 0.5,
                 smooth: float = 1.0,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 label_smoothing: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100,
                 num_classes: int = 1000):
        super(DiceLossWithFocal, self).__init__(smooth=smooth)
        self.focal_weight = focal_weight
        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
            reduction=reduction,
            ignore_index=ignore_index,
            num_classes=num_classes)

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        focal = self.focal_weight * self.focal(inputs, targets)
        dice = super(DiceLossWithFocal, self).forward(inputs, targets) + focal
        return dice


class TverskyLoss(nn.Module):
    def __init__(self, smooth: float = 1, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _channel_tversky(self, inputs, targets):
        tversky = 0.
        for i in range(inputs.shape[1]):
            c_inp = inputs[:, i].view(inputs.shape[0], -1)
            c_tgt = targets[:, i].view(inputs.shape[0], -1)

            true_pos = (c_inp * c_tgt).sum(dim=1)
            false_pos = ((1 - c_tgt) * c_inp).sum(dim=1)
            false_net = (c_tgt * (1 - c_inp)).sum(dim=1)

            c_tversky = (true_pos + self.smooth) / (
                        true_pos + self.alpha * false_pos + self.beta * false_net + self.smooth)
            tversky += (1 - c_tversky) ** self.gamma
        tversky = torch.mean(tversky) / float(inputs.shape[1])
        return tversky

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer

        if inputs.shape[1] > 1:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = F.sigmoid(inputs)

        return self._channel_tversky(inputs, targets)
