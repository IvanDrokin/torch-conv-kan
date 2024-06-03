from typing import List

import torch
import torch.nn as nn


class MatryoshkaHead(nn.Module):
    """
    https://github.com/RAIVNLab/MRL/blob/main/MRL.py
    Matryoshka linear layer for Matryoshka Representation Learning
    """
    def __init__(self, nesting_list: List, num_classes=1000, efficient=False, **kwargs):
        super(MatryoshkaHead, self).__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes  # Number of classes for classification
        self.efficient = efficient
        if self.efficient:
            setattr(self, f"nesting_classifier_{0}", nn.Linear(nesting_list[-1], self.num_classes, **kwargs))
        else:
            for i, num_feat in enumerate(self.nesting_list):
                setattr(self, f"nesting_classifier_{i}", nn.Linear(num_feat, self.num_classes, **kwargs))

    def reset_parameters(self):
        if self.efficient:
            self.nesting_classifier_0.reset_parameters()
        else:
            for i in range(len(self.nesting_list)):
                getattr(self, f"nesting_classifier_{i}").reset_parameters()

    def forward(self, x):
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.efficient:
                if self.nesting_classifier_0.bias is None:
                    nesting_logits += (
                        torch.matmul(x[:, :num_feat], (self.nesting_classifier_0.weight[:, :num_feat]).t()),)
                else:
                    nesting_logits += (torch.matmul(x[:, :num_feat], (
                        self.nesting_classifier_0.weight[:, :num_feat]).t()) + self.nesting_classifier_0.bias,)
            else:
                nesting_logits += (getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),)

        return nesting_logits
