# Regularization study, CIFAR100

## Isolated experiments

Baseline model was chosen to be simple networks with 8 convolutional layers with Gram polynomials as basis functions. To reduce dimensionality, convolutions with dilation=2 were used: the second, third, and sixth convolutions had dilation=2.

We explore two sets of convolutional layers filters:

**Slim**: 16, 32, 64, 128, 256, 256, 512, 512

**Wide**: 32, 64, 128, 256, 512, 512, 1024, 1024

After the convolutions, Global Average Pooling was applied, followed by a linear output layer. In these experiments, a dropout after Global Pooling hasn't been used.

Augmentations have been used:
```python
from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy

transform_train = v2.Compose([

            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                             v2.AutoAugment(AutoAugmentPolicy.IMAGENET),
                             v2.AutoAugment(AutoAugmentPolicy.SVHN),
                             v2.TrivialAugmentWide()]),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
```

All experiments were conducted on an NVIDIA RTX 3090 with identical training parameters.

As part of this research, we aim to find answers to the following questions:

- What is the impact of L1 and L2 activation penalties of ConvKAN layers on the model?

- What is the impact of L1 regularization of the weights of ConvKAN layers on the model?

- What is the impact of the placement of dropout within a ConvKAN layer on the model? Essentially, there are three placements: before the layer (Full), before the calculation of the polynomial basis (Poly), and before applying the weights to the calculated polynomials (Degree).

- Since KAN models are supposed to be more robust to noise in the data, can we replace the Dropout layer with additive Gaussian noise as regularization technique?

### Noise Injection

Here is the code for ```NoiseInjection``` layer:

```python
import torch
import torch.nn as nn


class NoiseInjection(nn.Module):
    def __init__(self, p: float = 0.0, alpha: float = 0.05):
        super(NoiseInjection, self).__init__()
        self.p = p
        self.alpha = alpha

    def get_noise(self, x):
        std = torch.std(x, dim=1, keepdim=True)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * std
        return noise

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask = (mask < self.p).float() * 1
            x = x + self.alpha * mask * self.get_noise(x)
            return x
        return x
```

### Slim model
#### L1 and L2 activation penalties

|Train Accuracy|Val. Accuracy|   Regularization   |
|--------------|-------------|--------------------|
|     73.73    |    61.30    |        None        |
|     74.49    |    61.20    |L1 Activation, 1e-08|
|     74.64    |    61.41    |L1 Activation, 1e-07|
|     74.11    |  **61.74**  |L1 Activation, 1e-06|
|     73.85    |    61.36    |L2 Activation, 1e-08|
|     73.96    |    61.55    |L2 Activation, 1e-07|
|     73.10    |    60.29    |L2 Activation, 1e-06|

Here is training and validation accuracy during training, L1 activation penalty.
![l1_activation_penalties](../assets/reg/slim_l1_activation.png)

Here is training and validation accuracy during training, L2 activation penalty.
![l2_activation_penalties](../assets/reg/slim_l2_activation.png)

#### L1 weight regularization

|Train Accuracy|Val. Accuracy| Regularization |
|--------------|-------------|----------------|
|     73.73    |    61.30    |      None      |
|     74.31    |  **61.39**  |L1 Weight, 1e-08|
|     73.48    |    61.08    |L1 Weight, 1e-07|
|     73.60    |    60.96    |L1 Weight, 1e-06|

Here is training and validation accuracy during training, L1 weight regularization.
![l2_activation_penalties](../assets/reg/slim_l1_weight.png)

#### Dropout

|Train Accuracy|Val. Accuracy|   Regularization   |
|--------------|-------------|--------------------|
|     73.73    |    61.30    |        None        |
|     68.69    |    60.98    | Poly Dropout, 0.05 |
|     60.98    |  **63.64**  | Full Dropout, 0.05 |
|     70.12    |    62.91    |Degree Dropout, 0.05|
|     65.43    |    58.63    | Poly Dropout, 0.15 |
|     43.77    |    56.10    | Full Dropout, 0.15 |
|     64.37    |    62.28    |Degree Dropout, 0.15|
|     61.93    |    51.27    | Poly Dropout, 0.25 |
|     32.99    |    47.65    | Full Dropout, 0.25 |
|     60.09    |    60.69    |Degree Dropout, 0.25|

Here is training and validation accuracy during training, dropouts.
![l2_activation_penalties](../assets/reg/slim_dropout.png)

#### Noise Injection

|Train Accuracy|Val. Accuracy|       Regularization       |
|--------------|-------------|----------------------------|
|     73.73    |    61.30    |            None            |
|     70.42    |    62.06    | Poly Noise Injection, 0.05 |
|     59.75    |  **62.63**  | Full Noise Injection, 0.05 |
|     69.45    |    62.40    |Degree Noise Injection, 0.05|
|     65.75    |    57.64    | Poly Noise Injection, 0.15 |
|     43.78    |    56.15    | Full Noise Injection, 0.15 |
|     64.52    |    61.78    |Degree Noise Injection, 0.15|
|     63.38    |    53.14    | Poly Noise Injection, 0.25 |
|     33.54    |    48.24    | Full Noise Injection, 0.25 |
|     60.17    |    60.98    |Degree Noise Injection, 0.25|

Here is training and validation accuracy during training, noise injection.
![l2_activation_penalties](../assets/reg/slim_noisy_injection.png)

### Wide model
#### L1 and L2 activation penalties

|Train Accuracy|Val. Accuracy|   Regularization   |
|--------------|-------------|--------------------|
|     86.88    |    67.04    |        None        |
|     86.69    |    66.41    |L1 Activation, 1e-08|
|     86.60    |    66.88    |L1 Activation, 1e-07|
|     86.73    |    67.09    |L1 Activation, 1e-06|
|     86.61    |  **67.49**  |L2 Activation, 1e-08|
|     86.78    |    66.81    |L2 Activation, 1e-07|
|     86.99    |    66.30    |L2 Activation, 1e-06|

Here is training and validation accuracy during training, L1 activation penalty.
![l1_activation_penalties](../assets/reg/wide_l1_activation.png)

Here is training and validation accuracy during training, L2 activation penalty.
![l2_activation_penalties](../assets/reg/wide_l2_activation.png)

#### L1 weight regularization
|Train Accuracy|Val. Accuracy| Regularization |
|--------------|-------------|----------------|
|     86.88    |    67.04    |      None      |
|     86.41    |    66.79    |L1 Weight, 1e-08|
|     86.68    |  **67.23**  |L1 Weight, 1e-07|
|     87.10    |    66.83    |L1 Weight, 1e-06|

Here is training and validation accuracy during training, L1 weight regularization.
![l2_activation_penalties](../assets/reg/wide_l1_weight.png)

#### Dropout
|Train Accuracy|Val. Accuracy|   Regularization   |
|--------------|-------------|--------------------|
|     86.88    |    67.04    |        None        |
|     84.97    |    66.83    | Poly Dropout, 0.05 |
|     79.53    |  **68.93**  | Full Dropout, 0.05 |
|     85.01    |    67.75    |Degree Dropout, 0.05|
|     82.07    |    62.26    | Poly Dropout, 0.15 |
|     63.19    |    66.38    | Full Dropout, 0.15 |
|     81.24    |    68.70    |Degree Dropout, 0.15|
|     80.22    |    59.70    | Poly Dropout, 0.25 |
|     49.49    |    59.56    | Full Dropout, 0.25 |
|     77.46    |    67.85    |Degree Dropout, 0.25|

Here is training and validation accuracy during training, dropouts.
![l2_activation_penalties](../assets/reg/wide_dropout.png)

#### Noise Injection
|Train Accuracy|Val. Accuracy|       Regularization       |
|--------------|-------------|----------------------------|
|     86.88    |    67.04    |            None            |
|     84.65    |    67.11    | Poly Noise Injection, 0.05 |
|     79.71    |  **69.18**  | Full Noise Injection, 0.05 |
|     85.00    |    68.43    |Degree Noise Injection, 0.05|
|     82.78    |    63.82    | Poly Noise Injection, 0.15 |
|     63.30    |    66.07    | Full Noise Injection, 0.15 |
|     81.33    |    68.23    |Degree Noise Injection, 0.15|
|     80.74    |    59.96    | Poly Noise Injection, 0.25 |
|     49.58    |    59.70    | Full Noise Injection, 0.25 |
|     77.26    |    67.82    |Degree Noise Injection, 0.25|

Here is training and validation accuracy during training, noise injection.
![l2_activation_penalties](../assets/reg/wide_noisy_injection.png)

## Discussion

Based on the conducted experiments, the following conclusions can be drawn:

 - It seems that Full Dropout and Full Noise Injection are the best options for regularization, helping to combat model overfitting.
 - L1/L2 activation penalties, as well as L1 weight penalty, slightly improve the situation, but not significantly.
 - In a wider model, the impact of regularization is greater.


## Next steps

 - Scaling study: how depth, width and degree impack model performance
 - Hyper parameters tuning