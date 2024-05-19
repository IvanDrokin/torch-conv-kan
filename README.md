# TorchConv KAN: A Convolutional Kolmogorov-Arnold Networks Collection

This project introduces and demonstrates the training, validation, and quantization of the Convolutional KAN model using PyTorch with CUDA acceleration. The `torch-conv-kan` evaluates performance on the MNIST and CIFAR datasets.

## Project Status: Under Development
### Updates

- ✅ [2024/05/13] Convolutional KALN layers are available

- ✅ [2024/05/14] Convolutional KAN and Fast KAN layers are available
  
- ✅ [2024/05/15] Convolutional ChebyKAN are available now. MNIST, CIFAR10 and CIFAR100 benchmarks are added.
  
- ✅ [2024/05/19] ResNet-like, U-net like and MoE-based (don't ask why=)) models released with accelerate-based training code.

### TODO list
- [ ] Expand model zoo
- [ ] Add more benchmarks
- [ ] Perform hyperparameters optimisation

---

## Introducing Convolutional KAN layers

- The `KANConv1DLayer`, `KANConv2DLayer`, `KANConv3DLayer` classes represents a convolutional layers based on Kolmogorov Arnold Network, introduced in [1]. Baseline model implemented in `models/baselines/conv_kan_baseline.py`.

- The `KALNConv1DLayer`, `KALNConv2DLayer`, `KALNConv3DLayer` classes represents a convolutional layers based on Kolmogorov Arnold Legendre Network, introduced in [2]. Baseline model implemented in `models/baselines/conv_kaln_baseline.py`.

- The `FastKANConv1DLayer`, `FastKANConv2DLayer`, `FastKANConv3DLayer` classes represents a convolutional layers based on Fast Kolmogorov Arnold Network, introduced in [3]. Baseline model implemented in `models/baselines/fast_conv_kan_baseline.py`.

- The `KACNConv1DLayer`, `KACNConv1DLayer`, `KACNConv1DLayer` classes represents a convolutional layers based on Kolmogorov Arnold Network with Chebyshev polynomials insted of B-splines, introduced in [4]. Baseline model implemented in `models/baselines/conv_kacn_baseline.py`.

## Model Zoo

### ResKANets

We introduce ResKANets - an ResNet-like model with KAN convolutions instead of regular one. Main class ```ResKANet``` could be found ```models/reskanet.py```. Our implementation supports Basic and Bottleneck blocks with KAN, Fast KAN, KALN and KACN convolutional layers.

After 75 training epochs on CIFAR10 ResKANet 18 with Kolmogorov Arnold Legendre convolutions achieved 84.17% accuracy and 0.985 AUC (OVO).

After 75 training epochs on Tiny Imagenet ResKANet 18 with Kolmogorov Arnold Legendre convolutions achieved 28.62% accuracy, 55.49% top-5 accuracy, and 0.932 AUC (OVO).

Please, take into account that this is preliminary results and more experiments are in progress right now.

### UKANet

We introduce UKANets - an U-net like model with KAN convolutions instead of regular one, based on resnet blocks. Main class ```UKANet``` could be found ```models/ukanet.py```. Our implementation supports Basic and Bottleneck blocks with KAN, Fast KAN, KALN and KACN convolutional layers.

## Performance Metrics

Baseline models were chosen to be simple networks with 4 and 8 convolutional layers. To reduce dimensionality, convolutions with dilation=2 were used. In the 4-layer model, the second and third convolutions had dilation=2, while in the 8-layer model, the second, third, and sixth convolutions had dilation=2.

The number of channels in the convolutions was the same for all models:

For 4 layers: 32, 64, 128, 512
For 8 layers: 2, 64, 128, 512, 1024, 1024, 1024, 1024
After the convolutions, Global Average Pooling was applied, followed by a linear output layer.

In the case of classic convolutions, a traditional structure was used: convolution - batch normalization - ReLU.

All experiments were conducted on an NVIDIA RTX 3090 with identical training parameters. For more details, please refer to the file ```mnist_conv.py```.

### MNIST
Accuracy on the training and validation MNIST datasets, 4 convolutional layer models
![MNIST train and validation accuracy for 4 layer models](./assets/MNIST.png)

Accuracy on the training and validation MNIST datasets, 8 convolutional layer models
![MNIST train and validation accuracy for 8 layer models](./assets/MNIST_8.png)

| Model                       | Val. Accuracy | Parameters | Eval Time, s |
|-----------------------------|---------------|------------|--------------|
| SimpleConv, 4 layers        | 98.94         | 178122     | 0.4017       |
| SimpleKANConv, 4 layers     | **99.48**     | 1542199    | 1.7437       |
| SimpleFastKANConv, 4 layers | 98.29         | 1542186    | 0.4558       |
| SimpleKALNConv, 4 layers    | 99.40         | 859050     | 0.5085       |
| SimpleKACNConv, 4 layers    | 97.54         | 689738     | 0.4225       |
| SimpleConv, 8 layers        | 99.37         | 42151850   | 1.7582       |
| SimpleKANConv, 8 layers     | 99.39         | 75865159   | 5.7914       |
| SimpleFastKANConv, 8 layers | 99.09         | 75865130   | 2.4105       |
| SimpleKALNConv, 8 layers    | 99.36         | 42151850   | 1.7582       |
| SimpleKACNConv, 8 layers    | 99.22         | 33733194   | 0.8509       |

### CIFAR10
Accuracy on the training and validation CIFAR10 datasets, 4 convolutional layer models
![CIFAR10 train and validation accuracy for 4 layer models](./assets/CIFAR10.png)

Accuracy on the training and validation CIFAR10 datasets, 4 convolutional layer models
![CIFAR10 train and validation accuracy for 8 layer models](./assets/CIFAR10_8.png)

|           Model           |Val. Accuracy|Parameters|Eval Time, s|
|---------------------------|-------------|----------|------------|
|    SimpleConv, 4 layers   |    69.69    |  178698  |   0.5481   |
|  SimpleKANConv, 4 layers  |    59.37    |  1547383 |   2.2969   |
|SimpleFastKANConv, 4 layers|    57.39    |  1547370 |   0.6169   |
|  SimpleKALNConv, 4 layers |    63.86    |  861930  |   0.6824   |
|  SimpleKACNConv, 4 layers |    57.03    |  692042  |   0.5853   |
|    SimpleConv, 8 layers   |  **76.08**  |  8453642 |   0.5647   |
|  SimpleKANConv, 8 layers  |    64.62    | 75870343 |   7.0329   |
|SimpleFastKANConv, 8 layers|    57.21    | 75870314 |   2.7723   |
|  SimpleKALNConv, 8 layers |    66.77    | 42154730 |   1.9057   |
|  SimpleKACNConv, 8 layers |    64.98    | 33735498 |   0.9085   |

### CIFAR100
Accuracy on the training and validation CIFAR100 datasets, 4 convolutional layer models
![CIFAR100 train and validation accuracy for 4 layer models](./assets/CIFAR100.png)

Accuracy on the training and validation CIFAR100 datasets, 8 convolutional layer models
![CIFAR100 train and validation accuracy for 8 layer models](./assets/CIFAR100_8.png)

|           Model           |Val. Accuracy|Parameters|Eval Time, s|
|---------------------------|-------------|----------|------------|
|    SimpleConv, 4 layers   |    38.99    |  224868  |   0.5533   |
|  SimpleKANConv, 4 layers  |    20.53    |  1593553 |   2.3098   |
|SimpleFastKANConv, 4 layers|    32.48    |  1593540 |   0.6175   |
|  SimpleKALNConv, 4 layers |    22.36    |  908100  |   0.6540   |
|  SimpleKACNConv, 4 layers |    34.17    |  738212  |   0.5820   |
|    SimpleConv, 8 layers   |  **43.32**  |  8545892 |   0.5663   |
|  SimpleKANConv, 8 layers  |    23.22    | 75962593 |   7.0452   |
|SimpleFastKANConv, 8 layers|    23.75    | 75962564 |   2.7713   |
|  SimpleKALNConv, 8 layers |    18.90    | 42246980 |   1.8955   |
|  SimpleKACNConv, 8 layers |     0.98    | 33827748 |   0.9093   |


## Discussion

First and foremost, it should be noted that the results obtained are preliminary. The model architecture has not been thoroughly explored and represents only two of many possible design variants.

Nevertheless, the experiments indicate that Kolmogorov-Arnold convolutional networks outperform the classical convolutional architecture on the MNIST dataset, but significantly underperform on CIFAR-10 and CIFAR-100 in terms of quality. The ChebyKAN-based convolution encounters stability issues during training, necessitating further investigation.

As a next step, I plan to search for a suitable architecture for KAN convolutions that can achieve acceptable quality on CIFAR-10/100 and attempt to scale these models to more complex datasets.

## Prerequisites

Ensure you have the following installed on your system:

- Python (version 3.9 or higher)
- CUDA Toolkit (corresponding to your PyTorch installation's CUDA version)
- cuDNN (compatible with your installed CUDA Toolkit)

## Usage

Below is an example of a simple model based on KAN convolutions:
```python
import torch
import torch.nn as nn

from kan_convs import KANConv2DLayer


class SimpleConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 3,
            groups: int = 1):
        super(SimpleConvKAN, self).__init__()

        self.layers = nn.Sequential(
            KANConv2DLayer(input_channels, layer_sizes[0], spline_order, kernel_size=3, groups=1, padding=1, stride=1,
                           dilation=1),
            KANConv2DLayer(layer_sizes[0], layer_sizes[1], spline_order, kernel_size=3, groups=groups, padding=1,
                           stride=2, dilation=1),
            KANConv2DLayer(layer_sizes[1], layer_sizes[2], spline_order, kernel_size=3, groups=groups, padding=1,
                           stride=2, dilation=1),
            KANConv2DLayer(layer_sizes[2], layer_sizes[3], spline_order, kernel_size=3, groups=groups, padding=1,
                           stride=1, dilation=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output = nn.Linear(layer_sizes[3], num_classes)

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.output(x)
        return x
```

To run the training and testing of the baseline models on the MNIST, CIFAR-10, and CIFAR-100 datasets, execute the following line of code:

```python mnist_conv.py```

This script will train baseline models on MNIST, CIFAR10 or CIFAR100, validate them, quantise and log performance metrics.

### Accelerate-based training

We introduce training code with Accelerate, Hydra configs and Wandb logging. 

#### 1. Clone the Repository

Clone the `torch-conv-kan` repository and set up the project environment:

```bash
git clone https://github.com/IvanDrokin/torch-conv-kan.git
cd torch-conv-kan
pip install -r requirements.txt
```

#### 2. Configure Weights & Biases (wandb)

To monitor experiments and model performance with wandb:

1. **Set Up wandb Account:**

- Sign up or log in at [Weights & Biases](https://wandb.ai).
- Locate your API key in your account settings.

2. **Initialize wandb in Your Project:**

Before running the training script, initialize wandb:

```python
wandb login
```

Enter your API key when prompted to link your script executions to your wandb account.

3. **Adjust the Entity Name in `configs/cifar10-reskanet.yaml` or `configs/tiny-imagenet-reskanet.yaml` to Your Username or Team Name**

#### Run

Update any parameters in configs and run

```python
accelerate launch cifar.py
```

This script trains the model, validates it, and logs performance metrics using wandb on CIFAR10 dataset.

```python
accelerate launch tiny_imagenet.py
```

This script trains the model, validates it, and logs performance metrics using wandb on Tiny Imagenet dataset.

### Using your own dataset or model

If you would like to use your own dataset, please follow this steps:

1. Copy ```tiny_imagenet.py``` and modify ```get_data()``` method. If basic implementation of Classification dataset is not suitable for your data - please, upgrade it or write your own one.
2. Replace ```model = reskalnet_18x64p(...)``` with your own one if necessary.
3. Create config yaml in ```config``` forlders, following provided templates.
4. Run ```accelerate launch your_script.py```

## Cite this Project

If you use this project in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```bibtex
@misc{torch-conv-kan,
  author = {Ivan Drokin},
  title = {Torch Conv KAN},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/IvanDrokin/torch-conv-kan}}
}
```

## Contributions

Contributions are welcome. Please raise issues as necessary.

## Acknowledgements

This repository based on [TorchKAN](https://github.com/1ssb/torchkan/), [FastKAN](https://github.com/ZiyaoLi/fast-kan), and [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN), and we would like to say thanks for their open research and exploration.


## References

- [1] Ziming Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024, arXiv. https://arxiv.org/abs/2404.19756
- [2] https://github.com/1ssb/torchkan
- [3] https://github.com/ZiyaoLi/fast-kan
- [4] https://github.com/SynodicMonth/ChebyKAN  
- [5] https://github.com/KindXiaoming/pykan
- [6] https://github.com/Blealtan/efficient-kan
