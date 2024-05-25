# TorchConv KAN: A Convolutional Kolmogorov-Arnold Networks Collection

This project introduces and demonstrates the training, validation, and quantization of the Convolutional KAN model using PyTorch with CUDA acceleration. The `torch-conv-kan` evaluates performance on the MNIST and CIFAR datasets.

## Project Status: Under Development
### Updates

- ✅ [2024/05/13] Convolutional KALN layers are available

- ✅ [2024/05/14] Convolutional KAN and Fast KAN layers are available
  
- ✅ [2024/05/15] Convolutional ChebyKAN are available now. MNIST, CIFAR10 and CIFAR100 benchmarks are added.
  
- ✅ [2024/05/19] ResNet-like, U-net like and MoE-based (don't ask why=)) models released with accelerate-based training code.
  
- ✅ [2024/05/21] VGG-like and DenseNet-like models released! Gram KAN convolutional layers added.
  
- ✅ [2024/05/23] WavKAN convolutional layers added. Fixed a bug with output hook in ```trainer.py```.
  
- ✅ [2024/05/25] U2-net like models added. Fixed a memory leak in ```trainer.py```.

### TODO list and next steps
- [ ] Expand model zoo 
- [ ] Add more benchmarks
- [ ] Perform hyperparameters optimisation

---
## Table of content:
 - [Convolutional KAN layers](#item-one)
 - [Model Zoo](#item-two)
 - [Performance Metrics](#item-three)
 - [Discussion](#item-four)
 - [Usage](#item-five)
 - [Accelerate-based training](#item-six)
 - [Contributions](#item-seven)
 - [Acknowledgements](#item-eight)
 - [References](#item-nine)


<a id="item-one"></a>
## Introducing Convolutional KAN layers

- The `KANConv1DLayer`, `KANConv2DLayer`, `KANConv3DLayer` classes represents a convolutional layers based on Kolmogorov Arnold Network, introduced in [1]. Baseline model implemented in `models/baselines/conv_kan_baseline.py`.

- The `KALNConv1DLayer`, `KALNConv2DLayer`, `KALNConv3DLayer` classes represents a convolutional layers based on Kolmogorov Arnold Legendre Network, introduced in [2]. Baseline model implemented in `models/baselines/conv_kaln_baseline.py`.

- The `FastKANConv1DLayer`, `FastKANConv2DLayer`, `FastKANConv3DLayer` classes represents a convolutional layers based on Fast Kolmogorov Arnold Network, introduced in [3]. Baseline model implemented in `models/baselines/fast_conv_kan_baseline.py`.

- The `KACNConv1DLayer`, `KACNConv1DLayer`, `KACNConv1DLayer` classes represents a convolutional layers based on Kolmogorov Arnold Network with Chebyshev polynomials instead of B-splines, introduced in [4]. Baseline model implemented in `models/baselines/conv_kacn_baseline.py`.

- The `KAGNConv1DLayer`, `KAGNConv1DLayer`, `KAGNConv1DLayer` classes represents a convolutional layers based on Kolmogorov Arnold Network with Gram polynomials instead of B-splines, introduced in [5]. Baseline model implemented in `models/baselines/conv_kagn_baseline.py`.

- The `WavKANConv1DLayer`, `WavKANConv1DLayer`, `WavKANConv1DLayer` classes represents a convolutional layers based on Wavelet Kolmogorov Arnold Network, introduced in [6]. Baseline model implemented in `models/baselines/conv_wavkan_baseline.py`.

<a id="item-two"></a>
## Model Zoo

### ResKANets

We introduce ResKANets - an ResNet-like model with KAN convolutions instead of regular one. Main class ```ResKANet``` could be found ```models/densekanet.py```. Our implementation supports blocks with KAN, Fast KAN, KALN, KAGN and KACN convolutional layers.

After 75 training epochs on CIFAR10 ResKANet 18 with Kolmogorov Arnold Legendre convolutions achieved 84.17% accuracy and 0.985 AUC (OVO).

After 75 training epochs on Tiny Imagenet ResKANet 18 with Kolmogorov Arnold Legendre convolutions achieved 28.62% accuracy, 55.49% top-5 accuracy, and 0.932 AUC (OVO).

Please, take into account that this is preliminary results and more experiments are in progress right now.

### DenseKANets

We introduce DenseKANets - an DenseNet-like model with KAN convolutions instead of regular one. Main class ```DenseKANet``` could be found ```models/reskanet.py```. Our implementation supports blocks with KAN, Fast KAN, KALN, KAGN and KACN convolutional layers.

After 250 training epochs on Tiny Imagenet DenseNet 121 with Kolmogorov Arnold Gram convolutions achieved 40.61% accuracy, 65.08% top-5 accuracy, and 0.957 AUC (OVO).

Please, take into account that this is preliminary results and more experiments are in progress right now.

### VGGKAN

We introduce VGGKANs - an VGG like models with KAN convolutions instead of regular one, based on resnet blocks. Main class ```VGG``` could be found ```models/vggkan.py```. 


### UKANet and U2KANet

We introduce UKANets and U2KANets - an U-net/U2-net like model with KAN convolutions instead of regular one, based on resnet blocks. Main class ```UKANet``` could be found ```models/ukanet.py```. Our implementation supports Basic and Bottleneck blocks with KAN, Fast KAN, KALN, KAGC and KACN convolutional layers.

<a id="item-three"></a>
## Performance Metrics

[Baseline models on MNIST and CIFAR10/100](./reports/mnist_cifar_baseline.md)

<a id="item-four"></a>
## Discussion

First and foremost, it should be noted that the results obtained are preliminary. The model architecture has not been thoroughly explored and represents only two of many possible design variants.

Nevertheless, the experiments indicate that Kolmogorov-Arnold convolutional networks outperform the classical convolutional architecture on the MNIST dataset, but significantly underperform on CIFAR-10 and CIFAR-100 in terms of quality. The ChebyKAN-based convolution encounters stability issues during training, necessitating further investigation.

As a next step, I plan to search for a suitable architecture for KAN convolutions that can achieve acceptable quality on CIFAR-10/100 and attempt to scale these models to more complex datasets.

## Prerequisites

Ensure you have the following installed on your system:

- Python (version 3.9 or higher)
- CUDA Toolkit (corresponding to your PyTorch installation's CUDA version)
- cuDNN (compatible with your installed CUDA Toolkit)

<a id="item-five"></a>
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

<a id="item-six"></a>
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
<a id="item-seven"></a>
## Contributions

Contributions are welcome. Please raise issues as necessary.
<a id="item-eight"></a>
## Acknowledgements

This repository based on [TorchKAN](https://github.com/1ssb/torchkan/), [FastKAN](https://github.com/ZiyaoLi/fast-kan), [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN), [GRAMKAN](https://github.com/Khochawongwat/GRAMKAN) and [WavKAN](https://github.com/zavareh1/Wav-KAN). and we would like to say thanks for their open research and exploration.

<a id="item-nine"></a>
## References

- [1] Ziming Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024, arXiv. https://arxiv.org/abs/2404.19756
- [2] https://github.com/1ssb/torchkan
- [3] https://github.com/ZiyaoLi/fast-kan
- [4] https://github.com/SynodicMonth/ChebyKAN
- [5] https://github.com/Khochawongwat/GRAMKAN
- [6] https://github.com/zavareh1/Wav-KAN  
- [7] https://github.com/KindXiaoming/pykan
- [8] https://github.com/Blealtan/efficient-kan

## Star History

<a href="https://star-history.com/#IvanDrokin/torch-conv-kan&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=IvanDrokin/torch-conv-kan&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=IvanDrokin/torch-conv-kan&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=IvanDrokin/torch-conv-kan&type=Date" />
 </picture>
</a>
