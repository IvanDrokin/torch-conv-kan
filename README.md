# TorchConv KAN: A Convolutional KAN & KAL Net: Kolmogorov Arnold Legendre Network

This project introduces and demonstrates the training, validation, and quantization of the Convolutional KAN model using PyTorch with CUDA acceleration. The `torch-conv-kan` evaluates performance on the MNIST and CIFAR datasets.

## Project Status: Under Development
### Roadmap
- [x] Convolutional KAN layers
- [x] Convolutional KALN layers
- [x] Convolutional Fast KAN layers
- [ ] MNIST Benchmarks
- [ ] CIFAR Benchmarks

---

## Introducing Convolutional KAN & KALN models

- The `KANConv1DLayer`, `KANConv2DLayer`, `KANConv3DLayer` classes represents a convolutional layers based on Kolmogorov Arnold Network, introduced in [1]. Baseline model implemented in `models/baselines/conv_kan_baseline.py`.

- The `KALNConv1DLayer`, `KALNConv2DLayer`, `KALNConv3DLayer` classes represents a convolutional layers based on Kolmogorov Arnold Legendre Network, introduced in [2]. Baseline model implemented in `models/baselines/conv_kaln_baseline.py`.

- The `FastKANConv1DLayer`, `FastKANConv2DLayer`, `FastKANConv3DLayer` classes represents a convolutional layers based on Fast Kolmogorov Arnold Network, introduced in [3]. Baseline model implemented in `models/baselines/fast_conv_kan_baseline.py`.


KAN-like convolutions are coming later.

## Performance Metrics
- **Accuracy:** `SimpleConvKALN` achieved an impressive **99.31% accuracy on the MNIST dataset**, demonstrating its capability to handle complex patterns in image data.
- More benchmarks are coming soon.

## Prerequisites

Ensure you have the following installed on your system:

- Python (version 3.9 or higher)
- CUDA Toolkit (corresponding to your PyTorch installation's CUDA version)
- cuDNN (compatible with your installed CUDA Toolkit)

## Usage

```python
python mnist_conv.py
```
This script will train the model, validate it, quantise and log performance metrics.

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

This repository based on [TorchKAN](https://github.com/1ssb/torchkan/) and [FastKAN](https://github.com/ZiyaoLi/fast-kan) , and we would like to say thanks for their open research and exploration.


## References

- [1] Ziming Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024, arXiv. https://arxiv.org/abs/2404.19756
- [2] https://github.com/1ssb/torchkan
- [3] https://github.com/ZiyaoLi/fast-kan  
- [4] https://github.com/KindXiaoming/pykan
- [5] https://github.com/Blealtan/efficient-kan