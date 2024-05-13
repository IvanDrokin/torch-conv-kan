# TorchConv KAN: A Convolutional KAN & KAL Net: Kolmogorov Arnold Legendre Network

This project introduces and demonstrates the training, validation, and quantization of the Convolutional KAN model using PyTorch with CUDA acceleration. The `torch-conv-kan` evaluates performance on the MNIST dataset.

## Project Status: Under Development
### Roadmap
- [ ] Convolutional KAN layers
- [x] Convolutional KALN layers
- [ ] MNIST Benchmarks

---

## Introducing Convolutional KAN & KALN models

The `KALNConv1DLayer`, `KALNConv2DLayer`, `KALNConv3DLayer` classes represents a convolutional layers based on Kolmogorov Arnold Legendre Network, introduced in [1]. Baseline model implemented in `conv_kaln/conv_kanl_baseline.py`.

KAN-like convolutions are coming later.

## Performance Metrics
- **Accuracy:** `SimpleConvKANL` achieved an impressive **99.31% accuracy on the MNIST dataset**, demonstrating its capability to handle complex patterns in image data.
- More benchmarks are coming soon.

## Prerequisites

Ensure you have the following installed on your system:

- Python (version 3.9 or higher)
- CUDA Toolkit (corresponding to your PyTorch installation's CUDA version)
- cuDNN (compatible with your installed CUDA Toolkit)

## Usage

```python
mnist_conv_kaln.py
```
This script will train the model, validate it, quantise and log performance metrics using wandb.

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

This repository based on [TorchKAN](https://github.com/1ssb/torchkan/), and we would like to sya thanks for their open research and exploration.


## References

- [1] https://github.com/1ssb/torchkan
- [2] Ziming Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024, arXiv. https://arxiv.org/abs/2404.19756
- [3] https://github.com/KindXiaoming/pykan
- [4] https://github.com/Blealtan/efficient-kan