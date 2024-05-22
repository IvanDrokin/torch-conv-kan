## Baseline models on MNIST and CIFAR10/100

Baseline models were chosen to be simple networks with 4 and 8 convolutional layers. To reduce dimensionality, convolutions with dilation=2 were used. In the 4-layer model, the second and third convolutions had dilation=2, while in the 8-layer model, the second, third, and sixth convolutions had dilation=2.

The number of channels in the convolutions was the same for all models:

For 4 layers: 32, 64, 128, 512
For 8 layers: 2, 64, 128, 512, 1024, 1024, 1024, 1024
After the convolutions, Global Average Pooling was applied, followed by a linear output layer.

In the case of classic convolutions, a traditional structure was used: convolution - batch normalization - ReLU.

All experiments were conducted on an NVIDIA RTX 3090 with identical training parameters. For more details, please refer to the file ```mnist_conv.py```.

### MNIST
Accuracy on the training and validation MNIST datasets, 4 convolutional layer models
![MNIST train and validation accuracy for 4 layer models](../assets/MNIST.png)

Accuracy on the training and validation MNIST datasets, 8 convolutional layer models
![MNIST train and validation accuracy for 8 layer models](../assets/MNIST_8.png)

| Model                       | Val. Accuracy | Parameters | Eval Time, s |
|-----------------------------|---------------|------------|--------------|
| SimpleConv, 4 layers        | 98.94         | 178122     | 0.4017       |
| SimpleKANConv, 4 layers     | **99.48**     | 1542199    | 1.7437       |
| SimpleFastKANConv, 4 layers | 98.29         | 1542186    | 0.4558       |
| SimpleKALNConv, 4 layers    | 99.40         | 859050     | 0.5085       |
| SimpleKACNConv, 4 layers    | 97.54         | 689738     | 0.4225       |
| SimpleKAGNConv, 4 layers    | 98.01         | 859066     | 0.4541       |
| SimpleConv, 8 layers        | 99.37         | 42151850   | 1.7582       |
| SimpleKANConv, 8 layers     | 99.39         | 75865159   | 5.7914       |
| SimpleFastKANConv, 8 layers | 99.09         | 75865130   | 2.4105       |
| SimpleKALNConv, 8 layers    | 99.36         | 42151850   | 1.7582       |
| SimpleKACNConv, 8 layers    | 99.22         | 33733194   | 0.8509       |
| SimpleKAGNConv, 8 layers    | 99.37         | 42151882   |   1.6168     |

### CIFAR10
Accuracy on the training and validation CIFAR10 datasets, 4 convolutional layer models
![CIFAR10 train and validation accuracy for 4 layer models](../assets/CIFAR10.png)

Accuracy on the training and validation CIFAR10 datasets, 4 convolutional layer models
![CIFAR10 train and validation accuracy for 8 layer models](../assets/CIFAR10_8.png)

|           Model           |Val. Accuracy|Parameters|Eval Time, s|
|---------------------------|-------------|----------|------------|
|    SimpleConv, 4 layers   |    69.69    |  178698  |   0.5481   |
|  SimpleKANConv, 4 layers  |    59.37    |  1547383 |   2.2969   |
|SimpleFastKANConv, 4 layers|    57.39    |  1547370 |   0.6169   |
|  SimpleKALNConv, 4 layers |    63.86    |  861930  |   0.6824   |
|  SimpleKACNConv, 4 layers |    57.03    |  692042  |   0.5853   |
|  SimpleKAGNConv, 4 layers |    50.31    |  861946  |   0.6559   |
|    SimpleConv, 8 layers   |  **76.08**  |  8453642 |   0.5647   |
|  SimpleKANConv, 8 layers  |    64.62    | 75870343 |   7.0329   |
|SimpleFastKANConv, 8 layers|    57.21    | 75870314 |   2.7723   |
|  SimpleKALNConv, 8 layers |    66.77    | 42154730 |   1.9057   |
|  SimpleKACNConv, 8 layers |    64.98    | 33735498 |   0.9085   |
|  SimpleKAGNConv, 8 layers |    65.87    | 42154762 |   1.7415   |

### CIFAR100
Accuracy on the training and validation CIFAR100 datasets, 4 convolutional layer models
![CIFAR100 train and validation accuracy for 4 layer models](../assets/CIFAR100.png)

Accuracy on the training and validation CIFAR100 datasets, 8 convolutional layer models
![CIFAR100 train and validation accuracy for 8 layer models](../assets/CIFAR100_8.png)

|           Model           |Val. Accuracy|Parameters|Eval Time, s|
|---------------------------|-------------|----------|------------|
|    SimpleConv, 4 layers   |    38.99    |  224868  |   0.5533   |
|  SimpleKANConv, 4 layers  |    20.53    |  1593553 |   2.3098   |
|SimpleFastKANConv, 4 layers|    32.48    |  1593540 |   0.6175   |
|  SimpleKALNConv, 4 layers |    22.36    |  908100  |   0.6540   |
|  SimpleKACNConv, 4 layers |    34.17    |  738212  |   0.5820   |
|  SimpleKAGNConv, 4 layers |    11.77    |  908116  |   0.6247   |
|    SimpleConv, 8 layers   |  **43.32**  |  8545892 |   0.5663   |
|  SimpleKANConv, 8 layers  |    23.22    | 75962593 |   7.0452   |
|SimpleFastKANConv, 8 layers|    23.75    | 75962564 |   2.7713   |
|  SimpleKALNConv, 8 layers |    18.90    | 42246980 |   1.8955   |
|  SimpleKACNConv, 8 layers |     0.98    | 33827748 |   0.9093   |
|  SimpleKAGNConv, 8 layers |    19.42    | 42247012 |   1.7440   |

