# VGG-like Kolmogorov-Arnold Convolutional network on  Imagenet1k

## VGG 11v2 with Gram polynomials
### Model description

The model consists of consecutive 10 Gram ConvKAN Layers with InstanceNorm2d, polynomial degree equal to 5, GlobalAveragePooling and Linear classification head:

1. KAGN Convolution, 32 filters, 3x3
2. Max pooling, 2x2
3. KAGN Convolution, 64 filters, 3x3
4. Max pooling, 2x2
5. KAGN Convolution, 128 filters, 3x3
6. KAGN Convolution, 128 filters, 3x3
7. Max pooling, 2x2
8. KAGN Convolution, 256 filters, 3x3
9. KAGN Convolution, 256 filters, 3x3
10 Max pooling, 2x2
11. KAGN Convolution, 256 filters, 3x3
12. KAGN Convolution, 256 filters, 3x3
13. Max pooling, 2x2
14. KAGN Convolution, 256 filters, 3x3
15. KAGN Convolution, 256 filters, 3x3
16. Global Average pooling
17. Output layer, 1000 nodes.

![model image](https://github.com/IvanDrokin/torch-conv-kan/blob/main/assets/vgg_kagn_11_v2.png?raw=true)

### Training data
This model trained on Imagenet1k dataset (1281167 images in train set)

### Training procedure

Model was trained during 200 full epochs with AdamW optimizer, with following parameters:
```python
{'learning_rate': 0.0009, 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_weight_decay': 5e-06,
'adam_epsilon': 1e-08, 'lr_warmup_steps': 7500, 'lr_power': 0.3, 'lr_end': 1e-07, 'set_grads_to_none': False}
```
And this augmnetations:
```python
from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy


transforms_train = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomResizedCrop(224, antialias=True),
    v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                     v2.AutoAugment(AutoAugmentPolicy.IMAGENET)
                     ]),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Evaluation results

On Imagenet1k Validation:

| Accuracy, top1 | Accuracy, top5 | AUC (ovo) | AUC (ovr) |
|:--------------:|:--------------:|:---------:|:---------:|
|      59.1      |      82.29     |   99.43   |   99.43   |

On Imagenet1k Test:
Coming soon


## VGG 11v4 with Gram polynomials
### Model description

The model consists of consecutive 10 Gram ConvKAN Layers with InstanceNorm2d, polynomial degree equal to 5, GlobalAveragePooling and Linear classification head:

1. KAGN Convolution, 32 filters, 3x3
2. Max pooling, 2x2
3. KAGN Convolution, 64 filters, 3x3
4. Max pooling, 2x2
5. KAGN Convolution, 128 filters, 3x3
6. KAGN Convolution, 128 filters, 3x3
7. Max pooling, 2x2
8. KAGN Convolution, 256 filters, 3x3
9. KAGN Convolution, 256 filters, 3x3
10 Max pooling, 2x2
11. KAGN Convolution, 256 filters, 3x3
12. KAGN Convolution, 256 filters, 3x3
13. Max pooling, 2x2
14. KAGN Convolution, 512 filters, 3x3
15. KAGN Convolution, 512 filters, 3x3
16. Global Average pooling
17. Output layer, 1000 nodes.

![model image](https://github.com/IvanDrokin/torch-conv-kan/blob/main/assets/vgg_kagn_11_v4.png?raw=true)

### Training data
This model trained on Imagenet1k dataset (1281167 images in train set)

### Training procedure

Model was trained during 200 full epochs with AdamW optimizer, with following parameters:
```python
{'learning_rate': 0.0009, 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_weight_decay': 5e-06,
'adam_epsilon': 1e-08, 'lr_warmup_steps': 7500, 'lr_power': 0.3, 'lr_end': 1e-07, 'set_grads_to_none': False}
```
And this augmnetations:
```python
from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy


transforms_train = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomResizedCrop(224, antialias=True),
    v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                     v2.AutoAugment(AutoAugmentPolicy.IMAGENET)
                     ]),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Evaluation results

On Imagenet1k Validation:

| Accuracy, top1 | Accuracy, top5 | AUC (ovo) | AUC (ovr) |
|:--------------:|:--------------:|:---------:|:---------:|
|      61.17     |      83.26     |   99.42   |   99.43   |

On Imagenet1k Test:
Coming soon

## VGG 11v4 with Bottleneck Gram polynomials
### Model description

The model consists of consecutive 10 Bottleneck Gram ConvKAN Layers with BatchNorm2D, polynomial degree equal to 5, GlobalAveragePooling and Linear classification head:

1. BottleNeck KAGN Convolution, 32 filters, 3x3
2. Max pooling, 2x2
3. BottleNeck KAGN Convolution, 64 filters, 3x3
4. Max pooling, 2x2
5. BottleNeck KAGN Convolution, 128 filters, 3x3
6. BottleNeck KAGN Convolution, 128 filters, 3x3
7. Max pooling, 2x2
8. BottleNeck KAGN Convolution, 256 filters, 3x3
9. BottleNeck KAGN Convolution, 256 filters, 3x3
10 Max pooling, 2x2
11. BottleNeck KAGN Convolution, 256 filters, 3x3
12. BottleNeck KAGN Convolution, 256 filters, 3x3
13. Max pooling, 2x2
14. BottleNeck KAGN Convolution, 512 filters, 3x3
15. BottleNeck KAGN Convolution, 512 filters, 3x3
16. Global Average pooling
17. Output layer, 1000 nodes.

![model image](https://github.com/IvanDrokin/torch-conv-kan/blob/main/assets/vgg_kagn_11_v4.png?raw=true)

### Training data
This model trained on Imagenet1k dataset (1281167 images in train set)

### Training procedure

Model was trained during 200 full epochs with AdamW optimizer, with following parameters:
```python
{'learning_rate': 0.0005, 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_weight_decay': 5e-06,
'adam_epsilon': 1e-08, 'lr_warmup_steps': 7500, 'lr_power': 0.3, 'lr_end': 1e-07, 'set_grads_to_none': False}
```
And this augmnetations:
```python
from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy


transforms_train = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomResizedCrop(224, antialias=True),
    v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                     v2.AutoAugment(AutoAugmentPolicy.IMAGENET)
                     ]),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Evaluation results

On Imagenet1k Validation:

| Accuracy, top1 | Accuracy, top5 | AUC (ovo) | AUC (ovr) |
|:--------------:|:--------------:|:---------:|:---------:|
|      68.5      |      88.46     |   99.61   |   99.61   |

On Imagenet1k Test:
Coming soon

## How to use

First, clone the repository:

```
git clone https://github.com/IvanDrokin/torch-conv-kan.git
cd torch-conv-kan
pip install -r requirements.txt
```
Then you can initialize the model and load weights.

```python
import torch
from models import vggkagn_bn
model = vggkagn_bn(3,
                   1000,
                   groups=1,
                   degree=5,
                   dropout=0.05,
                   l1_decay=0,
                   width_scale=2,
                   affine=True,
                   norm_layer=nn.BatchNorm2d,
                   expected_feature_shape=(1, 1),
                   vgg_type='VGG11v4')
model.from_pretrained('brivangl/vgg_kagn_bn11_v4')
```

Transforms, used for validation on Imagenet1k:

```python
from torchvision.transforms import v2
transforms_val = v2.Compose([
        v2.ToImage(),
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```
