import hydra
import torch
import torch.nn as nn
import torchvision
from ray import tune
from torch.utils.data import random_split
from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy

from gram_dropout_placement import EightSimpleConvKAGN, SixteenSimpleConvKAGN
from train.param_tune import tune_params


def get_data(cfg):
    transforms_train = v2.Compose([
        v2.ToImage(),
        v2.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms_val = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transforms_train)

    test_abs = int(len(train_dataset) * 0.8)
    train_dataset, val_dataset = random_split(
        train_dataset, [test_abs, len(train_dataset) - test_abs]
    )
    val_dataset.transform = transforms_val
    # Load and transform the CIFAR100 validation dataset
    # val_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms_val)
    return {'train': train_dataset, 'val': val_dataset}


def get_model8(cfg):
    ws = cfg.model.width_scale
    layer_sizes = [8 * ws, 16 * ws, 32 * ws, 64 * ws, 128 * ws, 128 * ws, 256 * ws, 256 * ws]

    return EightSimpleConvKAGN(
        layer_sizes,
        num_classes=cfg.model.num_classes,
        input_channels=cfg.model.input_channels,
        degree=cfg.model.degree,
        degree_out=cfg.model.degree_out,
        groups=cfg.model.groups,
        dropout_poly=cfg.model.dropout_poly,
        dropout_full=cfg.model.dropout_full,
        dropout_degree=cfg.model.dropout_degree,
        dropout_linear=cfg.model.dropout_linear,
        l1_penalty=cfg.model.l1_decay,
        affine=True,
        norm_layer=nn.BatchNorm2d,
        drop_type=cfg.model.drop_type)


def get_model16(cfg):
    ws = cfg.model.width_scale
    layer_sizes = [8 * ws, 8 * ws, 16 * ws, 16 * ws, 32 * ws, 32 * ws, 64 * ws, 64 * ws,
                   128 * ws, 128 * ws, 128 * ws, 128 * ws, 256 * ws, 256 * ws, 256 * ws, 256 * ws]
    stride_indexes = [2, 4, 8, 12]
    return SixteenSimpleConvKAGN(
        layer_sizes,
        stride_indexes,
        num_classes=cfg.model.num_classes,
        input_channels=cfg.model.input_channels,
        degree=cfg.model.degree,
        degree_out=cfg.model.degree_out,
        groups=cfg.model.groups,
        dropout_poly=cfg.model.dropout_poly,
        dropout_full=cfg.model.dropout_full,
        dropout_degree=cfg.model.dropout_degree,
        dropout_linear=cfg.model.dropout_linear,
        l1_penalty=cfg.model.l1_penalty,
        affine=True,
        norm_layer=nn.BatchNorm2d,
        drop_type=cfg.model.drop_type)


def loss_func(cfg):
    return nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])


@hydra.main(version_base=None, config_path="./configs/", config_name="cifar100_raytune.yaml")
def ray_main(cfg):
    search_config = {
        "l1_activation_penalty": tune.choice([10**(-i) for i in range(5, 9)] + [0, ]),
        "l2_activation_penalty": tune.choice([10**(-i) for i in range(5, 9)] + [0, ]),
        "l1_decay": tune.choice([10**(-i) for i in range(5, 9)] + [0, ]),
        "dropout_linear": tune.uniform(0.0, 0.5),
        "dropout_poly": tune.uniform(0.0, 0.15),
        "dropout_degree": tune.uniform(0.0, 0.15),
        "dropout_full": tune.uniform(0.0, 0.15),
        "drop_type": tune.choice(['regular', 'noise']),
        "width_scale": tune.choice([1, 2, 3, 4, 5, 6, 7, 8]),
        "degree": tune.choice([3, 5, 7]),
        "lr_power": tune.uniform(0.1, 2.),
        "adam_weight_decay": tune.uniform(1e-9, 1e-5),
        "learning_rate": tune.uniform(1e-6, 1e-3),
        "label_smoothing": tune.uniform(0.0, 0.2)
    }

    tune_params(search_config, get_model8, cfg, get_data, loss_func,
                num_samples=cfg.raytune.num_samples, max_num_epochs=cfg.raytune.max_num_epochs,
                gpus_per_trial=cfg.raytune.gpus_per_trial, cpus_per_trial=cfg.raytune.cpus_per_trial)


if __name__ == '__main__':
    ray_main()
