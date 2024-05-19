import hydra
import torch
import torch.nn as nn
from datasets import load_dataset
from torchinfo import summary
from torchvision.transforms import v2

from models.reskanet import moe_reskalnet_50x64p
from train import Classification, train_model


def get_data():
    dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir='./data/tiny-imagenet')
    transforms_train = v2.Compose([
        v2.ToImage(),
        v2.AutoAugment(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms_val = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Classification(dataset['train'], transform=transforms_train)
    val_dataset = Classification(dataset['valid'], transform=transforms_val)
    return train_dataset, val_dataset


@hydra.main(version_base=None, config_path="./configs/", config_name="tiny-imagenet-reskanet.yaml")
def main(cfg):
    model = moe_reskalnet_50x64p(3, 200, groups=cfg.model.groups,
                                 degree=cfg.model.degree, width_scale=cfg.model.width_scale,
                                 dropout=cfg.model.dropout, dropout_linear=cfg.model.dropout_linear)

    summary(model, (64, 3, 64, 64), device='cpu')
    dataset_train, dataset_test = get_data()
    loss_func = nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)

    train_model(model, dataset_train, dataset_test, loss_func, cfg)


if __name__ == '__main__':
    main()
