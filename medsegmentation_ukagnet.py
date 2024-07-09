import os
from copy import deepcopy
from glob import glob

import albumentations as A
import cv2
import hydra
import numpy as np
import torch.nn as nn
import torch.utils.data
from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from omegaconf import OmegaConf, open_dict
from sklearn.model_selection import train_test_split
from torchinfo import summary

from models import UKAGNet
from train import train_model, DiceLossWithBCE


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            # print(os.path.join(self.mask_dir, str(i),
            #             img_id + self.mask_ext))

            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                                                img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        if mask.max() < 1:
            mask[mask > 0] = 1.0

        return img, mask


def get_data(dataset_name, data_dir, input_h, input_w, num_classes, seed):
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(data_dir, dataset_name, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=seed)

    train_transform = Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.Rotate(p=0.5),
        Resize(input_h, input_w),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(input_h, input_w),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(data_dir, dataset_name, 'images'),
        mask_dir=os.path.join(data_dir, dataset_name, 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=num_classes,
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(data_dir, dataset_name, 'images'),
        mask_dir=os.path.join(data_dir, dataset_name, 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=num_classes,
        transform=val_transform)

    return train_dataset, val_dataset


@hydra.main(version_base=None, config_path="./configs/", config_name="medseg_ukans.yaml")
def main(cfg):
    assert cfg.model.model_type in ['unet', 'u2net', 'u2net_small', 'vanilla_u2net',
                                    'u2kagnet_bn'], "Unimplemented model"
    model = None
    for dataset_name, image_size in [("busi", 256), ("cvc", 256), ("glas", 512)]:
        for mixer_type in ['conv', ]:
            for use_bottleneck in [False, True]:
                for ws in [4, 2, 1]:
                    if use_bottleneck:
                        model_type = f'bottleneck_ukagnet_{mixer_type}'
                    else:
                        model_type = f'ukagnet_{mixer_type}'

                    model = UKAGNet(
                        input_channels=3,
                        num_classes=1,
                        unet_depth=5,
                        unet_layers=2,
                        width_scale=1,
                        use_bottleneck=use_bottleneck,
                        mixer_type=mixer_type,
                        groups=cfg.model.groups,
                        degree=cfg.model.degree,
                        dropout=cfg.model.dropout,
                        affine=True,
                        norm_layer=nn.BatchNorm2d,
                        focal_window=3,
                        focal_level=2,
                        focal_factor=2,
                        use_postln_in_modulation=True,
                        normalize_modulator=True,
                        full_kan=True,
                    )

                    run_cfg = deepcopy(cfg)
                    OmegaConf.set_struct(run_cfg, True)
                    with open_dict(run_cfg):
                        run_cfg.wandb.runname = f"{model_type} x {ws}"
                        run_cfg.wandb.project_name = f'ukagnets-{dataset_name.lower()}'
                        run_cfg.output_dir = f"./experiments/{model_type}_{dataset_name.lower()}x{ws}/"
                        run_cfg.logging_dir = f"./experiments/{model_type}_{dataset_name.lower()}x{ws}/train_logs/"
                        run_cfg.model_name = f"{model_type}x{ws}"

                    summary(model, (4, 3, 256, 256), device='cuda:0')
                    dataset_train, dataset_val = get_data(dataset_name, "./data/UKAN_DATA/", image_size, image_size,
                                                          cfg.model.num_classes, cfg.seed)

                    loss_func = DiceLossWithBCE(smooth=1.0, bce_weight=0.1)
                    train_model(model, dataset_train, dataset_val, loss_func, run_cfg, dataset_test=None, cam_reporter=None)


if __name__ == '__main__':
    main()
