from typing import Any, Tuple
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data


class Classification(data.Dataset):

    def __init__(self, dataset, transform=None) -> None:
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.length = len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = self.dataset[index]
        sample, label = sample['image'], sample['label']
        if isinstance(sample, str):
            sample = Image.open(sample)

        sample = sample.convert('RGB')

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __len__(self) -> int:
        return self.length


class Segmentation(data.Dataset):
    def __init__(self, dataset, num_classes, transform_input=None, transform_target=None, augmentations=None) -> None:
        super().__init__()

        self.dataset = dataset
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.augmentations = augmentations
        self.num_classes = num_classes
        self.length = len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = self.dataset[index]
        sample, segmentation_map = sample['image'], sample['label']
        if isinstance(sample, str):
            sample = Image.open(sample)

        if self.augmentations:
            augmented = self.augmentations(image=np.asarray(sample), mask=np.asarray(segmentation_map))
            sample = augmented['image']
            segmentation_map = augmented['mask']

        if self.transform_input:
            sample = self.transform_input(sample)
        if self.transform_target:
            segmentation_map = self.transform_target(segmentation_map)
            if self.num_classes > 2:
                segmentation_map = torch.nn.functional.one_hot(segmentation_map, num_classes=self.num_classes)

        return sample, segmentation_map

    def __len__(self) -> int:
        return self.length
