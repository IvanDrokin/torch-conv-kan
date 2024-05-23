from typing import Any, Tuple

import torch.utils.data as data


class Classification(data.Dataset):

    def __init__(self, dataset, transform=None) -> None:
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.length = len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = self.dataset[index]
        sample, label = sample['image'].convert('RGB'), sample['label']

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __len__(self) -> int:
        return self.length
