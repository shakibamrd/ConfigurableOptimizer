from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.io import read_image
from torchvision.transforms import Compose

from .imgnet16 import ImageNet16

DS = Tuple[Union[Dataset, None], Union[Sampler, None]]


class CUTOUT:
    def __init__(self, length: int):
        self.length = length

    def __repr__(self) -> str:
        return "{name}(length={length})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)  # type: ignore
        mask = mask.expand_as(img)  # type: ignore
        img *= mask
        return img


class AbstractData(ABC):
    def __init__(self, root: str, train_portion: float = 1.0) -> None:
        self.root = root
        self.train_portion = train_portion
        if train_portion == 1:
            self.shuffle = True
        else:
            self.shuffle = False

    @abstractmethod
    def build_datasets(self) -> tuple[DS, DS, Dataset]:
        ...

    @abstractmethod
    def get_transforms(self) -> tuple[Compose, Compose]:
        ...

    @abstractmethod
    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        ...

    def get_dataloaders(
        self, batch_size: int = 64, n_workers: int = 2
    ) -> tuple[DataLoader, DataLoader | None, DataLoader]:
        (
            (train_data, train_sampler),
            (val_data, val_sampler),
            test_data,
        ) = self.build_datasets()
        train_queue = DataLoader(
            train_data,  # type: ignore
            batch_size=batch_size,
            pin_memory=True,
            sampler=train_sampler,
            num_workers=n_workers,
            shuffle=self.shuffle,
        )
        if val_data is not None:
            valid_queue = DataLoader(
                val_data,
                batch_size=batch_size,
                pin_memory=True,
                sampler=val_sampler,
                num_workers=n_workers,
            )
        else:
            valid_queue = None

        test_queue = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=n_workers,
        )

        return train_queue, valid_queue, test_queue


class CIFARData(AbstractData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, train_portion)
        self.cutout = cutout

    def get_transforms(self) -> tuple[Compose, Compose]:
        lists = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),  # type: ignore
        ]

        if self.cutout > 0:
            lists += [CUTOUT(self.cutout)]
        train_transform = transforms.Compose(lists)

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),  # type: ignore
            ]
        )

        return train_transform, test_transform

    def build_datasets(self) -> tuple[DS, DS, Dataset]:
        train_transform, test_transform = self.get_transforms()
        train_data, test_data = self.load_datasets(
            self.root, train_transform, test_transform
        )

        if self.train_portion < 1:
            num_train = len(train_data)  # type: ignore
            indices = list(range(num_train))
            split = int(np.floor(self.train_portion * num_train))
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[:split]
            )
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train]
            )
            return (
                (train_data, train_sampler),
                (train_data, val_sampler),
                test_data,
            )

        return (train_data, None), (None, None), test_data


class ImageNetData(AbstractData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, train_portion)
        self.cutout = cutout
        self.mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        self.std = [x / 255 for x in [63.22, 61.26, 65.09]]

    def get_transforms(self) -> tuple[Compose, Compose]:
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ]
        if self.cutout > 0:
            lists += [CUTOUT(self.cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )

        return train_transform, test_transform

    def build_datasets(self) -> tuple[DS, DS, Dataset]:
        train_transform, test_transform = self.get_transforms()
        train_data, test_data = self.load_datasets(
            self.root, train_transform, test_transform
        )

        if self.train_portion > 0:
            num_train = len(train_data)  # type: ignore
            indices = list(range(num_train))
            split = int(np.floor(self.train_portion * num_train))
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[:split]
            )
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train]
            )
            return (
                (train_data, train_sampler),
                (train_data, val_sampler),
                test_data,
            )

        return (train_data, None), (None, None), test_data


class CIFAR10Data(CIFARData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, cutout, train_portion)
        self.mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.std = [x / 255 for x in [63.0, 62.1, 66.7]]

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = dset.CIFAR10(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR10(
            root, train=False, transform=test_transform, download=True
        )

        assert len(train_data) == 50000
        assert len(test_data) == 10000
        return train_data, test_data


class CIFAR100Data(CIFARData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, cutout, train_portion)
        self.mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        self.std = [x / 255 for x in [63.0, 62.1, 66.7]]

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = dset.CIFAR100(
            root, train=True, transform=train_transform, download=True
        )
        test_data = dset.CIFAR100(
            root, train=False, transform=test_transform, download=True
        )

        assert len(train_data) == 50000
        assert len(test_data) == 10000
        return train_data, test_data


class ImageNet16Data(ImageNetData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, cutout, train_portion)

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = ImageNet16(root, True, train_transform)
        test_data = ImageNet16(root, False, test_transform)
        assert len(train_data) == 1281167
        assert len(test_data) == 50000
        return train_data, test_data


class ImageNet16120Data(ImageNetData):
    def __init__(self, root: str, cutout: int, train_portion: float = 1.0):
        super().__init__(root, cutout, train_portion)

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = ImageNet16(root, True, train_transform, 120)
        test_data = ImageNet16(root, False, test_transform, 120)
        assert len(train_data) == 151700
        assert len(test_data) == 6000

        return train_data, test_data


class Taskonomy(Dataset):
    def __init__(
        self,
        domain: str,
        dataset_dir: str = "datasets/taskonomydata_mini",
        train: bool = True,
        custom_building_list: list[str] | None = None,
        transform: Compose = None,
    ) -> None:
        super().__init__()

        train_list = [
            "wainscott",
            "tolstoy",
            "klickitat",
            "pinesdale",
            "stockman",
            "beechwood",
            "coffeen",
            "corozal",
            "benevolence",
            "eagan",
            "forkland",
            "hanson",
            "hiteman",
            "ihlen",
        ]
        test_list = [
            "lakeville",
            "lindenwood",
            "marstons",
            "merom",
            "newfields",
            "pomaria",
            "shelbyville",
            "uvalda",
        ]
        if custom_building_list is None:
            self.building_list = train_list if train else test_list
        else:
            self.building_list = custom_building_list

        self.domain = domain
        self.domain_data_source = {
            "rgb": ("rgb", "png"),
            "autoencoder": ("rgb", "png"),
            "class_object": ("class_object", "npy"),
            "class_scene": ("class_scene", "npy"),
            "normal": ("normal", "png"),
            "room_layout": ("room_layout", "npy"),
            "segmentsemantic": ("segmentsemantic", "png"),
            "jigsaw": ("rgb", "png"),
        }

        self.all_templates = self.get_all_templates(dataset_dir)
        self.transform = transform
        self.dataset_dir = dataset_dir

    def __len__(self) -> int:
        return len(self.all_templates)

    def __getitem__(
        self, idx: torch.Tensor | list[int]
    ) -> tuple[torch.Tensor, np.ndarray]:
        if torch.is_tensor(idx):
            idx = idx.tolist()  # type: ignore
        template = os.path.join(
            self.dataset_dir, self.all_templates[idx]  # type: ignore
        )
        image = read_image(".".join([template.format(domain="rgb"), "png"]))
        label = self.get_label(template)
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_label(
        self,
        template: str,
        selected: bool = False,
        normalize: bool = True,
        final5k: bool = False,
    ) -> np.ndarray:
        label_path = ".".join(
            [
                template.format(domain=self.domain_data_source[self.domain][0]),
                self.domain_data_source[self.domain][1],
            ]
        )
        try:
            logits = np.load(label_path)
        except:
            print(f"corrupted: {label_path}!")
            raise
        lib_data_dir = os.path.abspath(os.path.dirname(__file__))
        if selected:
            selection_file = (
                os.path.join(lib_data_dir, f"{self.domain}_final5k.npy")
                if final5k
                else os.path.join(lib_data_dir, f"{self.domain}_selected.npy")
            )
            selection = np.load(selection_file)
            logits = logits[selection.astype(bool)]
            if normalize:
                logits = logits / logits.sum()
        label = np.asarray(logits)
        return label

    def get_all_templates(self, dataset_dir: str) -> list[str]:
        """Get all templates.

        Args:
            dataset_dir (str): the dir containing the taskonomy dataset

        Returns:
            list[str]: a list of absolute paths of all templates
            e.g. "{building}/{domain}/point_0_view_0_domain_{domain}".
        """
        all_template_paths = []
        for building in self.building_list:
            templates = [
                f"{building}/{{domain}}/"
                + os.path.basename(f.path).replace("_rgb.", "_{domain}.")
                for f in os.scandir(os.path.join(dataset_dir, building, "rgb"))
                if f.is_file()
            ]
            all_template_paths += sorted(templates)
        for i, path in enumerate(all_template_paths):
            f_split = path.split(".")
            if f_split[-1] in ["npy", "png"]:
                all_template_paths[i] = ".".join(f_split[:-1])
        return all_template_paths


class TaskonomyData(AbstractData):
    def __init__(
        self,
        root: str,
        cutout: int,
        train_portion: float = 1.0,
    ) -> None:
        super().__init__(root, train_portion)
        self.cutout = cutout

    def build_datasets(self) -> tuple[DS, DS, Dataset]:
        train_transform, test_transform = self.get_transforms()
        train_data, test_data = self.load_datasets(
            self.root, train_transform, test_transform
        )

        if self.train_portion < 1:
            num_train = len(train_data)  # type: ignore
            indices = list(range(num_train))
            split = int(np.floor(self.train_portion * num_train))
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[:split]
            )
            val_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train]
            )
            return (
                (train_data, train_sampler),
                (train_data, val_sampler),
                test_data,
            )

        return (train_data, None), (None, None), test_data


class ObjectClassificationData(TaskonomyData):
    def __init__(
        self,
        root: str,
        cutout: int,
        train_portion: float = 1,
        building_list: list[str] | None = None,
    ) -> None:
        super().__init__(root, cutout, train_portion)

        self.domain = "class_object"
        self.input_dim = (256, 256)
        self.normal_params = {
            "mean": [0.5224, 0.5222, 0.5221],
            "std": [0.2234, 0.2235, 0.2236],
        }
        self.building_list = building_list

    def get_transforms(self) -> tuple[Compose, Compose]:
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(list(self.input_dim)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(**self.normal_params),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(list(self.input_dim)),
                transforms.ToTensor(),
                transforms.Normalize(**self.normal_params),
            ],
        )
        return train_transform, test_transform

    def load_datasets(
        self, root: str, train_transform: Compose, test_transform: Compose
    ) -> tuple[Dataset, Dataset]:
        train_data = Taskonomy(
            self.domain,
            os.path.join(root, "taskonomydata_mini"),
            train=True,
            custom_building_list=self.building_list,
            transform=train_transform,
        )
        test_data = Taskonomy(
            self.domain,
            os.path.join(root, "taskonomydata_mini"),
            train=False,
            custom_building_list=self.building_list,
            transform=test_transform,
        )
        # assert len(train_data) == 151700
        # assert len(test_data) == 6000

        return train_data, test_data
