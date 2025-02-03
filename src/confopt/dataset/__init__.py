from typing import Type, Union

from confopt.enums import DatasetType

from .data import (
    AbstractData,
    CIFAR10Data,
    CIFAR100Data,
    ImageNet16Data,
    ImageNet16120Data,
    TaskonomyClassObjectData,
    TaskonomyClassSceneData,
    USPSData,
)


def get_taskonomy_dataset(domain: str) -> Type[AbstractData]:
    if domain == "class_object":
        return TaskonomyClassObjectData
    elif domain == "class_scene":  # noqa: RET505
        return TaskonomyClassSceneData
    raise ValueError("Invalid domain for Taskonomy dataset")


def get_dataset(
    dataset: DatasetType,
    domain: Union[str, None],
    root: str,
    cutout: int,
    cutout_length: int,
    train_portion: float = 1.0,
) -> AbstractData:
    dataset_cls: Type[AbstractData] = CIFAR10Data
    if dataset == DatasetType.CIFAR10:
        dataset_cls = CIFAR10Data
    elif dataset == DatasetType.CIFAR100:
        dataset_cls = CIFAR100Data
    elif dataset == DatasetType.IMGNET16:
        dataset_cls = ImageNet16Data
    elif dataset == DatasetType.IMGNET16_120:
        dataset_cls = ImageNet16120Data
    elif dataset == DatasetType.TASKONOMY:
        assert domain is not None, "Domain should be provided for Taskonomy dataset"
        dataset_cls = get_taskonomy_dataset(domain)
    elif dataset == DatasetType.USPS:
        dataset_cls = USPSData
    else:
        raise ValueError("Invalid dataset")

    return dataset_cls(
        root=root,
        cutout=cutout,
        cutout_length=cutout_length,
        train_portion=train_portion,
    )


__all__ = [
    "AbstractData",
    "CIFAR10Data",
    "CIFAR100Data",
    "ImageNet16Data",
    "ImageNet16120Data",
    "TaskonomyClassObjectData",
    "TaskonomyClassSceneData",
    "get_dataset",
]
