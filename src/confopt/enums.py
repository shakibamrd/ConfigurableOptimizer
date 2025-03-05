from __future__ import annotations

from enum import Enum


class SearchSpaceType(Enum):
    DARTS = "darts"
    NB201 = "nb201"
    NB1SHOT1 = "nb1shot1"
    TNB101 = "tnb101"
    BABYDARTS = "baby_darts"
    RobustDARTS = "robust_darts"

    def __str__(self) -> str:
        return self.value


class SamplerType(Enum):
    DARTS = "darts"
    DRNAS = "drnas"
    GDAS = "gdas"
    LAMBDADARTS = "lambda_darts"
    SNAS = "snas"
    REINMAX = "reinmax"

    def __str__(self) -> str:
        return self.value


class PerturbatorType(Enum):
    RANDOM = "random"
    ADVERSERIAL = "adverserial"
    NONE = "none"

    def __str__(self) -> str:
        return self.value


class DatasetType(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMGNET16 = "imgnet16"
    IMGNET16_120 = "imgnet16_120"
    TASKONOMY = "taskonomy"
    AIRCRAFT = "aircraft"

    def __str__(self) -> str:
        return self.value


class CriterionType(Enum):
    CROSS_ENTROPY = "cross_entropy"

    def __str__(self) -> str:
        return self.value


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ASGD = "asgd"

    def __str__(self) -> str:
        return self.value


class SchedulerType(Enum):
    CosineAnnealingLR = "cosine_annealing_lr"
    CosineAnnealingWarmRestart = "cosine_annealing_warm_restart"

    def __str__(self) -> str:
        return self.value
