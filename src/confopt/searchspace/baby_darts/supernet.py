from __future__ import annotations

from functools import partial

import torch
from torch import nn

from confopt.searchspace.common.base_search import (
    DrNASRegTermSupport,
    FairDARTSRegTermSupport,
    GradientMatchingScoreSupport,
    SearchSpace,
)
from confopt.searchspace.darts.core.genotypes import DARTSGenotype
from confopt.utils import update_gradient_matching_scores

from .core import BabyDARTSSearchModel
from .core.model_search import preserve_grads
from .core.operations import OLES_OPS

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# none does not work with the BabyDARTS
BABY_PRIMITIVES = [
    "skip_connect",
    "sep_conv_3x3",
    "dil_conv_3x3",
]


class BabyDARTSSearchSpace(
    SearchSpace,
    DrNASRegTermSupport,
    FairDARTSRegTermSupport,
    GradientMatchingScoreSupport,
):
    def __init__(self, **kwargs):  # type: ignore
        assert "layers" not in kwargs, "Layers parameter is hard coded to 1"
        assert "steps" not in kwargs, "Steps parameter is hard coded to 1"
        assert "multiplier" not in kwargs, "multiplier parameter is hard coded to 1"

        primitives = (
            BABY_PRIMITIVES if "primitives" not in kwargs else kwargs["primitives"]
        )
        kwargs.pop("primitives", None)

        model = BabyDARTSSearchModel(
            layers=1, steps=1, multiplier=1, primitives=primitives, **kwargs
        ).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        return self.model.arch_parameters()  # type: ignore

    @property
    def beta_parameters(self) -> list[nn.Parameter]:
        return self.model.beta_parameters()

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        assert len(arch_parameters) == len(self.arch_parameters)
        assert arch_parameters[0].shape == self.arch_parameters[0].shape
        (
            self.model.alphas_normal.data,
            self.model.alphas_reduce.data,
        ) = arch_parameters
        self.model._arch_parameters = [
            self.model.alphas_normal,
            self.model.alphas_reduce,
        ]

    def discretize(self) -> nn.Module:
        return self.model._discretize()  # type: ignore

    def get_genotype(self) -> DARTSGenotype:
        return self.model.genotype()

    def preserve_grads(self) -> None:
        self.model.apply(preserve_grads)

    def update_gradient_matching_scores(
        self,
        early_stop: bool = False,
        early_stop_frequency: int = 20,
        early_stop_threshold: float = 0.4,
    ) -> None:
        partial_fn = partial(
            update_gradient_matching_scores,
            oles_ops=OLES_OPS,
            early_stop=early_stop,
            early_stop_frequency=early_stop_frequency,
            early_stop_threshold=early_stop_threshold,
        )
        self.model.apply(partial_fn)

    def get_drnas_anchors(self) -> list[torch.Tensor]:
        return [self.model.anchor_normal]

    def get_sampled_weights(self) -> list[nn.Parameter]:
        return self.model.sampled_weights

    def get_fair_darts_arch_parameters(self) -> list[torch.Tensor]:
        return self.get_sampled_weights()
