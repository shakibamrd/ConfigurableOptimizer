from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import torch

from confopt.oneshot.base_component import OneShotComponent


class BasePerturbator(OneShotComponent):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        sample_frequency: Literal["epoch", "step"],
    ) -> None:
        super().__init__()
        self.arch_parameters = arch_parameters
        self.perturbed_alphas: list[torch.Tensor] = arch_parameters

        assert sample_frequency in [
            "epoch",
            "step",
        ], "sample_frequency must be either 'epoch' or 'step'"
        self.sample_frequency = sample_frequency

    @abstractmethod
    def perturb_alphas(
        self, arch_parameters: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        pass

    def _perturb_and_update_alphas(self) -> None:  # type: ignore
        perturbed_alphas = self.perturb_alphas(self.arch_parameters)
        # print(sampled_alphas)
        if perturbed_alphas is not None:
            self.perturbed_alphas = perturbed_alphas

    def new_epoch(self) -> None:
        super().new_epoch()

    def new_step(self) -> None:  # type: ignore
        super().new_step()
