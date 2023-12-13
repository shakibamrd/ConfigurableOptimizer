from __future__ import annotations

from typing import Literal

import torch

from confopt.oneshot.archsampler import BaseSampler


class DARTSSampler(BaseSampler):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        sample_frequency: Literal["epoch", "step"] = "step",
    ) -> None:
        super().__init__(
            arch_parameters=arch_parameters, sample_frequency=sample_frequency
        )

    def sample_alphas(
        self, arch_parameters: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        return torch.nn.functional.softmax(arch_parameters)
