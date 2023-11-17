from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812

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
        for i, alphas in enumerate(arch_parameters):
            arch_parameters[i] = F.softmax(alphas, dim=-1)
        return arch_parameters
