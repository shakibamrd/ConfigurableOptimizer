from __future__ import annotations

import torch
from torch import nn

from confopt.oneshot.archsampler import BaseSampler

__all__ = ["OperationChoices"]


class OperationChoices(nn.Module):
    def __init__(self, ops: list[nn.Module], is_reduction_cell: bool = False) -> None:
        super().__init__()
        self.ops = ops
        self.is_reduction_cell = is_reduction_cell

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        assert len(alphas) == len(
            self.ops
        ), "Number of operations and architectural weights do not match"
        states = [op(x) * alpha for op, alpha in zip(self.ops, alphas)]
        return sum(states)  # type: ignore


class OperationBlock:
    def __init__(
        self, ops: list[nn.Module], is_reduction_cell: bool, arch_sampler: BaseSampler
    ) -> None:
        self.ops = ops
        self.is_reduction_cell = is_reduction_cell
        self.arch_sampler = arch_sampler

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        alphas_index = 1 if self.is_reduction_cell else 0

        alphas = self.arch_sampler.sampled_alphas[alphas_index]

        # TODO: check if alphas are one-hot, and then apply only the selected op
        states = [op(x) * alpha for op, alpha in zip(self.ops, alphas)]

        return sum(states)  # type: ignore
