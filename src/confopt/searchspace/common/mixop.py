from __future__ import annotations

import torch
from torch import nn

__all__ = ["OperationChoices"]


class OperationChoices(nn.Module):
    def __init__(self, ops: list[nn.Module]):
        super().__init__()
        self.ops = ops

    def forward(self, x: torch.Tensor, alphas: list[torch.Tensor]) -> torch.Tensor:
        assert len(alphas) == len(
            self.ops
        ), "Number of operations and architectural weights do not match"
        states = [op(x) * alpha for op, alpha in zip(self.ops, alphas)]
        return sum(states)  # type: ignore
