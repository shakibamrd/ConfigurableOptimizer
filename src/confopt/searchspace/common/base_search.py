from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn  # noqa: PLR0402


class SearchSpace(nn.Module, ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    @abstractmethod
    def arch_parameters(self) -> list[nn.Parameter]:
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)  # type: ignore
