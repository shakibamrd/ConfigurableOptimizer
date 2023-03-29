from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseSampler(ABC):
    @abstractmethod
    def sample_epoch(self, alphas: list[torch.Tensor]) -> list[torch.Tensor] | None:
        pass

    @abstractmethod
    def sample_step(self, alphas: list[torch.Tensor]) -> list[torch.Tensor] | None:
        pass

    @abstractmethod
    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        pass
