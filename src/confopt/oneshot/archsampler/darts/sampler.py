from __future__ import annotations

import torch

from confopt.oneshot.sampler import BaseSampler


class DARTSSampler(BaseSampler):
    def sample_epoch(self, alphas: list[torch.Tensor]) -> list[torch.Tensor] | None:
        pass

    def sample_step(self, alphas: list[torch.Tensor]) -> list[torch.Tensor] | None:
        return alphas

    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        return alpha
