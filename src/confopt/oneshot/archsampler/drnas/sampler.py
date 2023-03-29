from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from confopt.oneshot.archsampler import BaseSampler


class DRNASSampler(BaseSampler):
    def sample_epoch(self, alphas: torch.Tensor) -> None:
        pass

    def sample_step(self, alphas: torch.Tensor) -> list[torch.Tensor]:
        sampled_alphas = []
        for alpha in alphas:
            sampled_alphas.append(self.sample(alpha))
        return sampled_alphas

    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        beta = F.elu(alpha) + 1
        weights = torch.distributions.dirichlet.Dirichlet(beta).rsample()
        return weights  # type: ignore
