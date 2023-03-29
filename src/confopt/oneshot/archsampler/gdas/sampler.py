from __future__ import annotations

import torch

from confopt.oneshot.archsampler import BaseSampler


class GDASSampler(BaseSampler):
    def __init__(self) -> None:
        self.tau_min = None
        self.tau_max = None
        self.tau_curr = None
        self.total_epochs = 0
        self.epoch = 0

    def set_taus(self, tau_min: float, tau_max: float) -> None:
        self.tau_min = torch.Tensor([tau_min])  # type: ignore
        self.tau_max = torch.Tensor([tau_max])  # type: ignore

    def set_total_epochs(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs

    def sample_epoch(self, alphas: list[torch.Tensor]) -> list[torch.Tensor] | None:
        pass

    def sample_step(self, alphas: list[torch.Tensor]) -> list[torch.Tensor]:
        sampled_alphas = []
        for alpha in alphas:
            sampled_alphas.append(self.sample(alpha))
        return sampled_alphas

    def sample(self, alpha: torch.Tensor) -> torch.Tensor:
        tau = self.tau_curr.to(alpha.device)  # type: ignore
        while True:
            gumbels = -torch.empty_like(alpha, device=alpha.device).exponential_().log()
            logits = (alpha.log_softmax(dim=-1) + gumbels) / tau[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits, device=alpha.device).scatter_(
                -1, index, 1.0
            )
            hardwts = one_h - probs.detach() + probs
            if not (
                (torch.isinf(gumbels).any())
                or (torch.isinf(probs).any())
                or (torch.isnan(probs).any())
            ):
                break

        return hardwts

    def before_epoch(self) -> None:
        if self.tau_max is None:
            raise Exception("tau_max has to be set in GDASSampler")

        self.tau_curr = self.tau_max - (self.tau_max - self.tau_min) * self.epoch / (
            self.total_epochs - 1
        )
        self.epoch += 1
