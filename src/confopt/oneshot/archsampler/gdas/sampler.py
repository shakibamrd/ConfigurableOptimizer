from __future__ import annotations

from typing import Literal

import torch

from confopt.oneshot.archsampler import BaseSampler


class GDASSampler(BaseSampler):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        sample_frequency: Literal["epoch", "step"] = "step",
        tau_min: float = 0.1,
        tau_max: float = 10,
        total_epochs: int = 250,
    ) -> None:
        super().__init__(
            arch_parameters=arch_parameters, sample_frequency=sample_frequency
        )

        self.tau_min = torch.Tensor([tau_min])
        self.tau_max = torch.Tensor([tau_max])
        self.total_epochs = total_epochs

        self.tau_curr = self.tau_max - (self.tau_max - self.tau_min) * self._epoch / (
            self.total_epochs - 1
        )

    def set_taus(self, tau_min: float, tau_max: float) -> None:
        self.tau_min = torch.Tensor([tau_min])  # type: ignore
        self.tau_max = torch.Tensor([tau_max])  # type: ignore

    def set_total_epochs(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs

    def _sample_and_update_alphas(self) -> None:  # type: ignore
        sampled_alphas = self.sample_alphas(self.arch_parameters)
        hardwts = sampled_alphas[0]
        # print(sampled_alphas)
        if sampled_alphas is not None:
            self.sampled_alphas = hardwts

    def sample_alphas(
        self, arch_parameters: list[torch.Tensor]
    ) -> list[list[torch.Tensor]]:
        sampled_hardwt = []
        sampled_indexes = []
        for alpha in arch_parameters:
            sampled_alpha = self.sample(alpha)
            hardwt, index = sampled_alpha[0], sampled_alpha[1]
            sampled_hardwt.append(hardwt)
            sampled_indexes.append(index)
        return [sampled_hardwt, sampled_indexes]

    def sample(self, alpha: torch.Tensor) -> tuple:
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

        return hardwts, index

    def new_epoch(self) -> None:
        self.tau_curr = self.tau_max - (self.tau_max - self.tau_min) * self._epoch / (
            self.total_epochs - 1
        )

        super().new_epoch()
