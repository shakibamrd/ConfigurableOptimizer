from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn  # noqa: PLR0402

from confopt.oneshot.base_component import OneShotComponent


class SearchSpace(nn.Module, ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.components: list[OneShotComponent] = []

    @property
    @abstractmethod
    def arch_parameters(self) -> list[nn.Parameter]:
        pass

    @abstractmethod
    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        pass

    def set_sample_function(self, sample_function: Callable) -> None:
        self.model.sample = sample_function

    def model_weight_parameters(self) -> list[nn.Parameter]:
        all_parameters = set(self.model.parameters())
        arch_parameters = set(self.arch_parameters)

        if hasattr(self, "beta_parameters"):
            beta_params = set(self.beta_parameters)
            return list(all_parameters - arch_parameters - beta_params)

        return list(all_parameters - arch_parameters)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)  # type: ignore

    def new_epoch(self) -> None:
        for component in self.components:
            component.new_epoch()

    def new_step(self) -> None:
        for component in self.components:
            component.new_step()
