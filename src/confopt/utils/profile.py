from __future__ import annotations

import torch

from confopt.oneshot import BaseSampler


class BaseProfile:
    def __init__(
        self,
        sampler: BaseSampler,
        partial_connector: torch.nn.Module | None = None,
        pertubration: BaseSampler | None = None,
    ) -> None:
        self.sampler = sampler
        self.partial_connector = partial_connector
        self.pertubration = pertubration
