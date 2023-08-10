from __future__ import annotations

from confopt.oneshot import BaseSampler


class BaseProfile:
    def __init__(
        self,
        sampler: BaseSampler,
    ) -> None:
        self.sampler = sampler
