from __future__ import annotations

from confopt.oneshot import BaseSampler


class BaseProfile:
    def __init__(
        self,
        samplers: list[BaseSampler],
    ) -> None:
        self.samplers = samplers
