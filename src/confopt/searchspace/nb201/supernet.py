from __future__ import annotations

from torch import nn

from confopt.searchspace.common.base_search import SearchSpace

from .core import NB201SearchModel


class NASBench201SearchSpace(SearchSpace):
    def __init__(self, *args, **kwargs):  # type: ignore
        model = NB201SearchModel(*args, **kwargs)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        return [self.model.arch_parameters]  # type: ignore
