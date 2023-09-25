from __future__ import annotations

import torch

from confopt.oneshot import BaseSampler
from confopt.oneshot.archsampler.darts.sampler import DARTSSampler
from confopt.oneshot.partial_connector import PartialConnector
from confopt.searchspace import DARTSSearchSpace
from confopt.searchspace.common import OperationBlock, OperationChoices, SearchSpace


class Profile:
    def __init__(
        self,
        sampler: BaseSampler,
        edge_normalization: bool = False,
        partial_connector: PartialConnector | None = None,
        perturbation: BaseSampler | None = None,
    ) -> None:
        self.sampler = sampler
        self.edge_normalization = edge_normalization
        self.partial_connector = partial_connector
        self.perturbation = perturbation

    def adapt_search_space(self, search_space: SearchSpace) -> None:
        if hasattr(search_space.model, "edge_normalization"):
            search_space.model.edge_normalization = self.edge_normalization

        for _, module in search_space.named_modules(remove_duplicate=False):
            if isinstance(module, OperationChoices):
                module = self._initialize_operation_block(
                    module.ops, module.is_reduction_cell
                )
        search_space.components.append(self.sampler)
        if self.perturbation:
            search_space.components.append(self.perturbation)

    def _initialize_operation_block(
        self, ops: torch.nn.Module, is_reduction_cell: bool = False
    ) -> OperationBlock:
        op_block = OperationBlock(
            ops,
            is_reduction_cell,
            self.sampler,
            self.perturbation,
            self.partial_connector,
        )
        return op_block


if __name__ == "__main__":
    search_space = DARTSSearchSpace()
    sampler = DARTSSampler(search_space.arch_parameters)
    profile = Profile(sampler=sampler)
    profile.adapt_search_space(search_space=search_space)
    print("success")
