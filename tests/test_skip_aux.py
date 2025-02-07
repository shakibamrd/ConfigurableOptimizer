from __future__ import annotations

import unittest
import torch

from confopt.oneshot.archsampler import DARTSSampler
from confopt.oneshot.partial_connector import PartialConnector
from confopt.oneshot.weightentangler import WeightEntangler
from confopt.searchspace import SearchSpace, NASBench201SearchSpace, DARTSSearchSpace
from confopt.train import SearchSpaceHandler

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestAuxilarySkipConnection(unittest.TestCase):
    def test_nb201_init_auxiliary(self) -> None:
        # NB201 does not have introduce any new parameters
        searchspace = NASBench201SearchSpace()
        self._test_vanilla_forward(searchspace)
        self._test_with_partial_connection(searchspace)
        self._test_with_weight_entanglement(searchspace)

    def test_darts_init_auxiliary(self) -> None:
        searchspace = DARTSSearchSpace()
        self._test_with_optimizer(searchspace)
        self._test_vanilla_forward(searchspace)
        self._test_with_partial_connection(searchspace)
        self._test_with_weight_entanglement(searchspace)

    def _test_with_optimizer(self, searchspace: SearchSpace) -> None:
        loss_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(searchspace.model_weight_parameters(), lr=1e-3)
        searchspace_handler = SearchSpaceHandler(
            sampler=DARTSSampler(searchspace.arch_parameters),
            use_auxiliary_skip_connection=True,
        )
        searchspace_handler.adapt_search_space(searchspace)

        # optimizer = torch.optim.SGD(searchspace.model_weight_parameters(), lr=1e-3)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(low=0, high=9, size=(2,)).to(DEVICE)

        _, logits = searchspace(x)
        loss = loss_criterion(logits, y)
        loss.backward()

        params_before_step = []
        for name, param in searchspace.named_parameters():
            if param.requires_grad is True and "aux_skip" in name:
                params_before_step.append(param.detach().clone())
                param.grad += torch.ones_like(param.grad) * 10

        optimizer.step()

        params_after_step = []
        for name, param in searchspace.named_parameters():
            if param.requires_grad is True and "aux_skip" in name:
                params_after_step.append(param.detach().clone())

        for param_before, param_after in zip(params_before_step, params_after_step):
            assert not torch.allclose(param_before, param_after)

    def _test_vanilla_forward(self, searchspace: SearchSpace) -> None:
        searchspace_handler = SearchSpaceHandler(
            sampler=DARTSSampler(searchspace.arch_parameters),
            use_auxiliary_skip_connection=True,
        )
        searchspace_handler.adapt_search_space(searchspace)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        _, logits = searchspace(x)

        assert logits.shape == torch.Size([2, 10])

    def _test_with_partial_connection(self, searchspace: SearchSpace) -> None:
        searchspace_handler = SearchSpaceHandler(
            sampler=DARTSSampler(searchspace.arch_parameters),
            partial_connector=PartialConnector(),
            use_auxiliary_skip_connection=True,
        )
        searchspace_handler.adapt_search_space(searchspace)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        _, logits = searchspace(x)

        assert logits.shape == torch.Size([2, 10])

    def _test_with_weight_entanglement(self, searchspace: SearchSpace) -> None:
        searchspace_handler = SearchSpaceHandler(
            sampler=DARTSSampler(searchspace.arch_parameters),
            weight_entangler=WeightEntangler(),
            use_auxiliary_skip_connection=True,
        )
        searchspace_handler.adapt_search_space(searchspace)
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        _, logits = searchspace(x)

        assert logits.shape == torch.Size([2, 10])


if __name__ == "__main__":
    unittest.main()
