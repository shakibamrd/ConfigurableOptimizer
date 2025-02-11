from __future__ import annotations

import unittest
import torch

from confopt.searchspace.common import LambdaReg
from confopt.searchspace import (
    DARTSSearchSpace,
    NASBench201SearchSpace,
    TransNASBench101SearchSpace,
    SearchSpace,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestLambda(unittest.TestCase):

    def _forward_and_backward_pass(self, model: SearchSpace) -> SearchSpace:
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(10, (2,)).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()

        _, preds = model(x)

        loss = criterion(preds, y)
        loss.backward()

        return model

    def test_get_arch_grads_darts(self) -> None:
        search_space = DARTSSearchSpace().to(DEVICE)
        search_space = self._forward_and_backward_pass(search_space)
        self._test_shapes(search_space, (6, 2), (14, 8))
        self._test_perturbations_disabled(search_space)
        self._test_perturbations_enabled(search_space)

    def test_get_arch_grads_nb201(self) -> None:
        search_space = NASBench201SearchSpace().to(DEVICE)
        search_space = self._forward_and_backward_pass(search_space)
        self._test_shapes(search_space, (15, 0), (6, 5))
        self._test_perturbations_disabled(search_space)

    def test_get_arch_grads_tnb101(self) -> None:
        search_space = TransNASBench101SearchSpace().to(DEVICE)
        search_space = self._forward_and_backward_pass(search_space)
        self._test_shapes(search_space, (10, 0), (6, 4))
        self._test_perturbations_disabled(search_space)

    def _test_shapes(self, search_space: SearchSpace, n_cells: tuple, grads_shape: tuple) -> None:
        grads_normal, grads_reduce = search_space.model.get_arch_grads()
        has_reduce = n_cells[1] > 0

        def assert_shape_correct(grads: list, n_cells: int, grads_shape: tuple) -> None:
            assert isinstance(grads, list)
            assert len(grads) == n_cells

            for grad in grads:
                assert grad.shape == (grads_shape[0] * grads_shape[1],)

        assert_shape_correct(grads_normal, n_cells[0], grads_shape)
        if has_reduce:
            assert_shape_correct(grads_reduce, n_cells[1], grads_shape)
        else:
            assert grads_reduce is None

        perts = search_space.get_perturbations()

        for p in perts:
            assert p.shape == grads_shape

    def _test_perturbations_disabled(self, model: SearchSpace) -> None:
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(10, (2,)).to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss()
        _, preds = model(x)

        loss = criterion(preds, y)
        loss.backward()

        old_grads = []
        for p in model.model_weight_parameters():
            if p.grad is not None:
                old_grads.append(p.grad.clone())

        model.add_lambda_regularization(x, y)

        for new_p, old_grad in zip(model.model_weight_parameters(), old_grads):
            if new_p.grad is not None:
                assert (new_p.grad == old_grad).all()

    def _test_perturbations_enabled(self, model: SearchSpace) -> None:
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        y = torch.randint(10, (2,)).to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss()
        _, preds = model(x)

        loss = criterion(preds, y)
        loss.backward()

        old_grads = []
        for p in model.model_weight_parameters():
            if p.grad is not None:
                old_grads.append(p.grad.clone())

        model.lambda_reg.enabled = True
        model.add_lambda_regularization(x, y)

        for new_p, old_grad in zip(model.model_weight_parameters(), old_grads):
            if new_p.grad is not None:
                assert (new_p.grad != old_grad).any()


if __name__ == "__main__":
    unittest.main()
