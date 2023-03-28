import unittest

import torch
import torch.nn as nn  # noqa: PLR0402

from confopt.searchspace import DARTSSearchSpace, NASBench201SearchSpace
from confopt.searchspace.darts.core.model_search import Cell as DARTSSearchCell
from confopt.searchspace.nb201.core import NAS201SearchCell
from confopt.searchspace.nb201.core.operations import ReLUConvBN, ResNetBasicblock
from utils import get_modules_of_type


class TestNASBench201SearchSpace(unittest.TestCase):
    def test_arch_parameters(self) -> None:
        search_space = NASBench201SearchSpace()
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 1
        assert isinstance(arch_params[0], nn.Parameter)

    def test_forward_pass(self) -> None:
        search_space = NASBench201SearchSpace()
        x = torch.randn(2, 3, 32, 32)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 64])
        assert out[1].shape == torch.Size([2, 10])

    def test_supernet_init(self) -> None:
        C = 32
        N = 3
        num_classes = 11
        search_space = NASBench201SearchSpace(C=C, N=N, num_classes=num_classes)

        search_cells = get_modules_of_type(search_space.model, NAS201SearchCell)
        assert len(search_cells) == N*3

        resnet_cells = get_modules_of_type(search_space.model, ResNetBasicblock)
        assert len(resnet_cells) == 2

        first_cell_edge_ops = search_cells[0].edges["1<-0"].ops
        for op in first_cell_edge_ops:
            if isinstance(op, ReLUConvBN):
                assert op.op[1].in_channels == C
                assert op.op[1].out_channels == C

        x = torch.randn(2, 3, 32, 32)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])

class TestDARTSSearchSpace(unittest.TestCase):
    def test_arch_parameters(self) -> None:
        search_space = DARTSSearchSpace()
        arch_params = search_space.arch_parameters
        assert len(arch_params) == 2
        assert isinstance(arch_params[0], nn.Parameter)
        assert isinstance(arch_params[1], nn.Parameter)

    def test_forward_pass(self) -> None:
        search_space = DARTSSearchSpace()
        x = torch.randn(2, 3, 64, 64)

        out = search_space(x)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], torch.Tensor)
        assert out[0].shape == torch.Size([2, 256])
        assert out[1].shape == torch.Size([2, 10])

    def test_supernet_init(self) -> None:
        C = 32
        layers = 6
        num_classes = 11
        search_space = DARTSSearchSpace(C=C, layers=layers, num_classes=num_classes)

        search_cells = get_modules_of_type(search_space.model, DARTSSearchCell)
        assert len(search_cells) == layers

        reduction_cells = [cell for cell in search_cells if cell.reduction is True]
        assert len(reduction_cells) == 2

        x = torch.randn(2, 3, 32, 32)
        out, logits = search_space(x)

        assert logits.shape == torch.Size([2, num_classes])

if __name__ == "__main__":
    unittest.main()