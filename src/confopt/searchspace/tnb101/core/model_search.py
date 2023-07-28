from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from confopt.searchspace.common import OperationChoices

from .operations import OPS, TRANS_NAS_BENCH_101

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TNB101SearchModel(nn.Module):
    def __init__(
        self,
        C: int = 16,
        stride: int = 1,
        max_nodes: int = 4,
        num_classes: int = 10,
        op_names: list[str] = TRANS_NAS_BENCH_101,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        """Initialize a TransNasBench-101 network consisting of one cell
        Args:
            C_in: in channel
            C_out: out channel
            stride: 1 or 2
            max_nodes: total amount of nodes in one cell
            num_classes: classes
            op_names: operations for cell structure
            affine: used for torch.nn.BatchNorm2D
            track_running_stats: used for torch.nn.BatchNorm2D.
        """
        super().__init__()
        assert stride == 1 or stride == 2, f"invalid stride {stride}"

        self.C = C
        self.stride = stride

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        self.op_names = deepcopy(op_names)
        self.max_nodes = max_nodes

        layer_channels = [C * 2, C * 2, C * 4, C * 4, C * 8]

        self.cells = nn.ModuleList()
        C_prev = C
        for _index, C_curr in enumerate(layer_channels):
            cell = TNB101SearchCell(
                C_prev, C_curr, stride, max_nodes, op_names, affine, track_running_stats
            ).to(DEVICE)
            self.cells.append(cell)
            C_prev = cell.C_out
        self.num_edge = len(self.cells[0].edges)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU())
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._arch_parameters = nn.Parameter(
            1e-3 * torch.randn(self.num_edge, len(op_names))  # type: ignore
        ).to(DEVICE)

    def arch_parameters(self) -> nn.Parameter:
        return self._arch_parameters

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alphas = nn.functional.softmax(self._arch_parameters, dim=-1)

        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature, alphas)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits


class TNB101SearchCell(nn.Module):
    expansion = 1

    def __init__(
        self,
        C_in: int = 16,
        C_out: int = 8,
        stride: int = 2,
        max_nodes: int = 4,
        op_names: list[str] = TRANS_NAS_BENCH_101,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """Initialize a TransNasBench-101 cell
        Args:
            C_in: in channel
            C_out: out channel
            stride: 1 or 2
            max_nodes: total amount of nodes in one cell
            op_names: operations for cell structure
            affine: used for torch.nn.BatchNorm2D
            track_running_stats: used for torch.nn.BatchNorm2D.
        """
        super().__init__()
        assert stride == 1 or stride == 2, f"invalid stride {stride}"

        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = f"{i}<-{j}"
                if j == 0:
                    xlists = nn.ModuleList(
                        [
                            OPS[op_name](
                                C_in, C_out, stride, affine, track_running_stats
                            )  # type: ignore
                            for op_name in op_names
                        ]
                    )
                else:
                    xlists = nn.ModuleList(
                        [
                            OPS[op_name](
                                C_in, C_out, 1, affine, track_running_stats
                            )  # type: ignore
                            for op_name in op_names
                        ]
                    )
                self.edges[node_str] = OperationChoices(ops=xlists)
        self.edge_keys = sorted(self.edges.keys())
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges: int = len(self.edges)

    def forward(self, inputs: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                weights = alphas[self.edge2index[node_str]]
                inter_nodes.append(self.edges[node_str](nodes[j], weights))
            nodes.append(sum(inter_nodes))
        return nodes[-1]
