from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from confopt.searchspace.common import OperationChoices

from . import operations as ops
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
        dataset: str = "class_object",
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

        self.op_names = deepcopy(op_names)
        self.max_nodes = max_nodes
        self.n_modules = 5
        self.blocks_per_module = [2] * self.n_modules

        self.module_stages = [
            "r_stage_1",
            "n_stage_1",
            "r_stage_2",
            "n_stage_2",
            "r_stage_3",
        ]

        self.cells = nn.ModuleList()
        C_in, C_out = C, C
        for idx, stage in enumerate(self.module_stages):
            for i in range(self.blocks_per_module[idx]):
                downsample = self._is_reduction_stage(stage) and i % 2 == 0
                if downsample:
                    C_out *= 2
                cell = TNB101SearchCell(
                    C_in,
                    C_out,
                    stride,
                    max_nodes,
                    op_names,
                    affine,
                    track_running_stats,
                    downsample,
                ).to(DEVICE)
                self.cells.append(cell)
                C_in = C_out
        self.num_edge = len(self.cells[0].edges)

        if dataset == "jigsaw":
            self.num_classes = 1000
        elif dataset == "class_object":
            self.num_classes = 100
        elif dataset == "class_scene":
            self.num_classes = 63
        else:
            self.num_classes = num_classes

        self.stem = self._get_stem_for_task(dataset)
        self.decoder = self._get_decoder_for_task(dataset, C_out)
        self.op_names = deepcopy(op_names)
        self.max_nodes = max_nodes

        self.lastact = nn.Sequential(nn.BatchNorm2d(num_classes), nn.ReLU())
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_classes, num_classes)

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

        out = self.decoder(feature)

        out = self.lastact(out)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits

    def _get_stem_for_task(self, task: str) -> nn.Module:
        if task == "jigsaw":
            return ops.StemJigsaw(C_out=self.C)
        if task in ["class_object", "class_scene"]:
            return ops.Stem(C_out=self.C)
        if task == "autoencoder":
            return ops.Stem(C_out=self.C)
        return ops.Stem(C_in=3, C_out=self.C)

    def _get_decoder_for_task(self, task: str, n_channels: int) -> nn.Module:
        if task == "jigsaw":
            return ops.SequentialJigsaw(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels * 9, self.num_classes),
            )
        if task in ["class_object", "class_scene"]:
            return ops.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(n_channels, self.num_classes),
            )
        if task == "autoencoder":
            if self.use_small_model:
                return ops.GenerativeDecoder((64, 32), (256, 2048))  # Short
            return ops.GenerativeDecoder((512, 32), (512, 2048))  # Full TNB

        return ops.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_channels, self.num_classes),
        )

    def _is_reduction_stage(self, stage: str) -> bool:
        return "r_stage" in stage


class TNB101SearchCell(nn.Module):
    expansion = 1

    def __init__(
        self,
        C_in: int = 16,
        C_out: int = 16,
        stride: int = 1,
        max_nodes: int = 4,
        op_names: list[str] = TRANS_NAS_BENCH_101,
        affine: bool = True,
        track_running_stats: bool = True,
        downsample: bool = True,
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

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = f"{i}<-{j}"
                if j == 0:
                    if downsample:
                        stride = 2
                    stride = 1
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
                                C_out, C_out, 1, affine, track_running_stats
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
