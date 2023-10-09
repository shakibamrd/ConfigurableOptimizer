##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
########################################################
# DARTS: Differentiable Architecture Search, ICLR 2019 #
########################################################
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .cells import NAS201SearchCell as SearchCell
from .genotypes import Structure
from .operations import NAS_BENCH_201, ResNetBasicblock


class NB201SearchModel(nn.Module):
    """Implementation of Nasbench201 search space.

    Args:
        C (int, optional): Number of channels. Defaults to 16.
        N (int, optional): Number of layers. Defaults to 5.
        max_nodes (int, optional): Maximum number of nodes in a cell. Defaults to 4.
        num_classes (int, optional): Number of classes of the dataset. Defaults to 10.
        steps (int, optional): Number of steps. Defaults to 3.
        search_space (list[str], optional): List of search space options. Defaults to
        NAS_BENCH_201.
        affine (bool, optional): Whether to use affine transformations in BatchNorm in
        cells. Defaults to False.
        track_running_stats (bool, optional): Whether to track running statistics in
        BatchNorm in cells. Defaults to False.
        edge_normalization (bool, optional): Whether to enable edge normalization for
        partial connection. Defaults to False.

    Attributes:
        stem (nn.Sequential): Stem network composed of Conv2d and BatchNorm2d layers.
        cells (nn.ModuleList): List of cells in the search space.
        op_names (list[str]): List of operation names.
        edge2index (dict[str, int]): Dictionary mapping edge names to indices.
        lastact (nn.Sequential): Sequential layer consisting of BatchNorm2d and ReLU.
        global_pooling (nn.AdaptiveAvgPool2d): Global pooling layer.
        classifier (nn.Linear): Linear classifier layer.
        arch_parameters (nn.Parameter): Parameter for architecture alpha values.
        beta_parameters (nn.Parameter): Parameter for beta values.

    Note:
        This is a custom neural network model with various hyperparameters and
        architectural choices.
    """

    def __init__(
        self,
        C: int = 16,
        N: int = 5,
        max_nodes: int = 4,
        num_classes: int = 10,
        steps: int = 3,
        search_space: list[str] = NAS_BENCH_201,
        affine: bool = False,
        track_running_stats: bool = False,
        edge_normalization: bool = False,
    ):
        super().__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self._steps = steps
        self.edge_normalization = edge_normalization
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for _index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(  # type: ignore
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    search_space,
                    affine,
                    track_running_stats,
                )

                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (
                        num_edge == cell.num_edges
                    ), f"invalid {num_edge} vs. {cell.num_edges}."
                    assert (
                        edge2index == cell.edge2index
                    ), f"invalid {num_edge} vs. {cell.num_edges}."
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index: dict[str, int] = edge2index  # type: ignore
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU())
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_parameters = nn.Parameter(
            1e-3 * torch.randn(num_edge, len(search_space))  # type: ignore
        )
        self.beta_parameters = nn.Parameter(
            1e-3 * torch.randn(num_edge)  # type: ignore
        )

    def get_weights(self) -> list[nn.Parameter]:
        """Get a list of learnable parameters in the model. (does not include alpha or
        beta parameters).

        Returns:
            list[nn.Parameter]: A list of learnable parameters in the model, including
            stem, cells, lastact, global_pooling, and classifier parameters.
        """
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self) -> list[torch.Tensor]:
        """Get a list containing the architecture parameters or alphas.

        Returns:
            list[torch.Tensor]: A list containing the architecture parameters, such as
            alpha values.
        """
        return [self.arch_parameters]

    def get_betas(self) -> list[torch.Tensor]:
        """Get a list containing the beta parameters of partial connection used for
        edge normalization.

        Returns:
            list[torch.Tensor]: A list containing the beta parameters for the model.
        """
        return [self.beta_parameters]

    def show_alphas(self) -> str:
        """Get a human-readable string representation of the architecture parameters
        (alphas).

        Returns:
            str: A string representing the architecture parameters after softmax
            operation.
        """
        with torch.no_grad():
            return "arch-parameters :\n{:}".format(
                nn.functional.softmax(self.arch_parameters, dim=-1).cpu()
            )

    def show_betas(self) -> str:
        """Get a human-readable string representation of the beta parameters.

        Returns:
            str: A string representing the beta parameters after softmax operation.
        """
        with torch.no_grad():
            return "beta-parameters:\n{:}".format(
                nn.functional.softmax(self.beta_parameters, dim=-1).cpu()
            )

    def get_message(self) -> str:
        """Get a human-readable message describing the model and its cells.

        Returns:
            str: A string message containing information about the model and its cells.
        """
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self) -> str:
        """Return a string containing extra information about the model.

        Returns:
            str: A string representation containing information about the model's class
            name, number of channels (C), maximum nodes (Max-Nodes), number of layers
            (N), and number of cells (L).
        """
        return "{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def genotype(self) -> Structure:
        """Get the genotype of the model, representing the architecture.

        Returns:
            Structure: An object representing the genotype of the model, which describes
            the architectural choices in terms of operations and connections between
            nodes.
        """
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = f"{i}<-{j}"
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    # betas = self.beta_parameters[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]  # type: ignore
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor to the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - The output tensor after the forward pass.
            - The logits tensor produced by the model.
        """
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)

        feature = self.stem(inputs)
        for _i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                if self.edge_normalization:
                    betas = torch.empty((0,)).to(alphas.device)
                    for v in range(1, self.max_nodes):
                        idx_nodes = []
                        for u in range(v):
                            node_str = f"{v}<-{u}"
                            idx_nodes.append(cell.edge2index[node_str])
                        beta_node_v = nn.functional.softmax(
                            self.beta_parameters[idx_nodes], dim=-1
                        )
                        betas = torch.cat([betas, beta_node_v], dim=0)

                    feature = cell(feature, alphas, betas)
                else:
                    feature = cell(feature, alphas)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits
