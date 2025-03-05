from __future__ import annotations

from functools import partial
from typing import Any, Literal

import torch
from torch import nn

from confopt.searchspace.common import (
    ArchAttentionSupport,
    DrNASRegTermSupport,
    FairDARTSRegTermSupport,
    FLOPSRegTermSupport,
    GradientMatchingScoreSupport,
    GradientStatsSupport,
    InsertCellSupport,
    LambdaDARTSSupport,
    LayerAlignmentScoreSupport,
    OperationStatisticsSupport,
    PerturbationArchSelectionSupport,
    SearchSpace,
)
from confopt.searchspace.darts.core.operations import OLES_OPS
from confopt.utils import update_gradient_matching_scores

from .core import DARTSSearchModel
from .core.genotypes import DARTSGenotype
from .core.model_search import preserve_grads

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DARTSSearchSpace(
    SearchSpace,
    ArchAttentionSupport,
    GradientMatchingScoreSupport,
    OperationStatisticsSupport,
    LambdaDARTSSupport,
    LayerAlignmentScoreSupport,
    DrNASRegTermSupport,
    FLOPSRegTermSupport,
    PerturbationArchSelectionSupport,
    InsertCellSupport,
    GradientStatsSupport,
    FairDARTSRegTermSupport,
):
    def __init__(self, *args, **kwargs):  # type: ignore
        """DARTS Search Space for Neural Architecture Search.

        This class represents a search space for neural architecture search using
        DARTS (Differentiable Architecture Search).

        Args:
            *args: Variable length positional arguments. These arguments will be
                passed to the constructor of the internal DARTSSearchModel.
            **kwargs: Variable length keyword arguments. These arguments will be
                passed to the constructor of the internal DARTSSearchModel.

        Keyword Args:
            C (int): Number of channels.
            num_classes (int): Number of output classes.
            layers (int): Number of layers in the network.
            criterion (nn.modules.loss._Loss): Loss function.
            steps (int): Number of steps in the search space cell.
            multiplier (int): Multiplier for channels in the cells.
            stem_multiplier (int): Stem multiplier for channels.
            edge_normalization (bool): Whether to use edge normalization.

        Methods:
            - arch_parameters: Get architectural parameters.
            - beta_parameters: Get beta parameters.
            - set_arch_parameters(arch_parameters): Set architectural parameters

        Example:
            You can create an instance of DARTSSearchSpace with optional arguments as:
            >>> search_space = DARTSSearchSpace(
                                    C=32,
                                    num_classes=20,
                                    layers=10,
                                    criterion=nn.CrossEntropyLoss(),
                                    steps=5,
                                    multiplier=3,
                                    stem_multiplier=2,
                                    edge_normalization=True,
                                    dropout=0.2)
        """
        model = DARTSSearchModel(*args, **kwargs).to(DEVICE)
        super().__init__(model)

    @property
    def arch_parameters(self) -> list[nn.Parameter]:
        """Get a list containing the alpha parameters of the model
        Return:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.
        """
        return self.model.arch_parameters()  # type: ignore

    @property
    def beta_parameters(self) -> list[nn.Parameter]:
        """Get a list containing the beta parameters of the model.

        Returns:
            list[nn.Parameter]: A list containing the beta parameters for the model.
        """
        return self.model.beta_parameters()

    def set_arch_parameters(self, arch_parameters: list[nn.Parameter]) -> None:
        """Set the architectural parameters of the model.

        Args:
            arch_parameters (list[nn.Parameter]): A list of architectural parameters
            (alpha values) to set.

        Note:
            This method sets the architectural parameters of the model to the provided
            values.
        """
        assert len(arch_parameters) == len(self.arch_parameters)
        assert arch_parameters[0].shape == self.arch_parameters[0].shape
        (
            self.model.alphas_normal.data,
            self.model.alphas_reduce.data,
        ) = arch_parameters
        self.model._arch_parameters = [
            self.model.alphas_normal,
            self.model.alphas_reduce,
        ]

    def get_cell_types(self) -> list[str]:
        return ["normal", "reduce"]

    def discretize(self) -> nn.Module:
        return self.model.discretize()  # type: ignore

    def get_genotype(self) -> DARTSGenotype:
        return self.model.genotype()  # type: ignore

    def preserve_grads(self) -> None:
        self.model.apply(preserve_grads)

    def update_gradient_matching_scores(
        self,
        early_stop: bool = False,
        early_stop_frequency: int = 20,
        early_stop_threshold: float = 0.4,
    ) -> None:
        partial_fn = partial(
            update_gradient_matching_scores,
            oles_ops=OLES_OPS,
            early_stop=early_stop,
            early_stop_frequency=early_stop_frequency,
            early_stop_threshold=early_stop_threshold,
        )
        self.model.apply(partial_fn)

    def get_mean_layer_alignment_score(self) -> tuple[float, float]:
        return self.model.get_mean_layer_alignment_score()

    def get_first_and_last_layer_alignment_score(self) -> tuple[float, float]:
        return self.model.get_mean_layer_alignment_score(only_first_and_last=True)

    def get_num_skip_ops(self) -> dict[str, int]:
        alphas_normal, alphas_reduce = self.model.arch_parameters()

        try:
            index_of_skip = self.model.primitives.index("skip_connect")
        except ValueError:
            return {"skip_connections/normal": -1, "skip_connections/reduce": -1}

        try:
            index_of_none = self.model.primitives.index("none")
        except ValueError:
            index_of_none = -1

        def count_skip(alphas: torch.Tensor) -> int:
            if index_of_none == -1:
                return (alphas.argmax(dim=1) == index_of_skip).sum().item()

            tmp_alphas = alphas.clone()
            tmp_alphas[:, index_of_none] = float("-inf")
            return (tmp_alphas.argmax(dim=1) == index_of_skip).sum().item()

        stats = {
            "skip_connections/normal": count_skip(alphas_normal),
            "skip_connections/reduce": count_skip(alphas_reduce),
        }

        return stats

    def get_drnas_anchors(self) -> list[torch.Tensor]:
        return [self.model.anchor_normal, self.model.anchor_reduce]

    def get_weighted_flops(self) -> torch.Tensor:
        return self.model.get_weighted_flops()

    def get_num_ops(self) -> int:
        return self.model.num_ops

    def get_num_edges(self) -> int:
        return self.model.num_edges

    def get_num_nodes(self) -> int:
        return self.model.num_nodes

    def get_candidate_flags(self, cell_type: Literal["normal", "reduce"]) -> list:
        if self.topology is None:
            self.set_topology(False)

        if self.topology:
            return self.model.candidate_flags_edge[cell_type]

        return self.model.candidate_flags[cell_type]

    def get_edges_at_node(self, selected_node: int) -> list:
        return self.model.nid2eids[selected_node]

    def remove_from_projected_weights(
        self,
        selected_edge: int,
        selected_op: int | None,
        cell_type: Literal["normal", "reduce"] = "normal",
    ) -> None:
        if self.topology is None:
            self.set_topology(False)

        self.model.remove_from_projected_weights(
            cell_type, selected_edge, selected_op, self.topology
        )

    def mark_projected_operation(
        self,
        selected_edge: int,
        selected_op: int,
        cell_type: Literal["normal", "reduce"],
    ) -> None:
        self.model.mark_projected_op(selected_edge, selected_op, cell_type)

    def mark_projected_edge(
        self,
        selected_node: int,
        selected_edges: list[int],
        cell_type: Literal["normal", "reduce"] = "normal",  # type: ignore
    ) -> None:
        self.model.mark_projected_edges(selected_node, selected_edges, cell_type)

    def set_projection_mode(self, value: bool) -> None:
        self.model.projection_mode = value

    def set_projection_evaluation(self, value: bool) -> None:
        self.model.projection_evaluation = value

    def is_topology_supported(self) -> bool:
        return True

    def get_max_input_edges_at_node(self, selected_node: int) -> int:  # noqa: ARG002
        return 2

    def insert_new_cells(self, num_of_cells: int) -> None:
        self.model.insert_new_cells(num_of_cells)

    def create_new_cell(self, position: int) -> nn.Module:
        return self.model.create_new_cell(position)

    def get_fair_darts_arch_parameters(self) -> list[torch.Tensor]:
        return self.get_sampled_weights()

    def get_projected_arch_parameters(self) -> list[torch.Tensor]:
        projected_arch_params = [
            self.model.get_projected_weights("normal"),
            self.model.get_projected_weights("reduce"),
        ]
        return projected_arch_params


class DARTSSearchSpaceShallowWide(DARTSSearchSpace):
    """DARTS search space with a shallow and wide architecture
    Number of cells: 4
    Inital Channels: 18
    1057132 parameters
    Normal cell -> Reduction cell -> Normal cell -> Reduction cell.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, C=18, layers=4)


class DARTSSearchSpaceDeepNarrow(DARTSSearchSpace):
    """DARTS search space with a shallow and wide architecture
    Number of cells: 16
    Inital Channels: 8
    1080382 parameters
    Reduction cells at the 5th and 11th position (index starting from 0).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, C=8, layers=16)


class DARTSSearchSpaceSingleCell(DARTSSearchSpace):
    """DARTS search space with a single cell
    Inital Channels: 26
    1048968 parameters
    Number of steps: 8
    Number of edges in the cell: 44.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, C=26, layers=1, steps=8)
