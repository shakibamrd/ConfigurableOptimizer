from __future__ import annotations

import itertools
from typing import Any, Generator

import ConfigSpace
from nasbench import api
import numpy as np

from .search_space import NB1Shot1Space
from .utils import INPUT, OUTPUT, OUTPUT_NODE


class NB1Shot1Space3(NB1Shot1Space):
    def __init__(self) -> None:
        self.search_space_number = 3
        self.num_intermediate_nodes = 5
        super().__init__(
            search_space_number=self.search_space_number,
            num_intermediate_nodes=self.num_intermediate_nodes,
        )
        """
        SEARCH SPACE 3
        """
        self.num_parents_per_node = {
            "0": 0,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 2,
            "5": 2,
            "6": 2,
        }
        if sum(self.num_parents_per_node.values()) > 9:
            raise ValueError("Each nasbench cell has at most 9 edges.")

        self.test_min_error = 0.05338543653488159
        self.valid_min_error = 0.04847759008407593

    def create_nasbench_adjacency_matrix(self, parents: dict[str, Any]) -> np.ndarray:
        # Create nasbench compatible adjacency matrix
        adjacency_matrix = self._create_adjacency_matrix(
            parents, adjacency_matrix=np.zeros([7, 7]), node=OUTPUT_NODE
        )
        return adjacency_matrix

    def create_nasbench_adjacency_matrix_with_loose_ends(
        self, parents: dict[str, Any]
    ) -> np.ndarray:
        return self._create_adjacency_matrix_with_loose_ends(parents)

    def generate_adjacency_matrix_without_loose_ends(
        self,
    ) -> Generator[np.ndarray, None, None]:
        yield from self._generate_adjacency_matrix(
            adjacency_matrix=np.zeros([7, 7]), node=OUTPUT_NODE
        )

    def generate_with_loose_ends(self) -> Generator[np.ndarray, None, None]:
        for (
            parent_node_2,
            parent_node_3,
            parent_node_4,
            parent_node_5,
            output_parents,
        ) in itertools.product(
            *[
                itertools.combinations(list(range(int(node))), num_parents)
                for node, num_parents in self.num_parents_per_node.items()
            ][2:]
        ):
            parents = {
                "0": [],
                "1": [0],
                "2": parent_node_2,
                "3": parent_node_3,
                "4": parent_node_4,
                "5": parent_node_5,
                "6": output_parents,
            }
            adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(
                parents
            )
            yield adjacency_matrix

    def objective_function(
        self,
        nasbench: api.NASBench,
        config: ConfigSpace.configuration_space.ConfigurationSpace,
        budget: int = 108,
    ) -> tuple:
        adjacency_matrix, node_list = super().convert_config_to_nasbench_format(config)
        node_list = [INPUT, *node_list, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench_data = nasbench.query(model_spec, epochs=budget)
        return nasbench_data["validation_accuracy"], nasbench_data["training_time"]
