from __future__ import annotations

from typing import Literal

import torch

from confopt.oneshot.base_component import OneShotComponent


class Dropout(OneShotComponent):
    """A class representing a dropout operation for architectural parameters."""

    def __init__(
        self,
        p: float,
        p_min: float | None = None,
        anneal_frequency: Literal["epoch", "step", None] = None,
        anneal_type: Literal["linear", "cosine", None] = None,
        seed: int = 1,
    ) -> None:
        """Instantiate a dropout class.

        Args:
            p (float): The initial dropout probability, must be in the range [0, 1).
            p_min (float, optional): The dropout probability to decay to.
            Must be in the range [0, 1)
            anneal_frequency (str, optional): The frequency at which to anneal the
            dropout probability. Defaults to None.
            anneal_type (str, optional): Type of probability annealing to be used.
            Defaults to None.
            seed (int, optional): The seed for random number generation. Defaults to 1.
        """
        super().__init__()
        assert p >= 0
        assert p < 1
        if p_min:
            assert p_min >= 0
            assert p_min < 1
            assert p_min < p
        assert anneal_frequency in ["epoch", "step", None]
        assert bool(anneal_frequency) == bool(anneal_type)

        self._p_init = p
        self._p_min = p_min
        self._anneal_frequency = anneal_frequency
        self._seed = seed

        self.p = self._p_init

    def apply_mask(self, parameters: torch.Tensor) -> torch.Tensor:
        r"""This function masks the parameters based on the drop probability p.
        Additionally, the values are scaled by the factor of :math:`\frac{1}{1-p}`
        in order to ensure that during evaluation the module simply computes an
        identity function.
        """
        random = torch.rand_like(parameters)
        dropout_mask = random >= self.p
        scale_mask = torch.ones_like(parameters)
        return scale_mask * dropout_mask * parameters

    def _anneal_probability(self) -> None:
        """This function decays the dropout probability to the goal probability."""

    def new_epoch(self) -> None:
        super().new_epoch()
        if self._anneal_frequency == "epoch":
            self._anneal_probability()

    def new_step(self) -> None:
        super().new_step()
        if self._anneal_frequency == "step":
            self._anneal_probability()
