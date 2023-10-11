from __future__ import annotations

from .profile_config import ProfileConfig


class DartsProfile(ProfileConfig):
    def __init__(
        self, is_partial_connection: bool = False, perturbation: str | None = None
    ) -> None:
        PROFILE_TYPE = "DARTS"
        super().__init__(PROFILE_TYPE)
        self.sampler_type = PROFILE_TYPE
        self.set_partial_connector(is_partial_connection)
        self.set_perturb(perturbation)

    def get_sampler_config(self) -> dict:
        darts_config = {"sample_frequency": "epoch"}
        return darts_config


class GDASProfile(ProfileConfig):
    def __init__(
        self, is_partial_connection: bool = False, perturbation: str | None = None
    ) -> None:
        PROFILE_TYPE = "DARTS"
        super().__init__(PROFILE_TYPE)
        self.sampler_type = PROFILE_TYPE
        self.set_partial_connector(is_partial_connection)
        self.set_perturb(perturbation)

    def get_sampler_config(self) -> dict:
        gdas_config = {"sample_frequency": "epoch", "tau_min": 0.1, "tau_max": 10}
        return gdas_config
