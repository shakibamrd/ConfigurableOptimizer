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


class SNASProfile(ProfileConfig):
    def __init__(
        self, is_partial_connection: bool = False, perturbation: str | None = None
    ) -> None:
        PROFILE_TYPE = "SNAS"
        super().__init__(PROFILE_TYPE)
        self.sampler_type = PROFILE_TYPE
        self.set_partial_connector(is_partial_connection)
        self.set_perturb(perturbation)

    def get_sampler_config(self) -> dict:
        snas_config = {
            "sample_frequency": "epoch",
            "temp_init": 1.0,
            "temp_min": 0.33,
            "temp_annealing": True,
            "total_epochs": 250,
        }
        return snas_config


class DRNAS(ProfileConfig):
    def __init__(
        self, is_partial_connection: bool = False, perturbation: str | None = None
    ) -> None:
        PROFILE_TYPE = "DRNAS"
        super().__init__(PROFILE_TYPE)
        self.sampler_type = PROFILE_TYPE
        self.set_partial_connector(is_partial_connection)
        self.set_perturb(perturbation)

    def get_sampler_config(self) -> dict:
        drnas_config = {
            "sample_frequency": "epoch",
        }
        return drnas_config
