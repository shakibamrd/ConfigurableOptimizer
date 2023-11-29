from __future__ import annotations

from .profile_config import ProfileConfig


class DartsProfile(ProfileConfig):
    def __init__(
        self,
        is_partial_connection: bool = False,
        perturbation: str | None = None,
        sampler_sample_frequency: str = "epoch",
        perturbator_sample_frequency: str = "epoch",
    ) -> None:
        PROFILE_TYPE = "DARTS"
        super().__init__(
            PROFILE_TYPE,
            is_partial_connection,
            perturbation,
            sampler_sample_frequency,
            perturbator_sample_frequency,
        )

    def get_sampler_config(self) -> dict:
        darts_config = {"sample_frequency": self.sampler_sample_frequency}
        return darts_config


class GDASProfile(ProfileConfig):
    def __init__(
        self,
        is_partial_connection: bool = False,
        perturbation: str | None = None,
        sampler_sample_frequency: str = "epoch",
        perturbator_sample_frequency: str = "epoch",
        tau_min: float = 0.1,
        tau_max: float = 10,
    ) -> None:
        PROFILE_TYPE = "GDAS"
        super().__init__(
            PROFILE_TYPE,
            is_partial_connection,
            perturbation,
            sampler_sample_frequency,
            perturbator_sample_frequency,
        )
        self.tau_min = tau_min
        self.tau_max = tau_max

    def get_sampler_config(self) -> dict:
        gdas_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
        }
        return gdas_config


class SNASProfile(ProfileConfig):
    def __init__(
        self,
        is_partial_connection: bool = False,
        perturbation: str | None = None,
        sampler_sample_frequency: str = "epoch",
        perturbator_sample_frequency: str = "epoch",
        temp_init: float = 1.0,
        temp_min: float = 0.33,
        temp_annealing: bool = True,
        total_epochs: int = 250,
    ) -> None:
        PROFILE_TYPE = "SNAS"
        super().__init__(
            PROFILE_TYPE,
            is_partial_connection,
            perturbation,
            sampler_sample_frequency,
            perturbator_sample_frequency,
        )
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.temp_annealing = temp_annealing
        self.total_epochs = total_epochs

    def get_sampler_config(self) -> dict:
        snas_config = {
            "sample_frequency": self.sampler_sample_frequency,
            "temp_init": self.temp_init,
            "temp_min": self.temp_min,
            "temp_annealing": self.temp_annealing,
            "total_epochs": self.total_epochs,
        }
        return snas_config


class DRNASProfile(ProfileConfig):
    def __init__(
        self,
        is_partial_connection: bool = False,
        perturbation: str | None = None,
        sampler_sample_frequency: str = "epoch",
        perturbator_sample_frequency: str = "epoch",
    ) -> None:
        PROFILE_TYPE = "DRNAS"
        super().__init__(
            PROFILE_TYPE,
            is_partial_connection,
            perturbation,
            sampler_sample_frequency,
            perturbator_sample_frequency,
        )

    def get_sampler_config(self) -> dict:
        drnas_config = {
            "sample_frequency": self.sampler_sample_frequency,
        }
        return drnas_config
