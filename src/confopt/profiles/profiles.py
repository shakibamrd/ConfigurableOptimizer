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


class DRNASProfile(ProfileConfig):
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


class DiscreteProfile:
    def get_trainer_config(self) -> dict:
        default_train_config = {
            "lr": 0.025,
            "epochs": 100,
            "optim": "sgd",
            "momentum": 0.9,
            "nesterov": 0,
            "criterion": "cross_entropy",
            "batch_size": 96,
            "learning_rate_min": 0.0,
            "weight_decay": 3e-4,
            "channel": 36,
            "drop_path_prob": 0.2,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.7,
            "use_data_parallel": 0,
            "checkpointing_freq": 1,
        }
        return default_train_config
