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
        super().__init__(PROFILE_TYPE)
        self.sampler_type = str.lower(PROFILE_TYPE)
        self.sampler_sample_frequency = sampler_sample_frequency
        self.set_partial_connector(is_partial_connection)
        self.set_perturb(perturbation, perturbator_sample_frequency)

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
        super().__init__(PROFILE_TYPE)
        self.sampler_type = str.lower(PROFILE_TYPE)
        self.sampler_sample_frequency = sampler_sample_frequency
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.set_partial_connector(is_partial_connection)
        self.set_perturb(perturbation, perturbator_sample_frequency)

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
        super().__init__(PROFILE_TYPE)
        self.sampler_type = str.lower(PROFILE_TYPE)
        self.sampler_sample_frequency = sampler_sample_frequency
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.temp_annealing = temp_annealing
        self.total_epochs = total_epochs
        self.set_partial_connector(is_partial_connection)
        self.set_perturb(perturbation, perturbator_sample_frequency)

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
        super().__init__(PROFILE_TYPE)
        self.sampler_type = str.lower(PROFILE_TYPE)
        self.sampler_sample_frequency = sampler_sample_frequency
        self.set_partial_connector(is_partial_connection)
        self.set_perturb(perturbation, perturbator_sample_frequency)

    def get_sampler_config(self) -> dict:
        drnas_config = {
            "sample_frequency": self.sampler_sample_frequency,
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
