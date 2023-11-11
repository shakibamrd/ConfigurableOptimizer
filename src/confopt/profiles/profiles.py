from __future__ import annotations

from ConfigSpace import Configuration
import torch

from .profile_config import ADVERSERIAL_DATA, ProfileConfig


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


# Dependent on confopt.utils.configspace
class ConfigSpaceProfile(ProfileConfig):
    def __init__(self, config: Configuration) -> None:
        self.config_dict = config
        super().__init__(config_type=config["sampler"])
        self.sampler_type = config["sampler"]
        self.set_partial_connector(config["is_partial_connector"])
        self.set_perturb(config["perturbator"])

    def get_sampler_config(self) -> dict:
        sampler_config = {}
        sampler_config["sample_frequency"] = self.config_dict[
            "sampler_sample_frequency"
        ]
        if self.sampler_type == "gdas":
            sampler_config.update(
                {
                    "tau_min": self.config_dict["tau_min"],
                    "tau_max": self.config_dict["tau_max"],
                }
            )
        elif self.sampler_type == "snas":
            sampler_config.update(
                {
                    "temp_init": self.config_dict["temp_init"],
                    "temp_min": self.config_dict["temp_min"],
                    "temp_annealing": self.config_dict["temp_annealing"],
                    "total_epochs": self.config_dict["total_epochs"],
                }
            )
        return sampler_config

    def get_perturb_config(self) -> dict | None:
        if self.perturb_type == "adverserial":
            perturb_config = {
                "epsilon": self.config_dict["epsilon"],
                "data": ADVERSERIAL_DATA,
                "loss_criterion": torch.nn.CrossEntropyLoss(),
                "steps": self.config_dict["steps"],
                "random_start": self.config_dict["random_start"],
                "sample_frequency": self.config_dict["perturbator_sample_frequency"],
            }
        elif self.perturb_type == "random":
            perturb_config = {
                "epsilon": self.config_dict["epsilon"],
                "sample_frequency": self.config_dict["perturbator_sample_frequency"],
            }
        else:
            return None
        return perturb_config

    def get_partial_conenctor(self) -> dict | None:
        partial_connector_config = {"k": self.config_dict.get("k", None)}
        return partial_connector_config

    def get_trainer_config(self) -> dict:
        trainer_config = {
            "epochs": 20,
            "lr": self.config_dict["lr"],
            "optim": self.config_dict["optim"],
            "arch_optim": self.config_dict["arch_optim"],
            "momentum": self.config_dict["momentum"],
            "nesterov": self.config_dict["nesterov"],
            "criterion": self.config_dict["criterion"],
            "batch_size": self.config_dict["batch_size"],
            "learning_rate_min": self.config_dict["learning_rate_min"],
            "weight_decay": self.config_dict["weight_decay"],
            "cutout": self.config_dict["cutout"],
            "cutout_length": self.config_dict.get("cutout_length", None),
            "train_portion": self.config_dict["train_portion"],
            "use_data_parallel": self.config_dict["use_data_parallel"],
            "checkpointing_freq": self.config_dict["checkpointing_freq"],
        }
        return trainer_config
