from __future__ import annotations

from ConfigSpace import Configuration
import torch

from .profile_config import ADVERSERIAL_DATA, ProfileConfig


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
