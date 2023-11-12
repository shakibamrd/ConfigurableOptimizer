from __future__ import annotations

import warnings

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
        warnings.warn(
            "The argument total epochs to SNAS sampler should be set same as \
            number of epochs for training",
            stacklevel=1,
        )

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
    def __init__(
        self, config: Configuration, epochs: int = 5, run_group: str = "untitled"
    ) -> None:
        self.config_dict = config
        self.epochs = epochs
        super().__init__(config_type=config["sampler"])
        self.sampler_type = config["sampler"]
        self.run_group = run_group
        self.set_partial_connector(config["is_partial_connector"])
        self.set_perturb(config["perturbator"])

    def get_config(self) -> dict:
        run_config = super().get_config()
        run_config.update(
            {
                "wandb_group": self.run_group,
            }
        )
        return run_config

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
                    "total_epochs": self.epochs,
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

    def _get_optim_config(self, search_optim: str) -> dict:
        assert search_optim in ["base", "arch"]
        optim_type = "arch_optim"
        search_key = "arch_opt"
        if search_optim == "base":
            optim_type = "optim"
            search_key = "opt"

        config_optim = self.config_dict.get(optim_type, "sgd")

        if config_optim == "adam":
            return {
                optim_type: config_optim,
                optim_type
                + "_config": {
                    "betas": (
                        self.config_dict.get(search_key + "_beta1", 0.9),
                        self.config_dict.get(search_key + "_beta2", 0.999),
                    ),
                },
            }

        if config_optim == "sgd":
            return {
                optim_type: config_optim,
                optim_type
                + "_config": {
                    "momentum": self.config_dict.get(search_key + "_momentum", 0.9),
                },
            }

        if config_optim == "asgd":
            return {
                optim_type: config_optim,
                optim_type
                + "_config": {
                    "lambd": self.config_dict.get(search_key + "_lambda", 0.9),
                },
            }

        return {optim_type: config_optim}

    def get_trainer_config(self) -> dict:
        trainer_config = {
            "epochs": self.epochs,
            "lr": self.config_dict.get("lr", 0.025),
            "arch_lr": self.config_dict.get("arch_lr", 0.001),
            "optim": self.config_dict.get("optim", "sgd"),
            "arch_optim": self.config_dict.get("arch_optim", "adam"),
            "criterion": self.config_dict.get("criterion", "cross_entropy"),
            "batch_size": self.config_dict.get("batch_size", 96),
            "learning_rate_min": self.config_dict.get("learning_rate_min", 0),
            "cutout": self.config_dict.get("cutout", -1),
            "cutout_length": self.config_dict.get("cutout_length", None),
            "train_portion": self.config_dict.get("train_portion", 0.7),
            "use_data_parallel": self.config_dict.get("use_data_parallel", 1),
            "checkpointing_freq": self.config_dict.get("checkpointing_freq", 3),
        }

        trainer_config.update(self._get_optim_config("base"))
        trainer_config.update(self._get_optim_config("arch"))

        return trainer_config
