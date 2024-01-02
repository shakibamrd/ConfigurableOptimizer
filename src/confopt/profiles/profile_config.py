from __future__ import annotations

from abc import abstractmethod

import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# TODO Change this to real data
ADVERSERIAL_DATA = torch.randn(2, 3, 32, 32).to(DEVICE), torch.randint(0, 9, (2,)).to(
    DEVICE
)


class ProfileConfig:
    def __init__(
        self,
        config_type: str,
        is_partial_connection: bool = False,
        perturbation: str | None = None,
        perturbator_sample_frequency: str = "epoch",
    ) -> None:
        self.config_type = config_type
        self._initialize_trainer_config()
        self._initialize_sampler_config()
        self.set_partial_connector(is_partial_connection)
        self.set_perturb(perturbation, perturbator_sample_frequency)

    def set_perturb(
        self,
        perturb_type: str | None = None,
        perturbator_sample_frequency: str = "epoch",
    ) -> None:
        assert perturbator_sample_frequency in ["epoch", "step"]
        assert perturb_type in ["adverserial", "random", "none", None]
        if perturb_type is None:
            self.perturb_type = "none"
        else:
            self.perturb_type = perturb_type
        self.perturbator_sample_frequency = perturbator_sample_frequency
        self._initialize_perturbation_config()

    def set_partial_connector(self, is_partial_connection: bool = False) -> None:
        self.is_partial_connection = is_partial_connection
        self._initialize_partial_connector_config()

    def get_config(self) -> dict:
        assert (
            self.sampler_config is not None
        ), "atleast a sampler is needed to initialize the search space"
        config = {
            "sampler": self.sampler_config,
            "perturbator": self.perturb_config,
            "partial_connector": self.partial_connector_config,
            "trainer": self.trainer_config,
        }
        if hasattr(self, "searchspace_config") and self.searchspace_config is not None:
            config.update({"search_space": self.searchspace_config})
        return config

    @abstractmethod
    def _initialize_sampler_config(self) -> None:
        self.sampler_config = None

    @abstractmethod
    def _initialize_perturbation_config(self) -> None:
        if self.perturb_type == "adverserial":
            perturb_config = {
                "epsilon": 0.3,
                "data": ADVERSERIAL_DATA,
                "loss_criterion": torch.nn.CrossEntropyLoss(),
                "steps": 20,
                "random_start": True,
                "sample_frequency": self.perturbator_sample_frequency,
            }
        elif self.perturb_type == "random":
            perturb_config = {
                "epsilon": 0.3,
                "sample_frequency": self.perturbator_sample_frequency,
            }
        else:
            perturb_config = None

        self.perturb_config = perturb_config

    @abstractmethod
    def _initialize_partial_connector_config(self) -> None:
        partial_connector_config = {"k": 4} if self.is_partial_connection else None
        self.partial_connector_config = partial_connector_config

    @abstractmethod
    def _initialize_trainer_config(self) -> None:
        trainer_config = {
            "lr": 0.025,
            "epochs": 100,
            "optim": "sgd",
            "arch_optim": "adam",
            "momentum": 0.9,
            "nesterov": 0,
            "criterion": "cross_entropy",
            "batch_size": 96,
            "learning_rate_min": 0.0,
            "weight_decay": 3e-4,
            "cutout": -1,
            "cutout_length": 16,
            "train_portion": 0.7,
            "use_data_parallel": 0,
            "checkpointing_freq": 1,
        }

        self.trainer_config = trainer_config

    def configure_sampler(self, **kwargs) -> None:  # type: ignore
        assert self.sampler_config is not None
        for config_key in kwargs:
            assert (
                config_key in self.sampler_config  # type: ignore
            ), f"{config_key} not a valid configuration for the sampler of type \
                {self.config_type}"
            self.sampler_config[config_key] = kwargs[config_key]  # type: ignore

    def configure_perturbator(self, **kwargs) -> None:  # type: ignore
        assert (
            self.perturb_type != "none"
        ), "Perturbator is initialized with None, \
            re-initialize with random or adverserial"

        for config_key in kwargs:
            assert (
                config_key in self.perturb_config  # type: ignore
            ), f"{config_key} not a valid configuration for the perturbator of \
                type {self.perturb_type}"
            self.perturb_config[config_key] = kwargs[config_key]  # type: ignore

    def configure_partial_connector(self, **kwargs) -> None:  # type: ignore
        assert self.is_partial_connection is True
        for config_key in kwargs:
            assert (
                config_key in self.partial_connector_config  # type: ignore
            ), f"{config_key} not a valid configuration for the partial connector"
            self.partial_connector_config[config_key] = kwargs[  # type: ignore
                config_key
            ]

    def configure_trainer(self, **kwargs) -> None:  # type: ignore
        for config_key in kwargs:
            assert (
                config_key in self.trainer_config
            ), f"{config_key} not a valid configuration for the trainer"
            self.trainer_config[config_key] = kwargs[config_key]

    @abstractmethod
    def set_searchspace_config(self, config: dict) -> None:
        self.searchspace_config = config