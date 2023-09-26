from __future__ import annotations

import argparse
from collections import namedtuple
from enum import Enum
import random
from typing import Callable

import numpy as np
import torch
from torch.backends import cudnn
import wandb

from confopt.dataset import CIFAR10Data, CIFAR100Data, ImageNet16Data, ImageNet16120Data
from confopt.oneshot.archmodifier import SDARTSSampler
from confopt.oneshot.archsampler import (
    DARTSSampler,
    DRNASSampler,
    GDASSampler,
    SNASSampler,
)
from confopt.oneshot.partial_connector import PartialConnector
from confopt.searchspace import (
    DARTSSearchSpace,
    NASBench1Shot1SearchSpace,
    NASBench201SearchSpace,
    TransNASBench101SearchSpace,
)
from confopt.train import ConfigurableTrainer, Profile
from confopt.utils import Logger


class SearchSpace(Enum):
    DARTS = "darts"
    NB201 = "nb201"
    NB1SHOT1 = "nb1shot1"
    TNB101 = "tnb101"


class Samplers(Enum):
    DARTS = "darts"
    DRNAS = "drnas"
    GDAS = "gdas"
    SNAS = "snas"


class Perturbator(Enum):
    RANDOM = "random"
    ADVERSERIAL = "adverserial"


PERTUB_DEFAULT_EPSILON = 0.03


class DatasetType(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMGNET16 = "imgnet16"
    IMGNET16_120 = "imgnet16_120"


N_CLASSES = {
    DatasetType.CIFAR10: 10,
    DatasetType.CIFAR100: 100,
    DatasetType.IMGNET16_120: 120,
}


class Criterions(Enum):
    CROSS_ENTROPY = "cross_entropy"


class Optimizers(Enum):
    ADAM = "adam"
    SGD = "sgd"


class Experiment:
    def __init__(
        self,
        search_space: SearchSpace,
        dataset: DatasetType,
        sampler: Samplers,
        seed: int,
        perturbator: Perturbator | None = None,
        edge_normalization: bool = False,
        is_partial_connection: bool = False,
        is_wandb_log: bool = False,
    ) -> None:
        self.search_space_str = search_space
        self.sampler_str = sampler
        self.perturbator_str = perturbator
        self.edge_normalization = edge_normalization
        self.is_partial_connection = is_partial_connection
        self.dataset_str = dataset
        self.seed = seed
        self.is_wandb_log = is_wandb_log

    def set_seed(self, rand_seed: int) -> None:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        cudnn.benchmark = True
        torch.manual_seed(rand_seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(rand_seed)

    def run_with_profile(self, profile: Profile) -> ConfigurableTrainer:
        # TODO Run with profile
        # config = profile.get_config()
        pass

    def run(self, config: dict | None = None) -> ConfigurableTrainer:
        self.set_seed(self.seed)

        self.logger = Logger(log_dir="logs", seed=self.seed, exp_name="test")

        if config is None:
            assert (
                self.search_space_str == SearchSpace.NB201
            ), "Default config only works with nb201, Please initialize Experiment \
                    with SearchSpace of type NB201"
            assert (
                self.sampler_str == Samplers.DARTS
            ), "Default config only works with darts sampler, Please initialize \
                    Experiment with sampler of type darts"
            nb201_config = {
                "layers": 20,
                "lr": 0.025,
                "epochs": 600,
                "optim": "SGD",
                "momentum": 0.9,
                "nesterov": 0,
                "criterion": "cross_entropy",
                "batch_size": 96,
                "affine": 0,
                "learning_rate_min": 0.0,
                "weight_decay": 3e-4,
                "channel": 36,
                "auxiliary": False,
                "auxiliary_weight": 0.4,
                "track_running_stats": 1,
                "drop_path_prob": 0.2,
                "cutout": -1,
                "cutout_length": 16,
                "train_portion": 0,
                "use_data_parallel": 0,
            }

            darts_sampler_config = {
                "sample_frequency": "step",
            }

            adverserial_config = {
                "epsilon": 0.03,
                "search_space": self.search_space,
                "data": 0,
                "loss_criterion": torch.nn.CrossEntropyLoss(),
                "steps": 20,
                "random_start": True,
                "sample_frequency": "epoch",
            }

            partial_connector_config = {
                "k": 4,
            }

            confopt_config: dict = {
                "search_space": nb201_config,
                "sampler": darts_sampler_config,
                "perturbator": adverserial_config,
                "partial_connector": partial_connector_config,
                "logger": {"project_name": "Configurable_Optimizer"},
            }

            self._enum_to_objects(
                self.search_space_str,
                self.sampler_str,
                self.perturbator_str,
                confopt_config,
            )

        if self.is_wandb_log:
            wandb.init(  # type: ignore
                project=confopt_config["logger"]["project_name"],  # type: ignore
                config=confopt_config,  # type: ignore
            )

        if config is None:
            Arguments = namedtuple(  # type: ignore
                "Configure", " ".join(confopt_config["search_space"].keys())
            )
        else:
            Arguments = namedtuple(  # type: ignore
                "Configure", " ".join(nb201_config.keys())  # type: ignore
            )

        arg_config = Arguments(**nb201_config)  # type: ignore

        # TODO replace criterion with criterion getter function
        criterion = self._get_criterion(
            criterion_str=arg_config.criterion  # type: ignore
        )()

        # TODO replace data with data setter function
        data = self._get_dataset(self.dataset_str)
        model = self.search_space

        # TODO replace optimizers with optimizer setter function
        w_optimizer = torch.optim.SGD(
            model.parameters(),
            arg_config.lr,  # type: ignore
            momentum=arg_config.momentum,  # type: ignore
            weight_decay=arg_config.weight_decay,  # type: ignore
            nesterov=arg_config.nesterov,  # type: ignore
        )

        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=w_optimizer,
            T_max=float(arg_config.epochs),  # type: ignore
            eta_min=arg_config.learning_rate_min,  # type: ignore
        )

        arch_optimizer = torch.optim.Adam(model.arch_parameters)

        trainer = ConfigurableTrainer(
            model=model,
            data=data,
            model_optimizer=w_optimizer,
            arch_optimizer=arch_optimizer,
            scheduler=w_scheduler,
            criterion=criterion,
            logger=self.logger,
            batch_size=config.batch_size,  # type: ignore
            use_data_parallel=config.use_data_parallel,  # type: ignore
        )

        trainer.train(
            profile=self.profile,  # type: ignore
            epochs=config.epochs,  # type: ignore
            is_wandb_log=self.is_wandb_log,
        )

        return trainer

    def _enum_to_objects(
        self,
        search_space_enum: SearchSpace,
        sampler_enum: Samplers,
        perturbator_enum: Perturbator | None = None,
        config: dict | None = None,
    ) -> None:
        self.set_search_space(search_space_enum, config["search_space"])  # type: ignore
        self.set_sampler(sampler_enum)  # type: ignore
        if perturbator_enum is not None:
            self.set_perturbator(
                perturbator_enum, config["perturbator"]  # type: ignore
            )
        else:
            self.perturbator = None
        self.set_partial_connector(config["partial_connector"])  # type: ignore
        self.set_profile()

    def set_search_space(
        self,
        search_space: SearchSpace,
        config: dict | None = None,
    ) -> None:
        if search_space == SearchSpace.NB201:
            self.search_space = NASBench201SearchSpace(kwargs=config)
        elif search_space == SearchSpace.DARTS:
            self.search_space = DARTSSearchSpace(kwargs=config)
        elif search_space == SearchSpace.NB1SHOT1:
            self.search_space = NASBench1Shot1SearchSpace(kwargs=config)
        elif search_space == SearchSpace.TNB101:
            self.search_space = TransNASBench101SearchSpace(kwargs=config)

    def set_sampler(self, sampler: Samplers) -> None:
        arch_params = self.search_space.arch_parameters
        if sampler == Samplers.DARTS:
            self.sampler = DARTSSampler(arch_parameters=arch_params)
        elif sampler == sampler.DRNAS:
            self.sampler = DRNASSampler(arch_parameters=arch_params)
        elif sampler == sampler.GDAS:
            self.sampler = GDASSampler(arch_parameters=arch_params)
        elif sampler == sampler.SNAS:
            self.sampler = SNASSampler(arch_parameters=arch_params)

    def set_perturbator(
        self,
        petubrator_enum: Perturbator,
        pertub_config: dict | None = None,  # type : ignore
    ) -> None:
        if petubrator_enum is not None:
            self.perturbator = SDARTSSampler(
                pertub_config,
                arch_parameters=self.search_space.arch_parameters,
                attack_type=petubrator_enum,
            )

    def set_partial_connector(self, config: dict | None = None) -> None:
        if config is None:
            config = {}
        if self.is_partial_connection:
            self.partial_connector = PartialConnector(  # type : ignore
                k=config.get("k", 4)  # type : ignore
            )

    def set_profile(self) -> None:
        assert self.sampler is not None
        assert self.partial_connector is not None
        assert self.perturbator is not None

        self.profile = Profile(
            sampler=self.sampler,
            edge_normalization=self.edge_normalization,
            partial_connector=self.partial_connector,
            pertubration=self.perturbator,
        )

    def _get_dataset(self, dataset: DatasetType) -> Callable | None:
        if dataset == DatasetType.CIFAR10:
            return CIFAR10Data
        elif dataset == DatasetType.CIFAR100:  # noqa: RET505
            return CIFAR100Data
        elif dataset == DatasetType.IMGNET16:
            return ImageNet16Data
        elif dataset == DatasetType.IMGNET16_120:
            return ImageNet16120Data
        return None

    def _get_criterion(self, criterion_str: str) -> Callable | None:
        criterion = Criterions(criterion_str)
        if criterion == Criterions.CROSS_ENTROPY:
            return torch.nn.CrossEntropyLoss
        return None

    def _get_optimizer(self, optim_str: str) -> Callable | None:
        optim = Optimizers(optim_str)
        if optim == Optimizers.ADAM:
            return torch.optim.Adam
        elif optim == Optimizers.SGD:  # noqa: RET505
            return torch.optim.SGD
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Fine tuning and training searched architectures", add_help=False
    )
    parser.add_argument(
        "--searchspace",
        default="nb201",
        help="search space in (darts, nb201, nb1shot1, tnb101)",
        type=str,
    )
    parser.add_argument(
        "--sampler",
        default="darts",
        help="arch sampler used for search in (darts, drnas, gdas, snas)",
        type=str,
    )
    parser.add_argument(
        "--perturbator",
        default="adverserial",
        help="Type of perturbation in (random, adverserial)",
        type=str,
    )
    parser.add_argument(
        "--is_partial_connector",
        action="store_true",
        default=True,
        help="partial connection",
    )
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--logdir", default="./logs", type=str)
    parser.add_argument("--seed", default=444, type=int)
    parser.add_argument(
        "--model_path",
        default="/path/to/model.pth",
        type=str,
    )
    parser.add_argument(
        "--load_saved_model",
        action="store_true",
        default=False,
        help="Load the saved models before training them",
    )
    args = parser.parse_args()

    searchspace = SearchSpace(args.searchspace)
    sampler = Samplers(args.sampler)
    perturbator = Perturbator(args.perturbator)
    dataset = DatasetType(args.dataset)

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        sampler=sampler,
        seed=args.seed,
        perturbator=perturbator,
        edge_normalization=True,
        is_partial_connection=True,
    )

    trainer = experiment.run()
