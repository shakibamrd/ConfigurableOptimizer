from __future__ import annotations

import argparse
from collections import namedtuple
from enum import Enum
import random

import numpy as np
import torch
from torch.backends import cudnn

from confopt.dataset import CIFAR10Data
from confopt.oneshot.archsampler import DARTSSampler
from confopt.searchspace import NASBench201SearchSpace, adapt_search_space
from confopt.train.trainer import Trainer
from confopt.utils import Logger


class SearchSpace(Enum):
    DARTS = "darts"
    NB201 = "nb201"
    NB1SHOT1 = "nb1shot1"


class DatasetType(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMGNET16_120 = "imgnet16_120"


N_CLASSES = {
    DatasetType.CIFAR10: 10,
    DatasetType.CIFAR100: 100,
    DatasetType.IMGNET16_120: 120,
}


class Experiment:
    def __init__(
        self, search_space: SearchSpace, dataset: DatasetType, seed: int
    ) -> None:
        self.search_space = search_space
        self.dataset = dataset
        self.seed = seed

    def set_seed(self, rand_seed: int) -> None:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        cudnn.benchmark = True
        torch.manual_seed(rand_seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(rand_seed)

    def run(self) -> Trainer:
        self.set_seed(self.seed)

        self.logger = Logger(log_dir="logs", seed=self.seed, exp_name="test")

        # Using nb201 config for now
        nb201_config = {
            "layers": 20,
            "lr": 0.025,
            "epochs": 600,
            "optim": "SGD",
            "momentum": 0.9,
            "nesterov": 0,
            "criterion": "Softmax",
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

        Arguments = namedtuple(  # type: ignore
            "Configure", " ".join(nb201_config.keys())
        )

        config = Arguments(**nb201_config)  # type: ignore

        criterion = torch.nn.CrossEntropyLoss()
        data = CIFAR10Data("./data", cutout=0, train_portion=0.5)
        model = NASBench201SearchSpace()

        w_optimizer = torch.optim.SGD(
            model.parameters(),
            config.lr,  # type: ignore
            momentum=config.momentum,  # type: ignore
            weight_decay=config.weight_decay,  # type: ignore
            nesterov=config.nesterov,  # type: ignore
        )

        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=w_optimizer,
            T_max=float(config.epochs),  # type: ignore
            eta_min=config.learning_rate_min,  # type: ignore
        )

        arch_optimizer = torch.optim.Adam(model.arch_parameters)

        model = adapt_search_space(model, DARTSSampler, None)

        trainer = Trainer(
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

        trainer.search(config.epochs)  # type: ignore

        return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Fine tuning and training searched architectures", add_help=False
    )
    parser.add_argument(
        "--searchspace",
        default="darts",
        help="search space in (darts, nb201, nats)",
        type=str,
    )
    parser.add_argument(
        "--one_shot_opt", default="drnas", help="optimizer used for search", type=str
    )
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--logdir", default="./logs", type=str)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--seed", default=444, type=int)
    parser.add_argument("--genotype", default="darts", type=str)
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
    dataset = DatasetType(args.dataset)

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=args.seed,
    )

    trainer = experiment.run()
