from __future__ import annotations

import argparse

import torch
import wandb

from confopt.profiles import ConfigSpaceProfile
from confopt.train import (
    DatasetType,
    Experiment,
    PerturbatorEnum,
    SamplersEnum,
    SearchSpaceEnum,
)
from confopt.utils import get_configspace

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--logdir", default="./logs", type=str)
    parser.add_argument("--seed", default=444, type=int)
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        default=False,
        help="Load the best model found from the previous run",
    )
    parser.add_argument(
        "--load_saved_model",
        action="store_true",
        default=False,
        help="Load the last saved model in the last run of training them",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        help="Specify the start epoch to continue the training of the model from the \
        previous run",
        type=int,
    )
    parser.add_argument(
        "--num_exp",
        default=5,
        help="Number of random search experiment to perform",
        type=int,
    )
    args = parser.parse_args()
    print(args.num_exp)

    cs = get_configspace(seed=args.seed)
    for _ in range(args.num_exp):  # type: ignore
        config = cs.sample_configuration()
        cs_profile = ConfigSpaceProfile(config)

        searchspace = SearchSpaceEnum(args.searchspace)  # type: ignore
        sampler = SamplersEnum(config["sampler"])
        perturbator = PerturbatorEnum(config.get("perturbator", None))
        dataset = DatasetType(args.dataset)  # type: ignore
        edge_normalization = config["edge_normalization"]
        is_partial_connection = config["is_partial_connector"]

        # wandb.init(  # type: ignore
        #     project="Configurable_Optimizers",
        #     config=custom_profile.get_config(),
        # )
        # TODO change after PR in design of experiment
        experiment = Experiment(
            search_space=searchspace,
            dataset=dataset,
            sampler=sampler,
            seed=args.seed,  # type: ignore
            perturbator=perturbator,
            edge_normalization=True,
            is_partial_connection=is_partial_connection,
            is_wandb_log=False,  # True
        )

        experiment.run_with_profile(cs_profile)
