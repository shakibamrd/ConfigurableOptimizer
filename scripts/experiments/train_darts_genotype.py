from __future__ import annotations

import argparse

from confopt.profiles.profiles import DiscreteProfile
from confopt.train import Experiment
from confopt.train.experiment import DatasetType, SearchSpaceType


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train DARTS Genotype", add_help=False)

    parser.add_argument(
        "--genotype",
        help="genotype to train",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        help="dataset",
        type=str,
    )

    parser.add_argument(
        "--project-name",
        default="iclr-darts-genotypes",
        help="project name for wandb logging",
        type=str,
    )

    parser.add_argument(
        "--seed",
        default=100,
        help="random seed",
        type=int,
    )

    parser.add_argument(
        "--batch-size",
        default=48,
        help="batch size for train data",
        type=int,
    )

    parser.add_argument(
        "--epochs",
        default=600,
        help="number of epochs to train",
        type=int,
    )

    parser.add_argument(
        "--comments",
        default="None",
        help="Any additional comments",
        type=str,
    )

    parser.add_argument(
        "--meta-info",
        default="None",
        help="Any meta information about this run",
        type=str,
    )

    parser.add_argument(
        "--debug-mode", action="store_true", help="run experiment in debug mode"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()

    searchspace = "darts"
    assert args.dataset in [
        "cifar10",
        "cifar100",
        "imgnet16",
        "imgnet16_120",
    ], f"Soes not support dataset of type {args.dataset}"  # type: ignore

    profile = DiscreteProfile()
    profile.configure_trainer(
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        use_ddp=True,
        train_portion=1.0,
    )

    exp_type = f"DISCRETE_{searchspace}-{args.dataset}_seed{args.seed}"
    profile.genotype = args.genotype
    config = profile.get_trainer_config()
    config.update(
        {
            "genotype": profile.get_genotype(),
            "project_name": args.project_name,
            "extra__comments": args.comments,
            "extra__experiment_name": exp_type,
            "extra__is_debug": args.debug_mode,
            "extra__meta_info": args.meta_info,
        }
    )

    config = profile.get_trainer_config()

    experiment = Experiment(
        search_space=SearchSpaceType(searchspace),
        dataset=DatasetType(args.dataset),
        seed=args.seed,
        debug_mode=args.debug_mode,
        exp_name=exp_type,
        is_wandb_log=True,
    )

    experiment.init_ddp()

    discrete_trainer = experiment.train_discrete_model(profile)

    experiment.cleanup_ddp()
