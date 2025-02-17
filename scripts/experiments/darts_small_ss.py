from __future__ import annotations

import argparse
from confopt.profile import (
    BaseProfile,
    DARTSProfile,
    GDASProfile,
    DRNASProfile,
    ReinMaxProfile,
)
from confopt.train import Experiment
from confopt.enums import DatasetType, SearchSpaceType


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Darts-Small-Data", add_help=False)

    parser.add_argument(
        "--searchspace",
        default="darts",
        help="Searchspace to use (darts, nb201)",
        type=str,
    )

    parser.add_argument(
        "--sampler",
        default="darts",
        help="arch sampler to use (darts, drnas, gdas, reinmax)",
        type=str,
    )

    parser.add_argument(
        "--search_epochs",
        default=100,
        help="number of epochs to train the supernet",
        type=int,
    )

    parser.add_argument(
        "--num_channels",
        default=16,
        help="Number of channels",
        type=int,
    )

    parser.add_argument(
        "--layers",
        default=8,
        help="Number of layers in the network",
        type=int,
    )

    parser.add_argument(
        "--seed",
        default=9001,
        help="Seed for the experiment",
        type=int,
    )

    parser.add_argument(
        "--arch_lr",
        default=3e-4,
        help="arch learning rate for the experiment",
        type=float,
    )

    args = parser.parse_args()
    return args


def set_ss_config_from_args(profile: BaseProfile, args: argparse.Namespace) -> None:
    if profile.searchspace_type == SearchSpaceType.DARTS:
        searchspace_config = {
            "layers": args.layers,
            "C": args.num_channels,
        }
    else:
        searchspace_config = None

    profile.configure_searchspace(**searchspace_config)


def get_profile(args: argparse.Namespace) -> BaseProfile:
    if args.sampler == "darts":
        return DARTSProfile

    if args.sampler == "gdas":
        return GDASProfile

    if args.sampler == "drnas":
        return DRNASProfile

    if args.sampler == "reinmax":
        return ReinMaxProfile


if __name__ == "__main__":
    args = read_args()
    search_space = SearchSpaceType(args.searchspace)

    profile = get_profile(args)(
        searchspace=search_space,
        epochs=args.search_epochs,
        seed=args.seed,
    )

    set_ss_config_from_args(profile, args)

    # Configure the Trainer
    profile.configure_trainer(
        lr=0.025,
        arch_lr=args.arch_lr,
        batch_size=64,
        train_portion=0.5,
        use_data_parallel=False,
        checkpointing_freq=5,  # How frequently to save the supernet
    )

    # Add any additional configurations to this run
    # Used to tell runs apart in WandB, if required
    profile.configure_extra(
        **{
            "project_name": "small-data-small-ss",  # Name of the Wandb Project
            "run_purpose": "test-ss-modification",  # Purpose of the run
        }
    )

    experiment = Experiment(
        search_space=search_space,
        dataset=DatasetType.USPS,
        seed=args.seed,
        debug_mode=False,
        exp_name="small-data-darts",
        log_with_wandb=True,  # enable logging with Weights and Biases
    )

    experiment.train_supernet(
        profile,
        
        use_benchmark=False,  # query the benchmark at the end of every epoch
    )
