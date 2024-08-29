from __future__ import annotations

import argparse
import json

from confopt.profiles.profile_config import BaseProfile
from confopt.profiles.profiles import DARTSProfile, DRNASProfile, GDASProfile
from confopt.train import Experiment
from confopt.train.experiment import DatasetType, SearchSpaceType

dataset_size = {
    "cifar10": 10,
    "cifar100": 100,
    "imgnet16": 1000,
    "imgnet16_120": 120,
}


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DRNAS Baseline run", add_help=False)

    parser.add_argument(
        "--searchspace",
        required=True,
        help="choose the search space (darts, nb201)",
        type=str,
    )

    parser.add_argument(
        "--entangle_op_weights",
        action="store_true",
        default=False,
        help="Whether to use weight entanglement or not",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="dataset to be used (cifar10, cifar100, imagenet)",
        type=str,
    )

    parser.add_argument(
        "--search_epochs",
        required=True,
        help="number of epochs to train the supernet",
        type=int,
    )

    parser.add_argument(
        "--sampler",
        required=True,
        help="Choose sampler from (darts, drnas, gdas)",
        type=str,
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use lora or not",
    )

    parser.add_argument(
        "--lora_warm_epochs",
        default=0,
        help="number of warm epochs for lora to run on",
        type=int,
    )

    parser.add_argument(
        "--lora_rank",
        default=0,
        help="rank for the lora modules",
        type=int,
    )

    parser.add_argument(
        "--lora_alpha",
        default=1,
        help="alpha multiplier for the lora modules",
        type=int,
    )

    parser.add_argument(
        "--lora_dropout",
        default=0,
        help="dropout value for the lora modules",
        type=int,
    )

    parser.add_argument(
        "--lora_merge_weights",
        action="store_true",
        default=False,
        help="merge lora weights with conv weights",
    )

    parser.add_argument(
        "--seed",
        required=True,
        help="random seed",
        type=int,
    )

    parser.add_argument(
        "--wandb_log", action="store_true", default=True, help="turn wandb logging on"
    )

    parser.add_argument(
        "--debug_mode", action="store_true", default=False, help="run experiment in debug mode"
    )

    parser.add_argument("--cosine_anneal_restarts_T0", required=True, type=int)
    parser.add_argument("--cosine_anneal_restarts_Tmult", required=True, type=int)

    args = parser.parse_args()
    return args


def get_configuration(
    profile_type: BaseProfile, args: argparse.Namespace
) -> BaseProfile:
    profile = profile_type(
        epochs=args.search_epochs,
        lora_rank=args.lora_rank,
        lora_warm_epochs=args.lora_warm_epochs,
        entangle_op_weights=args.entangle_op_weights,
        searchspace_str=args.searchspace,
        calc_gm_score=True,
        seed=args.seed,
    )
    return profile


def get_profile(args: argparse.Namespace) -> BaseProfile:
    if args.sampler == "darts":
        profile_type = DARTSProfile
    elif args.sampler == "drnas":
        profile_type = DRNASProfile
    elif args.sampler == "gdas":
        profile_type = GDASProfile
    else:
        raise ValueError(f"Sampler {args.sampler} not supported")

    return get_configuration(profile_type, args)

if __name__ == "__main__":
    args = read_args()

    assert args.searchspace in ["darts"], f"Does not support space of type {args.searchspace}"  # type: ignore
    assert args.dataset in ["cifar10", "cifar100", "imagenet"], f"Does not support dataset of type {args.dataset}"  # type: ignore

    if args.use_lora:
        assert args.lora_warm_epochs > 0, "argument --lora_warm_epochs should not be 0 when argument --use_lora is provided"  # type: ignore
        assert args.lora_rank > 0, "argument --lora_rank should be greater than 0"  # type: ignore

    assert args.sampler in ["darts", "drnas", "gdas"], "This experiment supports only darts, drnas and gdas as samplers"  # type: ignore

    profile = get_profile(args)

    searchspace_config = {
        "num_classes": dataset_size[args.dataset],
    }
    profile.set_searchspace_config(searchspace_config)

    if args.use_lora:
        profile.configure_lora_config(
            lora_dropout=args.lora_dropout,  # type: ignore
            merge_weights=args.lora_merge_weights,  # type: ignore
            lora_alpha=args.lora_alpha,  # type: ignore
        )

    # Extra info for wandb tracking
    project_name = "confopt"
    profile.configure_extra(
        {
            "project_name": project_name,
            "experiment_type": f"cosine_annealing_with_restarts",
        }
    )

    print(json.dumps(profile.get_config(), indent=2, default=str))

    scheduler_config = {
        "T_0": args.cosine_anneal_restarts_T0,
        "T_mult": args.cosine_anneal_restarts_Tmult,
    }

    profile.configure_trainer(scheduler_config=scheduler_config)

    # Experiment name for logging
    experiment_name = f"{args.sampler}_restarts_T0{args.cosine_anneal_restarts_T0}_Tmult{args.cosine_anneal_restarts_Tmult}"


    experiment = Experiment(
        search_space=SearchSpaceType(args.searchspace),
        dataset=DatasetType(args.dataset),
        seed=args.seed,
        is_wandb_log=args.wandb_log,
        debug_mode=args.debug_mode,
        exp_name=experiment_name,
    )
    trainer = experiment.train_supernet(profile, use_benchmark=True)
