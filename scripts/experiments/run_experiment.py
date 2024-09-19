from __future__ import annotations

import time
import random
import argparse
import json

from confopt.profiles.profile_config import BaseProfile
from confopt.profiles.profiles import (
    DARTSProfile,
    DRNASProfile,
    GDASProfile,
    ReinMaxProfile,
)
from confopt.train import Experiment
from confopt.train.experiment import DatasetType, SearchSpaceType

dataset_size = {
    "cifar10": 10,
    "cifar100": 100,
    "imgnet16": 1000,
    "imgnet16_120": 120,
}


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Experiment run", add_help=False)

    parser.add_argument(
        "--searchspace",
        default="darts",
        help="choose the search space (darts, nb201)",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="dataset to be used (cifar10, cifar100, imagenet)",
        type=str,
    )

    parser.add_argument(
        "--sampler",
        default="darts",
        help="Choose sampler from (darts, drnas, gdas, reinmax)",
        type=str,
    )

    parser.add_argument(
        "--epochs",
        default=50,
        help="number of epochs to train the supernet",
        type=int,
    )

    parser.add_argument(
        "--partial-connection",
        action="store_true",
        default=False,
        help="Whether to use partial connection or not",
    )

    parser.add_argument(
        "--partial-connection-k",
        default=2,
        help="k value for partial connection",
        type=int,
    )

    parser.add_argument(
        "--partial-connection-warm-epochs",
        default=15,
        help="Number of warm epochs for partial connection",
        type=int,
    )

    parser.add_argument(
        "--perturbation",
        default="none",
        help="Type of perturbation to be used (none, random, adversarial)",
        type=str,
    )

    parser.add_argument(
        "--perturbator-epsilon",
        default=0.0,
        help="epsilon value for perturbator",
        type=float,
    )

    parser.add_argument(
        "--dropout",
        default=None,
        help="dropout value for the model",
        type=float,
    )

    parser.add_argument(
        "--sampler-arch-combine-fn",
        default="default",
        help="Use 'sigmoid' for Fair-DARTS post-processing of the arch parameters",
        type=str,
    )

    parser.add_argument(
        "--entangle-op-weights",
        action="store_true",
        default=False,
        help="Whether to use weight entanglement or not",
    )

    parser.add_argument(
        "--lora-rank",
        default=0,
        help="rank for the lora modules",
        type=int,
    )

    parser.add_argument(
        "--lora-warm-epochs",
        default=0,
        help="number of warm epochs for lora to run on",
        type=int,
    )

    parser.add_argument(
        "--seed",
        default=9001,
        help="random seed",
        type=int,
    )

    parser.add_argument(
        "--oles",
        action="store_true",
        default=False,
        help="Whether to use OLES or not",
    )

    parser.add_argument(
        "--oles-freq",
        default=20,
        help="Frequency of OLES",
        type=int,
    )

    parser.add_argument(
        "--oles-threshold",
        default=0.4,
        help="Threshold for OLES. If the GM score of a module is less than" +
             "this threshold, it is frozen.",
        type=float,
    )

    parser.add_argument(
        "--arch-attention-enabled",
        action="store_true",
        default=False,
        help="Whether to use architecture attention or not",
    )

    parser.add_argument(
        "--debug-mode", action="store_true", help="run experiment in debug mode"
    )

    parser.add_argument(
        "--project-name",
        default="lora-darts-iclr",
        help="project name for wandb logging",
        type=str,
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

    args = parser.parse_args()
    return args


def get_configuration(
    profile_type: BaseProfile, args: argparse.Namespace
) -> BaseProfile:
    partial_connector_config = None
    if args.partial_connection:
        partial_connector_config = {
            "k": args.partial_connection_k,
            "num_warm_epoch": args.partial_connection_warm_epochs,
        }

    perturbator_config = None
    if args.perturbation != "none":
        perturbator_config = {
            "epsilon": args.perturbator_epsilon,
        }

    reg_config = None
    is_regularization_enabled = False
    prune_epochs = None
    prune_fractions = None
    is_partial_connection = args.partial_connection

    if profile_type == DRNASProfile:
        reg_config = {
            "active_reg_terms": ["drnas"],
            "reg_weights": [0.5],
            "loss_weight": 0.5,
            "drnas_config": {
                "reg_scale": 2e-3,
            },
        }
        is_regularization_enabled = True
        is_partial_connection = True
        prune_epochs = [args.epochs // 2]
        prune_fractions = [0.5]

    profile = profile_type(
        epochs=args.epochs,
        is_partial_connection=is_partial_connection,
        partial_connector_config=partial_connector_config,
        perturbation=args.perturbation,
        perturbator_config=perturbator_config,
        sampler_sample_frequency="step",
        dropout=args.dropout,
        sampler_arch_combine_fn=args.sampler_arch_combine_fn,
        entangle_op_weights=args.entangle_op_weights,
        lora_rank=args.lora_rank,
        lora_warm_epochs=args.lora_warm_epochs,
        lora_toggle_epochs=None,
        lora_toggle_probability=None,
        seed=args.seed,
        searchspace_str=args.searchspace,
        oles=args.oles,
        calc_gm_score=True,
        prune_epochs=prune_epochs,
        prune_fractions=prune_fractions,
        is_arch_attention_enabled=args.arch_attention_enabled,
        is_regularization_enabled=is_regularization_enabled,
        regularization_config=reg_config,
        pt_select_architecture=False,  # TODO-ICLR: Add architecture selection?
    )
    return profile


if __name__ == "__main__":

    # Avoid collision with other runs that also make directories
    random_milliseconds = random.uniform(5, 55)
    time.sleep(random_milliseconds)

    args = read_args()

    assert args.searchspace in ["darts", "nb201"], \
        f"Does not support space of type {args.searchspace}"  # type: ignore
    assert args.dataset in ["cifar10", "cifar100", "imagenet"], \
        f"Soes not support dataset of type {args.dataset}"  # type: ignore

    lora_str = ""
    if args.lora_rank > 0:
        assert args.lora_warm_epochs > 0, \
        "argument --lora_warm_epochs should be greater than 0 when LoRA is enabled."
        lora_str = f"-lora-rank-{args.lora_rank}-warm-{args.lora_warm_epochs}"

    assert args.sampler in ["darts", "drnas", "gdas", "reinmax"], \
        "This experiment supports only darts, drnas, gdas and reinmax as samplers"  # type: ignore

    sampler_profiles = {
        "darts": DARTSProfile,
        "drnas": DRNASProfile,
        "gdas": GDASProfile,
        "reinmax": ReinMaxProfile,
    }

    profile = get_configuration(sampler_profiles[args.sampler], args)

    searchspace_config = {
        "num_classes": dataset_size[args.dataset],
    }

    if args.sampler == "drnas":
        searchspace_config.update({"C": 36, "layers": 8})
        profile.configure_partial_connector(num_warm_epoch=0, k=6)

    profile.set_searchspace_config(searchspace_config)

    # Extra info for wandb tracking
    project_name = args.project_name

    oles_str = ""
    if args.oles:
        profile.configure_oles(frequency=args.oles_freq, threshold=args.oles_threshold)
        oles_str = f"-oles-freq-{args.oles_freq}-threshold-{args.oles_threshold}"

    exp_type = f"{args.searchspace}-{args.sampler}{lora_str}{oles_str}-{args.dataset}"

    profile.configure_extra(
        {
            "project_name": project_name,
            "extra:comments": args.comments,
            "extra:experiment-name": exp_type,
            "extra:is-debug": args.debug_mode,
            "extra:meta-info": args.meta_info,
        }
    )

    print(json.dumps(profile.get_config(), indent=2, default=str))

    # Experiment name for logging
    experiment_name = f"{args.sampler}"

    experiment = Experiment(
        search_space=SearchSpaceType(args.searchspace),
        dataset=DatasetType(args.dataset),
        seed=args.seed,
        is_wandb_log=True,
        debug_mode=args.debug_mode,
        exp_name=experiment_name,
    )
    trainer = experiment.train_supernet(profile, use_benchmark=True)
