from __future__ import annotations
import argparse

from confopt.profile import (
    DARTSProfile,
    DRNASProfile,
    GDASProfile,
    BaseProfile,
)
from confopt.train import Experiment
from confopt.enums import SamplerType, SearchSpaceType, DatasetType
from confopt.utils import get_num_classes


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Synthetic Experiment", add_help=False)

    parser.add_argument(
        "--optimizer",
        default="darts",
        help="arch sampler to use (darts, drnas, gdas)",
        type=str,
    )

    parser.add_argument(
        "--fairdarts",
        type=bool,
        default=False,
        help="Use fair darts sampling and regularization",
    )

    parser.add_argument(
        "--oles", type=bool, default=False, help="Use operation-level early stopping"
    )

    parser.add_argument(
        "--sdarts",
        type=str,
        choices=["none", "adverserial", "random"],
        default="none",
        help="use perturbation",
    )

    parser.add_argument(
        "--pattern_type",
        type=int,
        default=1,
        help="Use pattern type = 1 for normal, = 2 for inverse case",
    )

    parser.add_argument(
        "--search_epochs",
        default=100,
        help="number of epochs to train the supernet",
        type=int,
    )

    parser.add_argument(
        "--signal_width",
        default=5,
        help="receptive width of the signal",
        type=int,
    )

    parser.add_argument(
        "--shortcut_width",
        default=3,
        help="receptive width of the shortcut",
        type=int,
    )

    parser.add_argument(
        "--shortcut_strength",
        default=0.1,
        help="strength of shortcut",
        type=float,
    )

    parser.add_argument(
        "--test_patch_width",
        default=10,
        help="reserved patch width for the test set",
        type=int,
    )

    parser.add_argument(
        "--seed",
        default=9001,
        help="Seed for the experiment",
        type=int,
    )

    args = parser.parse_args()
    return args


def get_profile(args: argparse.Namespace) -> BaseProfile:  # type: ignore
    supported_optimizers = {
        "darts": DARTSProfile,
        "gdas": GDASProfile,
        "drnas": DRNASProfile,
    }
    return supported_optimizers[args.optimizer]


if __name__ == "__main__":
    args = read_args()

    WANDB_LOG = True
    searchspace = SearchSpaceType.BABYDARTS
    dataset = DatasetType.SYNTHETIC
    sampler_type = SamplerType(args.optimizer)

    reg_weights = []
    active_reg_terms = []
    arch_combine_fn = "default"

    if sampler_type == SamplerType.DRNAS:
        drnas_weight = 1
        reg_weights.append(drnas_weight)
        active_reg_terms.append("drnas")

    if args.fairdarts:
        fairdarts_weight = 10
        reg_weights.append(fairdarts_weight)
        active_reg_terms.append("fairdarts")
        arch_combine_fn = "sigmoid"

    regularization_config = None
    if active_reg_terms:
        regularization_config = {
            "reg_weights": reg_weights,
            "loss_weight": 1,
            "active_reg_terms": active_reg_terms,
            "drnas_config": {"reg_scale": 1e-3, "reg_type": "l2"},
            "fairdarts_config": {},
        }

    profile = get_profile(args)(
        seed=args.seed,
        searchspace=searchspace,
        epochs=args.search_epochs,
        perturbation=args.sdarts,
        perturbator_sample_frequency="epoch",
        oles=args.oles,
        calc_gm_score=True,
        is_regularization_enabled=active_reg_terms,
        regularization_config=regularization_config,
        sampler_arch_combine_fn=arch_combine_fn,
    )

    profile.configure_synthetic_dataset(
        signal_width=args.signal_width,
        shortcut_width=args.shortcut_width,
        shortcut_strength=args.shortcut_strength,
        pattern_type=args.pattern_type,
        test_patch_width=args.test_patch_width,
    )

    assert args.pattern_type in [1, 2]
    if args.pattern_type == 1:
        # there is already a 3x3 stem, so select this carefully
        primitives = [
            "conv_3x3",
            "skip_connect",
        ]
    else:
        primitives = [
            "stacked_conv_3x3",
            "conv_5x5",
        ]

    profile.configure_searchspace(
        num_classes=get_num_classes(dataset.value),
        stem_multiplier=1,
        primitives=primitives,
        C=3,
    )
    project_name = "Synthetic-Benchsuite"
    exp_name = (
        f"synthetic-test-{args.optimizer}-"
        f"fairdarts-{args.fairdarts}-"
        f"oles-{args.oles}-"
        f"sdarts-{args.sdarts}-"
        f"sig{args.signal_width}x{args.signal_width}-"
        f"short{args.shortcut_width}x{args.shortcut_width}-"
        f"strength{args.shortcut_strength:.3f}"
    )
    profile.configure_extra(project_name=project_name, meta_info=exp_name)
    # Configure experiment parameters
    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=args.seed,
        debug_mode=False,
        exp_name=exp_name,
        log_with_wandb=WANDB_LOG,
    )

    # Execute the training process
    experiment.train_supernet(profile)
