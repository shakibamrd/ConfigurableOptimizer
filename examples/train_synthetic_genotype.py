from __future__ import annotations
import argparse


from confopt.profile.profiles import DiscreteProfile
from confopt.train import Experiment
from confopt.enums import SearchSpaceType, DatasetType
from confopt.utils import get_num_classes


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Synthetic Experiment", add_help=False)

    parser.add_argument(
        "--search_epochs",
        default=20,
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
        help="receptive width of the signal",
        type=int,
    )

    parser.add_argument(
        "--shortcut_strength",
        default=0.1,
        help="strength of shortcut",
        type=float,
    )

    parser.add_argument(
        "--seed",
        default=9001,
        help="Seed for the experiment",
        type=int,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()

    searchspace = SearchSpaceType.BABYDARTS
    dataset = DatasetType.SYNTHETIC

    profile = DiscreteProfile(searchspace=searchspace, epochs=20, batch_size=64)

    profile.configure_extra(
        synthetic_dataset_config={
            "signal_width": args.signal_width,
            "shortcut_width": args.shortcut_width,
            "shortcut_strength": args.shortcut_strength,
            "pattern_type": 2,
        }
    )

    profile.set_search_space_config(
        {
            "num_classes": get_num_classes(dataset.value),
            "stem_multiplier": 1,
            "C": 3,
        }
    )
    profile.configure_trainer(cutout=0, train_portion=1, seed=args.seed)
    # genotype_str = "stacked_conv_3x3"
    genotype_str = "conv_5x5"
    profile.genotype = genotype_str
    project_name = "Synthetic-Benchsuite-Discrete"
    exp_name = (
        f"sig{args.signal_width}x{args.signal_width}-"
        f"short{args.shortcut_width}x{args.shortcut_width}-"
        f"strength{args.shortcut_strength:.3f}-"
        f"{genotype_str}"
    )
    profile.train_config.update(
        {
            "project_name": project_name,
        }
    )

    profile.configure_extra(project_name=project_name, meta_info=exp_name)
    # Configure experiment parameters
    experiment = Experiment(
        search_space=SearchSpaceType.BABYDARTS,
        dataset=DatasetType.SYNTHETIC,
        seed=args.seed,
        debug_mode=False,
        exp_name=exp_name,
        log_with_wandb=True,
    )

    # Execute the training process
    experiment.train_discrete_model(profile)
