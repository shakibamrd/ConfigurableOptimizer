from __future__ import annotations
import argparse


from confopt.profile.profiles import DiscreteProfile
from confopt.train import Experiment
from confopt.enums import SearchSpaceType, DatasetType
from confopt.utils import get_num_classes


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Synthetic Experiment", add_help=False)

    parser.add_argument(
        "--epochs",
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
        "--operation",
        default="conv_5x5",
        help="learning rate",
        type=str,
    )

    parser.add_argument(
        "--lr",
        default=0.1,
        help="learning rate",
        type=float,
    )

    parser.add_argument(
        "--wd",
        default=3e-4,
        help="learning rate",
        type=float,
    )

    parser.add_argument(
        "--seed",
        default=100,
        help="Seed for tshe experiment",
        type=int,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()

    searchspace = SearchSpaceType.BABYDARTS
    dataset = DatasetType.SYNTHETIC

    profile = DiscreteProfile(
        searchspace=searchspace, epochs=args.epochs, batch_size=512
    )

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
            "C": 1,
        }
    )

    optimizer_config = {
        "momentum": 0.9,
        "nesterov": False,
        "weight_decay": args.wd,
    }

    profile.configure_trainer(
        lr=args.lr,
        optim_config=optimizer_config,
        cutout=0,
        train_portion=1,
        seed=args.seed,
    )
    profile.genotype = args.operation
    project_name = f"Synthetic-Benchsuite-Discrete-{args.operation}"
    exp_name = (
        f"sig{args.signal_width}x{args.signal_width}-"
        f"short{args.shortcut_width}x{args.shortcut_width}-"
        f"strength{args.shortcut_strength:.3f}-"
        f"{args.operation}"
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
