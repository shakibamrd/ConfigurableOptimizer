from __future__ import annotations

import argparse
import json

from confopt.profiles import DiscreteProfile, DRNASProfile
from confopt.train import DatasetType, Experiment, SearchSpaceType

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
        default="nb201",
        help="search space in (darts, nb201)",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="dataset to be used (cifar10, cifar100, imagenet)",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()
    assert args.searchspace in ["darts", "nb201"], "Unsupported searchspace"
    searchspace = SearchSpaceType(args.searchspace)  # type: ignore
    dataset = DatasetType(args.dataset)  # type: ignore
    seed = 100

    # Sampler and Perturbator have different sample_frequency
    profile = DRNASProfile(
        is_partial_connection=True,
        sampler_sample_frequency="step",
    )
    # nb201 take in default configs, but for darts, we require different config
    searchspace_config = {
        "num_classes": dataset_size[args.dataset],  # type: ignore
    }
    if args.searchspace == "darts":
        searchspace_config.update({"C": 36, "layers": 20})
    profile.set_searchspace_config(searchspace_config)
    profile.configure_trainer(train_portion=0.5)
    config = profile.get_config()

    print(json.dumps(config, indent=2, default=str))

    IS_DEBUG_MODE = True

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        seed=seed,
        debug_mode=IS_DEBUG_MODE,
    )

    trainer = experiment.run_with_profile(profile)

    discrete_profile = DiscreteProfile()
    discret_trainer = experiment.run_discrete_model_with_profile(
        discrete_profile,
        start_epoch=args.start_epoch,
        load_saved_model=args.load_saved_model,
        load_best_model=args.load_best_model,
    )
