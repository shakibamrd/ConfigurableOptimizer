from __future__ import annotations

import argparse
from confopt.profile import (
    DARTSProfile,
    # LambdaDARTSProfile,
    GDASProfile,
    DRNASProfile,
    ReinMaxProfile,
)

from confopt.train import Experiment
from confopt.enums import SearchSpaceType, DatasetType
from benchsuite import (
    BenchSuiteSpace,
    BenchSuiteOpSet,
    configure_profile_with_search_space,
)

parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", type=str)
parser.add_argument("--subspace", type=str)
parser.add_argument("--ops", type=str)
parser.add_argument("--seed", type=int)

args = parser.parse_args()

if __name__ == "__main__":
    DEBUG_MODE = False
    WANDB_LOG = True
    SEARCHSPACE = SearchSpaceType.DARTS
    DATASET = DatasetType.CIFAR10
    DATASET_DIR = "/path/to/datasets"  # UPDATE WITH YOUR DATASET DIR!!!

    subspace = BenchSuiteSpace(args.subspace)
    opset = BenchSuiteOpSet(args.ops)

    profile_classes = {
        "darts": DARTSProfile,
        "gdas": GDASProfile,
        "drnas": DRNASProfile,
    }

    epochs_profiles = {
        "darts": 50,
        "gdas": 250,
        "drnas": 50,
    }

    epochs = epochs_profiles[args.optimizer]
    profile = profile_classes[args.optimizer](
        searchspace=SEARCHSPACE,
        epochs=epochs,
        seed=args.seed,
    )

    configure_profile_with_search_space(
        profile,
        space=subspace,
        opset=opset,
    )

    exp_name = (
        f"{SEARCHSPACE}-{subspace}-{opset}-{profile.SAMPLER_TYPE}-{DATASET}-e{epochs}"
    )

    profile.configure_trainer(
        checkpointing_freq=10,
    )

    profile.configure_extra(
        space=subspace, opset=opset, benchmark=f"{subspace}-{opset}"
    )

    experiment = Experiment(
        search_space=SEARCHSPACE,
        dataset=DATASET,
        seed=args.seed,
        debug_mode=DEBUG_MODE,
        exp_name=exp_name,
        log_with_wandb=WANDB_LOG,
        dataset_dir=DATASET_DIR,
    )
    experiment.train_supernet(profile)
