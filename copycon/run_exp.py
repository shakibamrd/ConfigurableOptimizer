from __future__ import annotations

import argparse
from confopt.profile import (
    DARTSProfile,
    LambdaDARTSProfile,
    GDASProfile,
    DRNASProfile,
    ReinMaxProfile,
)

from confopt.train import Experiment
from confopt.enums import SearchSpaceType, DatasetType

parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int, default=0)
args = parser.parse_args()

if __name__ == "__main__":
    DEBUG_MODE = False
    WANDB_LOG = True
    SEARCHSPACE = SearchSpaceType.DARTS
    DATASET = DatasetType.AIRCRAFT
    EPOCHS = 100
    DATASET_DIR = "/path/to/datasets"  # UPDATE WITH YOUR DATASET DIR!!!

    profile = DARTSProfile(
        searchspace=SEARCHSPACE,
        epochs=EPOCHS,
    )

    profile.configure_searchspace(
        C=18,
        N=4,
        num_classes=30,
    )

    exp_name = f"{SEARCHSPACE}-{profile.SAMPLER_TYPE}-{DATASET}-e{EPOCHS}"

    profile.configure_trainer(
        checkpointing_freq=10,
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
