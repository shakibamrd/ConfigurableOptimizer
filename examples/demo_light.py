from __future__ import annotations

from confopt.profiles import GDASProfile
from confopt.train import DatasetType, Experiment, SearchSpaceType

if __name__ == "__main__":
    profile = GDASProfile(epochs=3)
    experiment = Experiment(
        search_space=SearchSpaceType.NB201,
        dataset=DatasetType.CIFAR10,
        seed=9001,
        debug_mode=True,
        exp_name="demo-simple",
    )
    experiment.train_supernet(profile)
