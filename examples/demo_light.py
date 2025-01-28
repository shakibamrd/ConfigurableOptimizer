from __future__ import annotations

from confopt.profile import GDASProfile
from confopt.train import Experiment
from confopt.enums import SearchSpaceType, DatasetType

if __name__ == "__main__":
    profile = GDASProfile(searchspace=SearchSpaceType.DARTS, epochs=3)
    experiment = Experiment(
        search_space=SearchSpaceType.NB201,
        dataset=DatasetType.CIFAR10,
        seed=9001,
        debug_mode=True,
        exp_name="demo-simple",
    )
    experiment.train_supernet(profile)
