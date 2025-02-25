from __future__ import annotations

from confopt.profile import GDASProfile
from confopt.train import Experiment
from confopt.enums import SearchSpaceType, DatasetType

if __name__ == "__main__":
    profile = GDASProfile(
        searchspace=SearchSpaceType.DARTS,
        epochs=3,
        early_stopper="skip_connection"
    )
    profile.configure_early_stopper(
        max_skip_normal = 0,
        max_skip_reduce = 0,
        min_epochs = 1,
        count_discrete = False,
    )
    experiment = Experiment(
        search_space=SearchSpaceType.DARTS,
        dataset=DatasetType.CIFAR10,
        seed=9001,
        debug_mode=True,
        exp_name="demo-simple",
    )
    experiment.train_supernet(profile)
