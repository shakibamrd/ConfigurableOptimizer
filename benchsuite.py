from __future__ import annotations

from enum import Enum
from confopt.enums import DatasetType, SearchSpaceType
from confopt.profile import BaseProfile, DARTSProfile
from confopt.searchspace.darts.core.genotypes import PRIMITIVES
from confopt.train import Experiment


class BenchSuiteSpace(Enum):
    SHALLOW_WIDE = "shallow_wide"
    DEEP_NARROW = "deep_narrow"
    SINGLE_CELL = "single_cell"

    def __str__(self) -> str:
        return self.value


class BenchSuiteOpSet(Enum):
    REGULAR = "regular"
    NO_SKIP = "no_skip"
    ALL_SKIP = "all_skip"

    def __str__(self) -> str:
        return self.value


search_space_configs = {
    BenchSuiteSpace.SHALLOW_WIDE: {
        "C": 18,
        "layers": 4,
    },
    BenchSuiteSpace.DEEP_NARROW: {
        "C": 8,
        "layers": 16,
    },
    BenchSuiteSpace.SINGLE_CELL: {
        "C": 26,
        "layers": 1,
        "steps": 8,
    },
}

opset_configs = {
    BenchSuiteOpSet.REGULAR: {"primitives": PRIMITIVES},
    BenchSuiteOpSet.NO_SKIP: {
        "primitives": [prim for prim in PRIMITIVES if prim != "skip_connect"]
    },
    BenchSuiteOpSet.ALL_SKIP: {
        "primitives": PRIMITIVES,
    },
}


def configure_profile_with_search_space(
    profile: BaseProfile,
    space: BenchSuiteSpace,
    opset: BenchSuiteOpSet,
) -> None:
    search_space = search_space_configs[space]
    operations = opset_configs[opset]
    profile.configure_searchspace(**search_space, **operations)

    if opset == BenchSuiteOpSet.ALL_SKIP:
        profile.use_auxiliary_skip_connection = True


if __name__ == "__main__":
    profile = DARTSProfile(
        searchspace=SearchSpaceType.DARTS,
        epochs=10,
    )

    configure_profile_with_search_space(
        profile,
        space=BenchSuiteSpace.SHALLOW_WIDE,
        opset=BenchSuiteOpSet.NO_SKIP,
    )

    profile.configure_searchspace(
        C=4,
    )

    experiment = Experiment(
        search_space=SearchSpaceType.DARTS,
        dataset=DatasetType.CIFAR10,
        seed=9001,
        debug_mode=True,
        exp_name="darts-shallow-wide-no-skip-debug-run",
    )
    experiment.train_supernet(profile)