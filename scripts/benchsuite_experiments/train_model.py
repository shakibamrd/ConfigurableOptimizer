from __future__ import annotations
import os
import argparse

from confopt.profile.profiles import DiscreteProfile
from confopt.train import Experiment
from benchsuite import ( # type: ignore
    configure_discrete_profile_with_search_space,  # type: ignore
    BenchSuiteOpSet, 
    BenchSuiteSpace,
) 
from confopt.enums import SearchSpaceType, DatasetType

WANDB_LOG = True
DEBUG_MODE = False

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cirfar100", "imagenet16", "imgnet16_120", "taskonomy", "aircraft"],
        default="cifar10",
        type=str,
        help="dataset to use",
    )
    parser.add_argument(
        "--optimizer",
        choices=["darts", "gdas", "drnas"],
        default="drnas",
        type=str,
        help="Optimizer which was used to obtain this model",
    )
    parser.add_argument(
        "--subspace",
        choices=["wide", "deep", "single_cell"],
        default="deep",
        type=str,
        help="benchsuit type to use",
    )
    parser.add_argument(
        "--opset",
        choices=["regular", "no_skip", "all_skip"],
        default="no_skip",
        type=str,
        help="benchsuit operation set to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--searchspace",
        type=str,
        default="darts",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--genotypes_folder",
        type=str,
        default="notebooks/genotypes/",
        help="path to the file of the genotype you want to run."
    )
    return parser

def set_profile_genotype(discrete_profile: DiscreteProfile, path: str) -> None:
    with open(path, 'r') as file:
        genotype = file.read()
    discrete_profile.genotype = str(genotype)

if __name__ == "__main__":
    args = parse_args().parse_args()
    discrete_profile = DiscreteProfile(
        searchspace="darts",
        epochs=args.epochs,
        seed=args.seed,
    )
    configure_discrete_profile_with_search_space(
        profile=discrete_profile,
        space=BenchSuiteSpace(args.subspace),
        opset=BenchSuiteOpSet(args.opset),
    )

    genotype_filename = f"{args.optimizer}-{args.subspace}-{args.opset}.txt"
    genotype_filepath = os.path.join(args.genotypes_folder, genotype_filename)

    print("Path to genotype file: ", genotype_filepath)

    experiment = Experiment(
        search_space=SearchSpaceType(args.searchspace),
        dataset=DatasetType(args.dataset),
        seed=args.seed,
        log_with_wandb=WANDB_LOG,
        debug_mode=DEBUG_MODE,
        exp_name=f"{genotype_filepath.split('/')[-1].split('.')[0]}",
    )

    set_profile_genotype(discrete_profile, genotype_filepath)
    experiment.train_discrete_model(discrete_profile)