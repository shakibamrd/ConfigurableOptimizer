from __future__ import annotations

import argparse
import copy

import torch
import wandb

from confopt.profiles.profiles import DiscreteProfile
from confopt.train import Experiment
from confopt.train.experiment import DatasetType, SearchSpaceType


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train DARTS Genotype", add_help=False)

    parser.add_argument(
        "--experiment-group",
        help="name of the experiment group",
        type=str,
    )

    parser.add_argument(
        "--genotype-1",
        help="genotype 1 to train",
        type=str,
    )

    parser.add_argument(
        "--genotype-2",
        help="genotype 2 to train",
        type=str,
    )

    parser.add_argument(
        "--genotype-3",
        help="genotype 3 to train",
        type=str,
    )

    parser.add_argument(
        "--genotype-4",
        help="genotype 4 to train",
        type=str,
    )

    parser.add_argument(
        "--dataset",
        help="dataset",
        type=str,
    )

    parser.add_argument(
        "--project-name",
        default="lora-darts-iclr",
        help="project name for wandb logging",
        type=str,
    )

    parser.add_argument(
        "--seed",
        default=100,
        help="random seed",
        type=int,
    )

    parser.add_argument(
        "--comments",
        default="None",
        help="Any additional comments",
        type=str,
    )

    parser.add_argument(
        "--meta-info",
        default="None",
        help="Any meta information about this run",
        type=str,
    )

    parser.add_argument(
        "--debug-mode", action="store_true", help="run experiment in debug mode"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()

    searchspace = "darts"
    assert args.dataset in [
        "cifar10",
        "cifar100",
        "imgnet16",
        "imgnet16_120",
    ], f"Soes not support dataset of type {args.dataset}"  # type: ignore

    genotypes = [args.genotype_1, args.genotype_2, args.genotype_3, args.genotype_4]
    best_genotype = -1
    best_acc = 0
    for i, genotype in enumerate(genotypes):
        torch.cuda.empty_cache()
        print(f"Training genotype for 100 epochs: {genotype}")

        profile = DiscreteProfile(epochs=100, use_ddp=False, train_portion=0.9)
        exp_type = f"DISCRETE_{searchspace}-{args.dataset}_seed{args.seed}_genotype_{i}"
        profile.genotype = genotype

        config = copy.deepcopy(profile.get_trainer_config())
        config.update({"genotype": profile.get_genotype()})

        config.update(
            {
                "project_name": args.project_name,
                "extra:comments": args.comments,
                "extra:experiment-name": exp_type,
                "extra:is-debug": args.debug_mode,
                "extra:meta-info": args.meta_info,
            }
        )

        # instantiate wandb run
        wandb.init(  # type: ignore
            name=exp_type,
            project=args.project_name,
            group=args.experiment_group,
            config=config,
        )

        experiment = Experiment(
            search_space=SearchSpaceType(searchspace),
            dataset=DatasetType(args.dataset),
            seed=args.seed,
            debug_mode=args.debug_mode,
            exp_name=exp_type,
            is_wandb_log=True,
        )

        # experiment.init_ddp()

        discrete_trainer = experiment.train_discrete_model(profile)

        # experiment.cleanup_ddp()

        _, val_loader, _ = discrete_trainer.data.get_dataloaders(
            batch_size=discrete_trainer.batch_size,
            n_workers=0,
            use_distributed_sampler=discrete_trainer.use_ddp,
        )
        val_metrics = discrete_trainer.evaluate(
            val_loader, discrete_trainer.model, discrete_trainer.criterion
        )
        acc = val_metrics.acc_top1
        if acc > best_acc:
            best_acc = acc
            best_genotype = i

        if i == 3:
            print(
                f"Genotype {best_genotype + 1} is the best genotype: ",
                genotypes[best_genotype],
            )
        wandb.finish() # type: ignore
