from __future__ import annotations

import argparse
import traceback

import torch
import wandb

from confopt.profiles import ConfigSpaceProfile
from confopt.train import (
    DatasetType,
    Experiment,
    PerturbatorType,
    SamplerType,
    SearchSpaceType,
)
from confopt.utils import get_configspace

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Fine tuning and training searched architectures", add_help=False
    )
    parser.add_argument(
        "--searchspace",
        default="nb201",
        help="search space in (darts, nb201, nb1shot1, tnb101)",
        type=str,
    )
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--logdir", default="./logs", type=str)
    parser.add_argument("--seed", default=444, type=int)
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        default=False,
        help="Load the best model found from the previous run",
    )
    parser.add_argument(
        "--load_saved_model",
        action="store_true",
        default=False,
        help="Load the last saved model in the last run of training them",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        help="Specify the start epoch to continue the training of the model from the \
        previous run",
        type=int,
    )
    parser.add_argument(
        "--num_exp",
        default=5,
        help="Number of random search experiment to perform",
        type=int,
    )
    parser.add_argument(
        "--epochs",
        default=5,
        help="Number of epochs for each experiment run",
        type=int,
    )
    parser.add_argument(
        "--wandb_log",
        action="store_true",
        default=False,
        help="log on wandb",
    )
    args = parser.parse_args()

    cs = get_configspace(seed=args.seed)

    # manage wandb runs
    run_names = {}

    for _ in range(args.num_exp):  # type: ignore
        config = cs.sample_configuration()
        cs_profile = ConfigSpaceProfile(config, epochs=5)
        searchspace = SearchSpaceType(args.searchspace)  # type: ignore
        sampler = SamplerType(config["sampler"])
        perturbator = PerturbatorType(config.get("perturbator", None))
        dataset = DatasetType(args.dataset)  # type: ignore
        edge_normalization = config.get("edge_normalization", False)
        is_partial_connection = config["is_partial_connector"]

        run_name = args.searchspace + "_" + config["sampler"]
        if config["perturbator"] != "none":
            run_name += "_" + config["perturbator"]
        if config["is_partial_connector"]:
            run_name += "_" + "pc"

        if args.wandb_log:  # type: ignore
            run_num = 0
            if run_name not in run_names:
                run_names[run_name] = run_num
            else:
                run_names[run_name] += 1
                run_num = run_names[run_name]
            run_name += "_" + str(run_num)

        experiment = Experiment(
            search_space=searchspace,
            dataset=dataset,
            seed=args.seed,  # type: ignore
            is_wandb_log=args.wandb_log,  # type: ignore
            exp_name=run_name,
            debug_mode=True,
        )
        try:
            experiment.run_with_profile(
                cs_profile,
            )
            if args.wandb_log:  # type: ignore
                wandb.finish()  # type: ignore
            experiment.logger.close()
        except Exception as e:  # noqa: BLE001
            print(e)
            traceback.print_exc()
            if args.wandb_log:  # type: ignore
                print("Check wandb logs to see where the search crashed")
                wandb.finish(1)  # type: ignore
            experiment.logger.close()
            continue
