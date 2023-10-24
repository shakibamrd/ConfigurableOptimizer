from __future__ import annotations

import argparse

import ConfigSpace
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
import torch
import wandb

from confopt.profiles import ProfileConfig
from confopt.train import (
    DatasetType,
    Experiment,
    PerturbatorEnum,
    SamplersEnum,
    SearchSpaceEnum,
)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ADVERSERIAL_DATA = torch.randn(2, 3, 32, 32).to(DEVICE), torch.randint(0, 9, (2,)).to(
    DEVICE
)


def get_loss(loss_str: str) -> torch.nn.Module:
    if loss_str == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    raise Exception(f"The loss type {loss_str} is not supported")


def get_configspace(seed: int) -> Configuration:
    """Returns the configuration space."""
    cs = ConfigSpace.ConfigurationSpace(seed)

    # Sampler Configuration
    sampler_type = CategoricalHyperparameter(
        "sampler", ["darts", "drnas", "gdas", "snas"], default_value="darts"
    )
    sample_frequency = CategoricalHyperparameter(
        "sample_frequency", ["epoch", "step"], default_value="epoch"
    )

    # TODO Change this later
    gdas_tau_min = Constant("tau_min", 0.1)
    gdas_tau_max = Constant("tau_max", 10)

    # TODO Change this later
    snas_temp_init = Constant("temp_init", 1.0)
    snas_temp_min = Constant("temp_min", 0.33)
    snas_temp_annealing = CategoricalHyperparameter(
        "temp_annealing", [True, False], default_value=True
    )
    snas_total_epochs = Constant("total_epochs", 250)

    # Perturbator config
    perturb_type = CategoricalHyperparameter(
        "perturbator", ["random", "adverserial", "none"], default_value="none"
    )
    epsilon = UniformFloatHyperparameter(
        "epsilon", 0.003, 0.3, default_value=0.03, log=True
    )
    adverserial_loss = Constant("adverserial_criterion", "cross_entropy")
    adverserial_steps = UniformIntegerHyperparameter("steps", 7, 30, default_value=20)
    adverserial_random_start = CategoricalHyperparameter(
        "random_start", [True, False], default_value=True
    )

    # Partial_connector config
    is_partial_connector = CategoricalHyperparameter(
        "is_partial_connector", [True, False], default_value=False
    )
    edge_normalization = Constant("edge_normalization", 1)
    partial_connector_k = UniformIntegerHyperparameter("k", 4, 16, default_value=8)

    # Training Configurations
    learning_rate = UniformFloatHyperparameter(
        "lr", 0.00001, 0.1, default_value=0.025, log=True
    )
    # opt_param = CategoricalHyperparameter(
    #     "optim", choices=["adam", "sgd"], default_value="sgd"
    # )
    opt_param = Constant("optim", "sgd")

    # arch_optim_param = CategoricalHyperparameter(
    #     "arch_optim", choices=["adam", "sgd"], default_value="adam"
    # )
    arch_optim_param = Constant("arch_optim", "adam")

    momentum = Constant("momentum", 0.9)
    nesterov = Constant("nesterov", 0)
    criterion = Constant("criterion", "cross_entropy")
    batch_size = CategoricalHyperparameter(
        "batch_size", choices=[64, 72, 96], default_value=64
    )
    learning_rate_min = Constant("learning_rate_min", 0)

    weight_decay = UniformFloatHyperparameter(
        "weight_decay", 3e-4, 0.05, default_value=3e-4, log=True
    )
    cutout = CategoricalHyperparameter(
        "cutout", choices=[True, False], default_value=False
    )
    cutout_length = UniformIntegerHyperparameter(
        "cutout_length", 0, 20, default_value=16
    )

    train_portion = UniformFloatHyperparameter(
        "train_portion", 0.5, 0.8, default_value=0.7
    )
    data_parallel = CategoricalHyperparameter(
        "use_data_parallel", [True, False], default_value=True
    )
    checkpointing_freq = Constant("checkpointing_freq", 5)
    cs.add_hyperparameters(
        [
            sampler_type,
            sample_frequency,
            gdas_tau_min,
            gdas_tau_max,
            snas_temp_init,
            snas_temp_min,
            snas_temp_annealing,
            snas_total_epochs,
            perturb_type,
            epsilon,
            adverserial_loss,
            adverserial_steps,
            adverserial_random_start,
            edge_normalization,
            is_partial_connector,
            partial_connector_k,
            learning_rate,
            opt_param,
            arch_optim_param,
            momentum,
            nesterov,
            criterion,
            batch_size,
            learning_rate_min,
            weight_decay,
            cutout,
            cutout_length,
            train_portion,
            data_parallel,
            checkpointing_freq,
        ]
    )

    # Add conditions to restrict the hyperparameter space
    # Cutout
    use_cutout = ConfigSpace.conditions.EqualsCondition(
        cutout_length,
        cutout,
        True,
    )

    # Sampler GDAS
    use_gdas_config1 = ConfigSpace.conditions.EqualsCondition(
        gdas_tau_min, sampler_type, "gdas"
    )
    use_gdas_config2 = ConfigSpace.conditions.EqualsCondition(
        gdas_tau_max, sampler_type, "gdas"
    )

    # Sampler SNAS
    use_snas_config1 = ConfigSpace.conditions.EqualsCondition(
        snas_temp_init, sampler_type, "snas"
    )
    use_snas_config2 = ConfigSpace.conditions.EqualsCondition(
        snas_temp_min, sampler_type, "snas"
    )
    use_snas_config3 = ConfigSpace.conditions.EqualsCondition(
        snas_temp_annealing, sampler_type, "snas"
    )
    use_snas_config4 = ConfigSpace.conditions.EqualsCondition(
        snas_total_epochs, sampler_type, "snas"
    )
    # Perturb random/ adverserial
    use_epsilon = ConfigSpace.conditions.InCondition(
        epsilon, perturb_type, ["random", "adverserial"]
    )
    use_adverserial_config1 = ConfigSpace.conditions.EqualsCondition(
        adverserial_loss, perturb_type, "adverserial"
    )
    use_adverserial_config2 = ConfigSpace.conditions.EqualsCondition(
        adverserial_random_start, perturb_type, "adverserial"
    )
    use_adverserial_config3 = ConfigSpace.conditions.EqualsCondition(
        adverserial_steps, perturb_type, "adverserial"
    )

    # Partial Connector
    use_partial_connector = ConfigSpace.conditions.EqualsCondition(
        partial_connector_k, is_partial_connector, True
    )
    use_edge_normalization = ConfigSpace.conditions.EqualsCondition(
        edge_normalization, is_partial_connector, True
    )
    # Add  multiple conditions on hyperparameters at once:
    cs.add_conditions(
        [
            use_cutout,
            use_gdas_config1,
            use_gdas_config2,
            use_snas_config1,
            use_snas_config2,
            use_snas_config3,
            use_snas_config4,
            use_epsilon,
            use_adverserial_config1,
            use_adverserial_config2,
            use_adverserial_config3,
            use_partial_connector,
            use_edge_normalization,
        ]
    )

    return cs


def build_profile(cfg: Configuration) -> ProfileConfig:
    class CustomProfile(ProfileConfig):
        def __init__(self, config: Configuration) -> None:
            self.config_dict = config
            super().__init__(config_type=cfg["sampler"])
            self.sampler_type = config["sampler"]
            self.set_partial_connector(config["is_partial_connector"])
            self.set_perturb(config["perturbator"])

        def get_sampler_config(self) -> dict:
            sampler_config = {}
            sampler_config["sample_frequency"] = self.config_dict["sample_frequency"]
            if self.sampler_type == "gdas":
                sampler_config.update(
                    {
                        "tau_min": self.config_dict["tau_min"],
                        "tau_max": self.config_dict["tau_max"],
                    }
                )
            elif self.sampler_type == "snas":
                sampler_config.update(
                    {
                        "temp_init": self.config_dict["temp_init"],
                        "temp_min": self.config_dict["temp_min"],
                        "temp_annealing": self.config_dict["temp_annealing"],
                        "total_epochs": self.config_dict["total_epochs"],
                    }
                )
            return sampler_config

        def get_perturb_config(self) -> dict | None:
            if self.perturb_type == "adverserial":
                perturb_config = {
                    "epsilon": self.config_dict["epsilon"],
                    "data": ADVERSERIAL_DATA,
                    "loss_criterion": torch.nn.CrossEntropyLoss(),
                    "steps": self.config_dict["steps"],
                    "random_start": self.config_dict["random_start"],
                    "sample_frequency": self.config_dict["sample_frequency"],
                }
            elif self.perturb_type == "random":
                perturb_config = {
                    "epsilon": self.config_dict["epsilon"],
                    "sample_frequency": self.config_dict["sample_frequency"],
                }
            else:
                return None
            return perturb_config

        def get_partial_conenctor(self) -> dict | None:
            partial_connector_config = {"k": self.config_dict.get("k", None)}
            return partial_connector_config

        def get_trainer_config(self) -> dict:
            trainer_config = {
                "epochs": 100,
                "lr": self.config_dict["lr"],
                "optim": self.config_dict["optim"],
                "arch_optim": self.config_dict["arch_optim"],
                "momentum": self.config_dict["momentum"],
                "nesterov": self.config_dict["nesterov"],
                "criterion": self.config_dict["criterion"],
                "batch_size": self.config_dict["batch_size"],
                "learning_rate_min": self.config_dict["learning_rate_min"],
                "weight_decay": self.config_dict["weight_decay"],
                "cutout": self.config_dict["cutout"],
                "cutout_length": self.config_dict.get("cutout_length", None),
                "train_portion": self.config_dict["train_portion"],
                "use_data_parallel": self.config_dict["use_data_parallel"],
                "checkpointing_freq": self.config_dict["checkpointing_freq"],
            }
            return trainer_config

    return CustomProfile(cfg)


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
    args = parser.parse_args()

    cs = get_configspace(seed=args.seed)
    config = cs.sample_configuration()

    custom_profile = build_profile(config)

    searchspace = SearchSpaceEnum(args.searchspace)  # type: ignore
    sampler = SamplersEnum(config["sampler"])
    perturbator = PerturbatorEnum(config.get("perturbator", None))
    dataset = DatasetType(args.dataset)  # type: ignore
    edge_normalization = config["edge_normalization"]
    is_partial_connection = config["is_partial_connector"]

    wandb.init(  # type: ignore
        project="Configurable_Optimizers",
        config=custom_profile.get_config(),
    )

    experiment = Experiment(
        search_space=searchspace,
        dataset=dataset,
        sampler=sampler,
        seed=args.seed,  # type: ignore
        perturbator=perturbator,
        edge_normalization=edge_normalization,
        is_partial_connection=is_partial_connection,
        is_wandb_log=False,  # True
    )

    experiment.run_with_profile(custom_profile)
