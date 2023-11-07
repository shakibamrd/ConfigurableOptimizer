from __future__ import annotations

import ConfigSpace
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
import torch


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
    sampler_sample_frequency = CategoricalHyperparameter(
        "sampler_sample_frequency", ["epoch", "step"], default_value="epoch"
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
    perturbator_sample_frequency = CategoricalHyperparameter(
        "perturbator_sample_frequency", ["epoch", "step"], default_value="epoch"
    )

    # Partial_connector config
    is_partial_connector = CategoricalHyperparameter(
        "is_partial_connector", [True, False], default_value=False
    )
    edge_normalization = Constant("edge_normalization", 1)
    partial_connector_k = CategoricalHyperparameter("k", [2, 4, 8], default_value=4)

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
            sampler_sample_frequency,
            gdas_tau_min,
            gdas_tau_max,
            snas_temp_init,
            snas_temp_min,
            snas_temp_annealing,
            snas_total_epochs,
            perturb_type,
            perturbator_sample_frequency,
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
