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


def get_orthogonal_module_configspace(seed: int) -> Configuration:
    """Return the configuration space with orthogonal modules."""
    cs = ConfigSpace.ConfigurationSpace(seed)

    # Sampler Configuration
    sampler_type = CategoricalHyperparameter(
        "sampler", ["darts", "drnas", "gdas", "snas"], default_value="darts"
    )
    sampler_sample_frequency = CategoricalHyperparameter(
        "sampler_sample_frequency", ["epoch", "step"], default_value="epoch"
    )

    # According to the GDAS paper
    gdas_tau_min = UniformFloatHyperparameter(
        "tau_min", 0.01, 2, default_value=0.1, log=True
    )
    gdas_tau_max = UniformFloatHyperparameter("tau_max", 5, 10, default_value=10)

    # TODO Change this later
    snas_temp_init = UniformFloatHyperparameter(
        "temp_init", 0.1, 5, default_value=2.2, log=True
    )
    snas_temp_min = Constant("temp_min", 0.003)
    snas_temp_annealing = CategoricalHyperparameter(
        "temp_annealing", [True, False], default_value=True
    )

    # Perturbator config
    perturb_type = CategoricalHyperparameter(
        "perturbator", ["random", "adverserial"], default_value="random"
    )
    epsilon = UniformFloatHyperparameter(
        "epsilon", 0.0005, 0.05, default_value=0.03, log=True
    )
    adverserial_loss = Constant("adverserial_criterion", "cross_entropy")
    adverserial_steps = UniformIntegerHyperparameter("steps", 1, 10, default_value=7)
    adverserial_random_start = CategoricalHyperparameter(
        "random_start", [True, False], default_value=True
    )
    perturbator_sample_frequency = CategoricalHyperparameter(
        "perturbator_sample_frequency", ["epoch", "step"], default_value="epoch"
    )

    # Partial_connector config
    is_partial_connector = Constant("is_partial_connector", True)
    edge_normalization = Constant("edge_normalization", 1)
    partial_connector_k = CategoricalHyperparameter("k", [2, 4, 8], default_value=4)

    cs.add_hyperparameters(
        [
            sampler_type,
            sampler_sample_frequency,
            gdas_tau_min,
            gdas_tau_max,
            snas_temp_init,
            snas_temp_min,
            snas_temp_annealing,
            perturb_type,
            perturbator_sample_frequency,
            epsilon,
            adverserial_loss,
            adverserial_steps,
            adverserial_random_start,
            edge_normalization,
            is_partial_connector,
            partial_connector_k,
        ]
    )

    return cs


def get_configspace(seed: int) -> Configuration:  # noqa: PLR0915
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
    gdas_tau_min = UniformFloatHyperparameter(
        "tau_min", 0.01, 2, default_value=0.1, log=True
    )
    gdas_tau_max = UniformFloatHyperparameter("tau_max", 5, 10, default_value=10)

    # TODO Change this later
    snas_temp_init = UniformFloatHyperparameter(
        "temp_init", 0.1, 5, default_value=2.2, log=True
    )
    snas_temp_min = Constant("temp_min", 0.003)
    snas_temp_annealing = CategoricalHyperparameter(
        "temp_annealing", [True, False], default_value=True
    )

    # Perturbator config
    perturb_type = CategoricalHyperparameter(
        "perturbator", ["random", "adverserial"], default_value="random"
    )
    epsilon = UniformFloatHyperparameter(
        "epsilon", 0.0005, 0.05, default_value=0.03, log=True
    )
    adverserial_loss = Constant("adverserial_criterion", "cross_entropy")
    adverserial_steps = UniformIntegerHyperparameter("steps", 1, 10, default_value=7)
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
        "lr", 1e-5, 0.1, default_value=0.025, log=True
    )
    opt_param = CategoricalHyperparameter(
        "optim", choices=["adam", "sgd", "asgd"], default_value="sgd"
    )

    arch_optim_param = CategoricalHyperparameter(
        "arch_optim", choices=["adam", "sgd", "asgd"], default_value="adam"
    )

    # adam params
    opt_beta1 = Constant("opt_beta1", 0.9)
    opt_beta2 = Constant("opt_beta2", 0.999)

    arch_opt_beta1 = Constant("arch_opt_beta1", 0.9)
    arch_opt_beta2 = Constant("arch_opt_beta2", 0.999)

    # sgd params
    opt_momentum = UniformFloatHyperparameter(
        "opt_momentum", 0.5, 0.99, default_value=0.9, log=True
    )
    arch_opt_momentum = UniformFloatHyperparameter(
        "arch_opt_momentum", 0.5, 0.99, default_value=0.9, log=True
    )

    # asgd params
    opt_lambda = Constant("opt_lambda", 1e-4)
    arch_opt_lambda = Constant("arch_opt_lambda", 1e-4)

    # weight decay
    opt_weight_decay = UniformFloatHyperparameter(
        "opt_weight_decay", 3e-4, 0.05, default_value=3e-4, log=True
    )
    arch_opt_weight_decay = UniformFloatHyperparameter(
        "arch_opt_weight_decay", 3e-4, 0.05, default_value=3e-4, log=True
    )

    criterion = Constant("criterion", "cross_entropy")
    batch_size = CategoricalHyperparameter(
        "batch_size", choices=[64, 72, 96], default_value=64
    )
    learning_rate_min = Constant("learning_rate_min", 0)

    cutout = CategoricalHyperparameter(
        "cutout", choices=[True, False], default_value=False
    )
    cutout_length = UniformIntegerHyperparameter(
        "cutout_length", 1, 20, default_value=16
    )

    # train_portion = UniformFloatHyperparameter(
    #     "train_portion", 0.5, 0.8, default_value=0.7
    # )
    train_portion = Constant("train_portion", 0.7)

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
            opt_beta1,
            opt_beta2,
            opt_momentum,
            opt_lambda,
            opt_weight_decay,
            arch_optim_param,
            arch_opt_beta1,
            arch_opt_beta2,
            arch_opt_momentum,
            arch_opt_lambda,
            arch_opt_weight_decay,
            criterion,
            batch_size,
            learning_rate_min,
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

    # Optimizer Conditions
    use_opt_adam_beta1 = ConfigSpace.conditions.EqualsCondition(
        opt_beta1, opt_param, "adam"
    )
    use_opt_adam_beta2 = ConfigSpace.conditions.EqualsCondition(
        opt_beta2, opt_param, "adam"
    )
    use_opt_sgd_momentum = ConfigSpace.conditions.EqualsCondition(
        opt_momentum, opt_param, "sgd"
    )
    use_opt_asgd_lambda = ConfigSpace.conditions.EqualsCondition(
        opt_lambda, opt_param, "asgd"
    )

    # Arch Optimizer Condition
    use_arch_opt_adam_beta1 = ConfigSpace.conditions.EqualsCondition(
        arch_opt_beta1, arch_optim_param, "adam"
    )
    use_arch_opt_adam_beta2 = ConfigSpace.conditions.EqualsCondition(
        arch_opt_beta2, arch_optim_param, "adam"
    )
    use_arch_opt_sgd_momentum = ConfigSpace.conditions.EqualsCondition(
        arch_opt_beta2, arch_optim_param, "sgd"
    )
    use_arch_opt_asgd_lambda = ConfigSpace.conditions.EqualsCondition(
        arch_opt_lambda, arch_optim_param, "asgd"
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
            use_epsilon,
            use_adverserial_config1,
            use_adverserial_config2,
            use_adverserial_config3,
            use_partial_connector,
            use_edge_normalization,
            use_opt_adam_beta1,
            use_opt_adam_beta2,
            use_opt_sgd_momentum,
            use_opt_asgd_lambda,
            use_arch_opt_adam_beta1,
            use_arch_opt_adam_beta2,
            use_arch_opt_sgd_momentum,
            use_arch_opt_asgd_lambda,
        ]
    )

    return cs
