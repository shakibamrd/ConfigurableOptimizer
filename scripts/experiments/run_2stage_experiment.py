from __future__ import annotations

from enum import Enum

import scipy.stats
import torch
import torch.nn as nn

from collections import namedtuple

from confopt.oneshot.archsampler import DRNASSampler
from confopt.oneshot.archsampler.base_sampler import BaseSampler
from confopt.profiles.profiles import DRNASProfile
from confopt.benchmarks import NB101Benchmark
from confopt.searchspace import NASBench1Shot1SearchSpace
from confopt.train.search_space_handler import SearchSpaceHandler
from confopt.utils import ExperimentCheckpointLoader, Logger
from confopt.dataset import CIFAR10Data

from confopt.searchspace.nb1_shot_1.core.search_spaces.search_space_1 import (
    NB1Shot1Space1,
)
from confopt.searchspace.nb1_shot_1.core.search_spaces.search_space_2 import (
    NB1Shot1Space2,
)
from confopt.searchspace.nb1_shot_1.core.search_spaces.search_space_3 import (
    NB1Shot1Space3,
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

benchmark_api = NB101Benchmark("full")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TrainingMetrics = namedtuple("TrainingMetrics", ["loss", "acc_top1", "acc_top5"])


class SamplerType(Enum):
    DARTS = "darts"
    DRNAS = "drnas"
    GDAS = "gdas"
    SNAS = "snas"
    REINMAX = "reinmax"


def get_sampler(
    arch_params: list[torch.Tensor],
    sampler_type: SamplerType,
    config: dict,
) -> BaseSampler:
    if sampler_type == SamplerType.DRNAS:
        sampler = DRNASSampler(**config, arch_parameters=arch_params)
    return sampler


class AverageMeter:
    def __init__(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_accuracy(output: torch.Tensor, target: torch.Tensor, topk) -> list[float]: # type: ignore
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def eval(network: nn.Module, dataloader) -> TrainingMetrics: # type: ignore
    test_losses, test_top1, test_top5 = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )

    criterion = nn.CrossEntropyLoss().cuda()
    network.eval()

    with torch.no_grad():
        for _step, (test_inputs, test_targets) in enumerate(dataloader):
            test_inputs = test_inputs.cuda()
            test_targets = test_targets.cuda()

            _, logits = network(test_inputs)
            test_loss = criterion(logits, test_targets)

            test_prec1, test_prec5 = calc_accuracy(
                logits.data, test_targets.data, topk=(1, 5)
            )

            test_losses.update(test_loss.item(), test_inputs.size(0))
            test_top1.update(test_prec1.item(), test_inputs.size(0)) # type: ignore
            test_top5.update(test_prec5.item(), test_inputs.size(0)) # type: ignore

    test_metrics = TrainingMetrics(test_losses.avg, test_top1.avg, test_top5.avg)

    return test_metrics


def discretize_arch_params(supernet: nn.Module) -> list[torch.Tensor]:

    new_arch_params = []

    for _, param in enumerate(supernet.arch_parameters):
        new_arch_params.append(torch.zeros_like(param))

    new_arch_params[0][range(5), supernet.arch_parameters[0].argmax(-1)] = 1.0
    new_arch_params[1][0, supernet.arch_parameters[1].argmax(-1)] = 1.0

    if isinstance(supernet.model.search_space, NB1Shot1Space3):
        num_parents_per_node = {
            "0": 0,
            "1": 1,
            "2": 1,
            "3": 1,
            "4": 2,
            "5": 2,
            "6": 2,
        }
        num_inputs = list(num_parents_per_node.values())[2:]
    elif isinstance(supernet.model.search_space, NB1Shot1Space2):
        num_parents_per_node = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 2, "5": 3}
        num_inputs = list(num_parents_per_node.values())[2:]
    elif isinstance(supernet.model.search_space, NB1Shot1Space1):
        num_parents_per_node = {"0": 0, "1": 1, "2": 2, "3": 2, "4": 2, "5": 2}
        num_inputs = list(num_parents_per_node.values())[3:-1]

    import torch.nn.functional as F
    import numpy as np

    def softmax(weights: torch.Tensor, axis: int = -1) -> np.ndarray:
        return F.softmax(torch.Tensor(weights), axis).data.cpu().numpy()

    def get_top_k(array: np.ndarray, k: int) -> list:
        return list(np.argpartition(array[0], -k)[-k:])

    parents = list(
        get_top_k(softmax(alpha, axis=1), num_input)
        for num_input, alpha in zip(num_inputs[:-1], new_arch_params[2:])
    )

    for parent, param in zip(parents, new_arch_params[2:]):
        param[:, parent] = 1.0

    return new_arch_params


###############################################################
###############################################################
###############################################################
###############################################################


def run_2stage_experiment(
    supernet: nn.Module,
    searchspace: str,
    log_path: str,
    runtime: str,
    lora_rank: int,
    lora_warm_epochs: int,
    n_samples: int,
    seed: int=0,
) -> float:
    profile = DRNASProfile(
        epochs=-1,
        sampler_sample_frequency="step",
        lora_rank=lora_rank,
        lora_warm_epochs=lora_warm_epochs,
        seed=seed,
        searchspace_str=searchspace,
    )
    config = profile.get_config()
    sampler = get_sampler(
        supernet.arch_parameters,
        sampler_type=SamplerType(profile.sampler_type),
        config=config.get("sampler"),
    )
    searchspace_handler = SearchSpaceHandler(
        sampler=sampler,
        lora_configs=config.get("lora"),
    )

    logger = Logger(log_dir="logs", custom_log_path=log_path, runtime=runtime)

    load_best_model = False
    load_saved_model = True
    start_epoch = 0

    epoch = None
    if load_best_model is True:
        src = "best"
    elif load_saved_model is True:
        src = "last"
    else:
        src = "epoch"
        epoch = start_epoch

    searchspace_handler.adapt_search_space(supernet)

    if searchspace_handler.lora_configs is not None:
        searchspace_handler.activate_lora(
            supernet, **searchspace_handler.lora_configs
        )  # type: ignore

    dummy_example = torch.randn(2, 3, 64, 64).to(DEVICE)
    supernet(dummy_example.to(DEVICE))

    checkpoint = ExperimentCheckpointLoader.load_checkpoint(logger, src, epoch)
    supernet.load_state_dict(checkpoint["model"])

    data = CIFAR10Data(".", 1, 16, train_portion=0.4)
    dataloaders = data.get_dataloaders(batch_size=1024)
    val_loader = dataloaders[1]

    original_arch_parameters = supernet.arch_parameters

    arch_params = []
    eval_results = []
    benchmark_results = []

    for _ in range(n_samples):
        new_sample_arch_params = [
            sampler.sample(alpha) for alpha in original_arch_parameters
        ]
        supernet.set_arch_parameters(new_sample_arch_params)

        arch_params_discrete = discretize_arch_params(supernet)
        supernet.set_arch_parameters(arch_params_discrete)

        print(arch_params_discrete)
        eval_result = eval(supernet, val_loader)
        genotype = supernet.get_genotype()
        result = benchmark_api.query(genotype)

        arch_params.append(arch_params_discrete)
        eval_results.append(eval_result)
        benchmark_results.append(result)

    results_1 = [result[1] for result in eval_results]
    results_2 = [result["benchmark/test_top1"] for result in benchmark_results]

    corr = scipy.stats.spearmanr(results_1, results_2)
    return corr


if __name__ == "__main__":
    supernet = NASBench1Shot1SearchSpace("S3")
    searchspace = "nb1shot1"
    log_path = "/work/dlclarge2/krishnan-confopt/ConfigurableOptimizer/logs/drnas/nb1shot1/cifar10/0/supernet/"
    runtime = "2024-09-29-16:53:03.227"
    lora_rank = 1
    lora_warm_epochs = 16
    n_samples = 500

    print("Running 2-stage experiment")
    print("Supernet:", supernet)
    print("Searchspace:", searchspace)
    print("Log path:", log_path)
    print("Runtime:", runtime)
    print("Lora rank:", lora_rank)
    print("Lora warm epochs:", lora_warm_epochs)
    print("Number of samples:", n_samples)

    corr = run_2stage_experiment(
        supernet, searchspace, log_path, runtime, lora_rank, lora_warm_epochs, n_samples
    )
    print("Correlation:", corr)
