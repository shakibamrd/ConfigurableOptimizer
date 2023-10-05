from __future__ import annotations

from collections import namedtuple
import time

import torch
from torch import nn
from typing_extensions import TypeAlias

from confopt.dataset import AbstractData
from confopt.searchspace import SearchSpace
from confopt.utils import AverageMeter, Logger, calc_accuracy

from .profile import Profile

TrainingMetrics = namedtuple("TrainingMetrics", ["loss", "acc_top1", "acc_top5"])

DataLoaderType: TypeAlias = torch.utils.data.DataLoader
OptimizerType: TypeAlias = torch.optim.Optimizer
LRSchedulerType: TypeAlias = torch.optim.lr_scheduler.LRScheduler
CriterionType: TypeAlias = torch.nn.modules.loss._Loss


class ConfigurableTrainer:
    def __init__(
        self,
        model: SearchSpace,
        data: AbstractData,
        model_optimizer: OptimizerType,
        arch_optimizer: OptimizerType,
        scheduler: LRSchedulerType,
        criterion: CriterionType,
        logger: Logger,
        batchsize: int,
        use_data_parallel: bool = False,
        print_freq: int = 20,
        drop_path_prob: float = 0.1,
        load_saved_model: bool = False,
    ) -> None:
        self.model = model
        self.model_optimizer = model_optimizer
        self.arch_optimizer = arch_optimizer
        self.scheduler = scheduler
        self.data = data
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.logger = logger
        self.criterion = criterion
        self.use_data_parallel = use_data_parallel
        self.print_freq = print_freq
        self.batch_size = batchsize
        self.load_saved_model = load_saved_model
        self.drop_path_prob = drop_path_prob

    def train(self, profile: Profile, epochs: int, is_wandb_log: bool = True) -> None:
        profile.adapt_search_space(self.model)
        if self.use_data_parallel is True:
            network, criterion = self._load_onto_data_parallel(
                self.model, self.criterion
            )
        else:
            network: nn.Module = self.model  # type: ignore
            criterion = self.criterion

        if self.load_saved_model:
            load_model = self._load_model_state_if_exists()
        else:
            load_model = False

        if not load_model:
            self._init_empty_model_state_info()

        start_time = time.time()
        search_time, epoch_time = AverageMeter(), AverageMeter()

        train_loader, val_loader, _ = self.data.get_dataloaders(
            batch_size=self.batch_size,
            n_workers=0,
        )

        for epoch in range(epochs):
            epoch_str = f"{epoch:03d}-{epochs:03d}"

            self._component_new_step_or_epoch(network, sample_frequency="step")

            base_metrics, arch_metrics = self.train_func(
                train_loader,
                val_loader,
                network,
                criterion,
                self.scheduler,
                self.model_optimizer,
                self.arch_optimizer,
                epoch_str,
                self.print_freq,
                self.logger,
            )

            # Logging
            search_time.update(time.time() - start_time)
            self.logger.log_metrics(
                "Search: Model metrics ", base_metrics, epoch_str, search_time.sum
            )

            self.logger.log_metrics(
                "Search: Architecture metrics ", arch_metrics, epoch_str
            )

            valid_metrics = self.valid_func(val_loader, self.model, self.criterion)
            self.logger.log_metrics("Evaluation: ", valid_metrics, epoch_str)

            if is_wandb_log:
                self.logger.wandb_log_metrics(
                    "search/model", base_metrics, epoch, search_time.sum
                )
                self.logger.wandb_log_metrics("search/arch", arch_metrics, epoch)
                self.logger.wandb_log_metrics("eval", valid_metrics, epoch)

            (
                self.valid_losses[epoch],
                self.valid_accs_top1[epoch],
                self.valid_accs_top5[epoch],
            ) = valid_metrics

            (
                self.search_losses[epoch],
                self.search_accs_top1[epoch],
                self.search_accs_top5[epoch],
            ) = base_metrics

            if valid_metrics.acc_top1 > self.valid_accs_top1["best"]:
                self.valid_accs_top1["best"] = valid_metrics.acc_top1
                self.logger.log(
                    f"<<<--->>> The {epoch_str}-th epoch : found the highest "
                    + f"validation accuracy : {valid_metrics.acc_top1:.2f}%."
                )
            #     is_best = True
            # else:
            #     is_best = False

            # save checkpoint
            # self._save_checkpoint(epoch, is_best)

            with torch.no_grad():
                for i, alpha in enumerate(self.model.arch_parameters):
                    self.logger.log(f"alpha {i} is {alpha}")

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

            if self.scheduler is not None:
                self.scheduler.step()

    def train_func(
        self,
        train_loader: DataLoaderType,
        valid_loader: DataLoaderType,
        network: SearchSpace,
        criterion: CriterionType,
        w_scheduler: LRSchedulerType,  # noqa: ARG002  TODO:Fix
        w_optimizer: OptimizerType,
        arch_optimizer: OptimizerType,
        epoch_str: str,  # noqa: ARG002  TODO:Fix
        print_freq: int,
        logger: Logger,  # noqa: ARG002  TODO:Fix
    ) -> tuple[TrainingMetrics, TrainingMetrics]:
        data_time, batch_time = AverageMeter(), AverageMeter()
        base_losses, base_top1, base_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        arch_losses, arch_top1, arch_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        network.train()
        end = time.time()

        for step, (base_inputs, base_targets) in enumerate(train_loader):
            # FIXME: What was the point of this? and is it safe to remove?
            # scheduler.update(None, 1.0 * step / len(xloader))

            self._component_new_step_or_epoch(network, sample_frequency="step")

            arch_inputs, arch_targets = next(iter(valid_loader))

            base_inputs, arch_inputs = base_inputs.to(self.device), arch_inputs.to(
                self.device
            )
            base_targets = base_targets.to(self.device, non_blocking=True)
            arch_targets = arch_targets.to(self.device, non_blocking=True)

            # measure data loading time
            data_time.update(time.time() - end)

            _, logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
            arch_loss.backward()
            arch_optimizer.step()

            self._update_meters(
                inputs=arch_inputs,
                logits=logits,
                targets=arch_targets,
                loss=arch_loss,
                loss_meter=arch_losses,
                top1_meter=arch_top1,
                top5_meter=arch_top5,
            )

            # update the model weights
            w_optimizer.zero_grad()
            arch_optimizer.zero_grad()

            _, logits = network(base_inputs)
            base_loss = criterion(logits, base_targets)
            base_loss.backward()
            # TODO: Does this vary with the one-shot optimizers?
            torch.nn.utils.clip_grad_norm_(network.model_weight_parameters(), 5)
            w_optimizer.step()

            w_optimizer.zero_grad()
            arch_optimizer.zero_grad()

            self._update_meters(
                inputs=base_inputs,
                logits=logits,
                targets=base_targets,
                loss=base_loss,
                loss_meter=base_losses,
                top1_meter=base_top1,
                top5_meter=base_top5,
            )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % print_freq == 0 or step + 1 == len(train_loader):
                # Tstr = f"Time {batch_time.val:.2f} ({batch_time.avg:.2f})" \
                #     +   f"Data {data_time.val:.2f} ({data_time.avg:.2f})"

                # Wstr = f"Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1" \
                #     +   f"{top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f}" \
                #     +   f"({top5.avg:.2f})]"

                # Astr = f"Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1" \
                #     +   f"{top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f}" \
                #     +   f"({top5.avg:.2f})]"

                # logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Astr)
                ...

        base_metrics = TrainingMetrics(base_losses.avg, base_top1.avg, base_top5.avg)
        arch_metrics = TrainingMetrics(arch_losses.avg, arch_top1.avg, arch_top5.avg)

        return base_metrics, arch_metrics

    def valid_func(
        self,
        valid_loader: DataLoaderType,
        network: SearchSpace,
        criterion: CriterionType,
    ) -> TrainingMetrics:
        arch_losses, arch_top1, arch_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        network.eval()

        with torch.no_grad():
            for _step, (arch_inputs, arch_targets) in enumerate(valid_loader):
                # if torch.cuda.is_available():
                #     arch_targets = arch_targets.cuda(non_blocking=True)
                #     arch_inputs = arch_inputs.cuda(non_blocking=True)

                # prediction
                arch_inputs = arch_inputs.to(self.device)
                arch_targets = arch_targets.to(self.device, non_blocking=True)

                _, logits = network(arch_inputs)
                arch_loss = criterion(logits, arch_targets)

                # record
                arch_prec1, arch_prec5 = calc_accuracy(
                    logits.data, arch_targets.data, topk=(1, 5)
                )

                arch_losses.update(arch_loss.item(), arch_inputs.size(0))
                arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
                arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

        return TrainingMetrics(arch_losses.avg, arch_top1.avg, arch_top5.avg)

    def _load_onto_data_parallel(
        self, network: nn.Module, criterion: CriterionType
    ) -> tuple[nn.Module, CriterionType]:
        if torch.cuda.is_available():
            network, criterion = (
                torch.nn.DataParallel(self.model).cuda(),
                criterion.cuda(),
            )

        return network, criterion

    def _load_model_state_if_exists(self) -> bool:
        last_info = self.logger.path("info")

        if last_info.exists():  # automatically resume from previous checkpoint
            self.logger.log(
                f"=> loading checkpoint of the last-info '{last_info}' start"
            )
            last_info = torch.load(last_info)
            self.start_epoch = last_info["epoch"]
            checkpoint = torch.load(last_info["last_checkpoint"])
            self.test_accs_top1 = checkpoint["test_accs_top1"]
            self.test_losses = checkpoint["test_losses"]
            self.test_accs_top5 = checkpoint["test_accs_top5"]
            self.train_losses = checkpoint["train_losses"]
            self.train_accs_top1 = checkpoint["train_accs_top1"]
            self.train_accs_top5 = checkpoint["train_accs_top5"]

            self.model.load_state_dict(checkpoint["model"])
            self.scheduler.load_state_dict(checkpoint["w_scheduler"])
            self.model_optimizer.load_state_dict(checkpoint["w_optimizer"])
            self.logger.log(
                f"=> loading checkpoint of the last-info {last_info}"
                + f"start with {self.start_epoch}-th epoch."
            )

            return True

        self.logger.log(f"=> did not find the last-info file : {last_info}")
        return False

    def _component_new_step_or_epoch(
        self, model: SearchSpace, sample_frequency: str
    ) -> None:
        assert sample_frequency in [
            "epoch",
            "step",
        ], "Sample Frequency should be either epoch or step"
        assert (
            len(model.components) > 0
        ), "There are no oneshot components inside the search space"
        for component in model.components:
            assert (
                component.sample_frequency == model.components[0].sample_frequency
            ), "The argument sample_frequency of all one-shot component should be same"
        if model.components[0].sample_frequency == sample_frequency:
            if sample_frequency == "epoch":
                model.new_epoch()
            elif sample_frequency == "step":
                model.new_step()

    def _init_empty_model_state_info(self) -> None:
        self.start_epoch = 0
        self.valid_losses: dict[int, float] = {}
        self.search_losses: dict[int, float] = {}
        self.search_accs_top1: dict[int, float] = {}
        self.search_accs_top5: dict[int, float] = {}
        self.valid_accs_top1: dict[int | str, float | int] = {"best": -1}
        self.valid_accs_top5: dict[int, float] = {}

    def _update_meters(
        self,
        inputs: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        loss_meter: AverageMeter,
        top1_meter: AverageMeter,
        top5_meter: AverageMeter,
    ) -> None:
        base_prec1, base_prec5 = calc_accuracy(logits.data, targets.data, topk=(1, 5))
        loss_meter.update(loss.item(), inputs.size(0))
        top1_meter.update(base_prec1.item(), inputs.size(0))
        top5_meter.update(base_prec5.item(), inputs.size(0))
