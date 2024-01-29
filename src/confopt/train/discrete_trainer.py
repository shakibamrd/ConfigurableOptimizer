from __future__ import annotations

from collections import namedtuple
import time

from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
import torch
from torch import nn
from typing_extensions import TypeAlias

from confopt.dataset import AbstractData
from confopt.searchspace import SearchSpace
from confopt.utils import AverageMeter, Logger, calc_accuracy

TrainingMetrics = namedtuple("TrainingMetrics", ["loss", "acc_top1", "acc_top5"])

DataLoaderType: TypeAlias = torch.utils.data.DataLoader
OptimizerType: TypeAlias = torch.optim.Optimizer
LRSchedulerType: TypeAlias = torch.optim.lr_scheduler.LRScheduler
CriterionType: TypeAlias = torch.nn.modules.loss._Loss


class DiscreteTrainer:
    def __init__(
        self,
        model: nn.Module,
        data: AbstractData,
        model_optimizer: OptimizerType,
        scheduler: LRSchedulerType,
        criterion: CriterionType,
        logger: Logger,
        batch_size: int,
        use_data_parallel: bool = False,
        print_freq: int = 20,
        drop_path_prob: float = 0.1,
        load_saved_model: bool = False,
        load_best_model: bool = False,
        start_epoch: int = 0,
        checkpointing_freq: int = 20,
        epochs: int = 100,
    ) -> None:
        self.model = model
        self.model_optimizer = model_optimizer
        self.scheduler = scheduler
        self.data = data
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.logger = logger
        self.criterion = criterion
        self.use_data_parallel = use_data_parallel
        self.print_freq = print_freq
        self.batch_size = batch_size
        self.drop_path_prob = drop_path_prob
        self.load_best_model = load_best_model
        self.load_saved_model = load_saved_model
        self.load_best_model = load_best_model
        self.start_epoch = start_epoch
        self.checkpointing_freq = checkpointing_freq
        self.epochs = epochs

    def train(self, epochs: int, is_wandb_log: bool = True) -> None:
        self.epochs = epochs
        self.model = self.model.discretize()  # type: ignore

        self._init_empty_model_state_info()
        if self.load_saved_model or self.load_best_model or self.start_epoch != 0:
            self._load_model_state_if_exists()

        if self.use_data_parallel is True:
            network, criterion = self._load_onto_data_parallel(
                self.model, self.criterion
            )
        else:
            network: nn.Module = self.model  # type: ignore
            criterion = self.criterion

        start_time = time.time()
        search_time, epoch_time = AverageMeter(), AverageMeter()

        train_loader, val_loader, _ = self.data.get_dataloaders(
            batch_size=self.batch_size,
            n_workers=0,
        )

        for epoch in range(self.start_epoch, epochs):
            epoch_str = f"{epoch:03d}-{epochs:03d}"

            base_metrics = self.train_func(
                train_loader,
                network,
                criterion,
                self.print_freq,
            )

            # Logging
            search_time.update(time.time() - start_time)
            self.logger.log_metrics(
                "[Discrete] Train: Model/Network metrics ",
                base_metrics,
                epoch_str,
                search_time.sum,
            )

            valid_metrics = self.valid_func(val_loader, self.model, self.criterion)
            self.logger.log_metrics("[Discrete] Evaluation: ", valid_metrics, epoch_str)

            if is_wandb_log:
                self.logger.wandb_log_metrics(
                    "discrete/train/model", base_metrics, epoch, search_time.sum
                )
                self.logger.wandb_log_metrics("discrete/eval", valid_metrics, epoch)

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

            checkpointables = self._get_checkpointables(epoch=epoch)
            self.periodic_checkpointer.step(
                iteration=epoch, checkpointables=checkpointables
            )

            if valid_metrics.acc_top1 > self.valid_accs_top1["best"]:
                self.valid_accs_top1["best"] = valid_metrics.acc_top1
                self.logger.log(
                    f"<<<--->>> The {epoch_str}-th epoch : found the highest "
                    + f"validation accuracy : {valid_metrics.acc_top1:.2f}%."
                )

                self.best_model_checkpointer.save(
                    name="best_model_discrete", checkpointables=checkpointables
                )

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

            if self.scheduler is not None:
                self.scheduler.step()

    def train_func(
        self,
        train_loader: DataLoaderType,
        network: SearchSpace,
        criterion: CriterionType,
        print_freq: int,
    ) -> TrainingMetrics:
        data_time, batch_time = AverageMeter(), AverageMeter()
        base_losses, base_top1, base_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        network.train()
        end = time.time()

        for step, (base_inputs, base_targets) in enumerate(train_loader):
            # FIXME: What was the point of this? and is it safe to remove?
            # scheduler.update(None, 1.0 * step / len(xloader))

            base_inputs = base_inputs.to(self.device)
            base_targets = base_targets.to(self.device, non_blocking=True)

            # measure data loading time
            data_time.update(time.time() - end)

            self.model_optimizer.zero_grad()
            _, logits = network(base_inputs)
            base_loss = criterion(logits, base_targets)
            base_loss.backward()

            if isinstance(network, torch.nn.DataParallel):
                torch.nn.utils.clip_grad_norm_(
                    network.module.model_weight_parameters(), 5
                )
            else:
                torch.nn.utils.clip_grad_norm_(network.model_weight_parameters(), 5)

            self.model_optimizer.step()

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
                # TODO: what is this doing ?
                ...

        base_metrics = TrainingMetrics(base_losses.avg, base_top1.avg, base_top5.avg)
        return base_metrics

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
            for _step, (valid_inputs, valid_targets) in enumerate(valid_loader):
                # prediction
                valid_inputs = valid_inputs.to(self.device)
                valid_targets = valid_targets.to(self.device, non_blocking=True)

                _, logits = network(valid_inputs)
                valid_loss = criterion(logits, valid_targets)

                # record
                valid_prec1, valid_prec5 = calc_accuracy(
                    logits.data, valid_targets.data, topk=(1, 5)
                )

                arch_losses.update(valid_loss.item(), valid_inputs.size(0))
                arch_top1.update(valid_prec1.item(), valid_inputs.size(0))
                arch_top5.update(valid_prec5.item(), valid_inputs.size(0))

        return TrainingMetrics(arch_losses.avg, arch_top1.avg, arch_top5.avg)

    def test(self, is_wandb_log: bool = True) -> TrainingMetrics:
        test_losses, test_top1, test_top5 = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        if self.use_data_parallel is True:
            network, criterion = self._load_onto_data_parallel(
                self.model, self.criterion
            )
        else:
            network: nn.Module = self.model  # type: ignore
            criterion = self.criterion
        network.eval()

        *_, test_loader = self.data.get_dataloaders(
            batch_size=self.batch_size,
            n_workers=0,
        )

        with torch.no_grad():
            for _step, (test_inputs, test_targets) in enumerate(test_loader):
                test_inputs = test_inputs.to(self.device)
                test_targets = test_targets.to(self.device, non_blocking=True)

                _, logits = network(test_inputs)
                test_loss = criterion(logits, test_targets)

                test_prec1, test_prec5 = calc_accuracy(
                    logits.data, test_targets.data, topk=(1, 5)
                )

                test_losses.update(test_loss.item(), test_inputs.size(0))
                test_top1.update(test_prec1.item(), test_inputs.size(0))
                test_top5.update(test_prec5.item(), test_inputs.size(0))

        test_metrics = TrainingMetrics(test_losses.avg, test_top1.avg, test_top5.avg)
        if is_wandb_log:
            self.logger.wandb_log_metrics("discrete/test", test_metrics)

        self.logger.log_metrics("[Discrete] Test", test_metrics, epoch_str="---")

        return test_metrics

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

    def _get_checkpointables(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "search_losses": self.search_losses,
            "search_accs_top1": self.search_accs_top1,
            "search_accs_top5": self.search_accs_top5,
            "valid_losses": self.valid_losses,
            "valid_accs_top1": self.valid_accs_top1,
            "valid_accs_top5": self.valid_accs_top5,
        }

    def _set_up_checkpointer(self, mode: str | None) -> Checkpointer:
        checkpoint_dir = self.logger.path(mode=mode)  # todo: check this
        # checkpointables = self._get_checkpointables(self.start_epoch)
        checkpointer = Checkpointer(
            model=self.model,
            save_dir=str(checkpoint_dir),
            save_to_disk=True,
        )
        checkpointer.add_checkpointable("w_scheduler", self.scheduler)
        checkpointer.add_checkpointable("w_optimizer", self.model_optimizer)
        return checkpointer

    def _init_periodic_checkpointer(self) -> None:
        self.checkpointer = self._set_up_checkpointer(mode="checkpoints")
        self.periodic_checkpointer = PeriodicCheckpointer(
            checkpointer=self.checkpointer,
            period=self.checkpointing_freq,
            max_iter=self.epochs,
        )

    def _set_checkpointer_info(self, last_checkpoint: dict) -> None:
        self.model.load_state_dict(last_checkpoint["model"])

    def _get_checkpoint_dir(self) -> str:
        assert (
            sum([self.load_best_model, self.load_saved_model, (self.start_epoch > 0)])
            <= 1
        )
        mode = None

        if self.load_saved_model or self.load_best_model:
            mode = "checkpoints"
        return self.logger.path(mode=mode)

    def _load_model_state_if_exists(self) -> None:
        checkpoint_dir = self._get_checkpoint_dir()
        checkpointer = Checkpointer(
            model=self.model,
            save_dir=checkpoint_dir,
            save_to_disk=True,
        )

        if self.load_best_model:
            last_info = self.logger.path("best_model_discrete")
            info = checkpointer._load_file(f=last_info)
            self.logger.log(
                f"=> loading checkpoint of the best-model '{last_info}' start"
            )
        elif self.start_epoch != 0:
            last_info = self.logger.path("checkpoints")
            last_info = "{}/{}_{:07d}.pth".format(last_info, "model", self.start_epoch)
            info = checkpointer._load_file(f=last_info)
            self.logger.log(
                f"resume from discrete network trained from {self.start_epoch} epochs"
            )
        elif self.load_saved_model:
            last_info = self.logger.path("last_checkpoint")
            info = checkpointer._load_file(f=last_info)
            self.logger.log(f"=> loading checkpoint of the last-info {last_info}")
        else:
            self.logger.log("=> did not find the any file")
            return

        self.logger.set_up_new_run()
        self.checkpointer.save_dir = self.logger.path(mode="checkpoints")
        self.best_model_checkpointer.save_dir = self.logger.path(mode=None)
        self._set_checkpointer_info(info)

    def _init_empty_model_state_info(self) -> None:
        self.search_losses: dict[int, float] = {}
        self.search_accs_top1: dict[int, float] = {}
        self.search_accs_top5: dict[int, float] = {}
        self.valid_losses: dict[int, float] = {}
        self.valid_accs_top1: dict[int | str, float | int] = {"best": -1}
        self.valid_accs_top5: dict[int, float] = {}

        self._init_periodic_checkpointer()
        self.best_model_checkpointer = self._set_up_checkpointer(mode=None)
        self.logger.set_up_run()

    def _load_onto_data_parallel(
        self, network: nn.Module, criterion: CriterionType
    ) -> tuple[nn.Module, CriterionType]:
        if torch.cuda.is_available():
            network, criterion = (
                torch.nn.DataParallel(self.model).cuda(),
                criterion.cuda(),
            )

        return network, criterion
