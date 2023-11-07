##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import IO, Any, NamedTuple

import torch
import wandb


def prepare_logger(
    save_dir: str,
    seed: int,
    exp_name: str,
    xargs: argparse.Namespace | None = None,
) -> Logger:
    logger = Logger(save_dir, seed, exp_name=exp_name)
    logger.log(f"Main Function with logger : {logger}")
    logger.log("Arguments : -------------------------------")

    if xargs is not None:
        for name, value in xargs._get_kwargs():
            logger.log(f"{name:16} : {value}")

    logger.log("Python  Version  : {:}".format(sys.version.replace("\n", " ")))
    logger.log(f"PyTorch Version  : {torch.__version__}")
    logger.log(f"cuDNN   Version  : {torch.backends.cudnn.version()}")
    logger.log(f"CUDA available   : {torch.cuda.is_available()}")
    logger.log(f"CUDA GPU numbers : {torch.cuda.device_count()}")
    logger.log(
        "CUDA_VISIBLE_DEVICES : {:}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]
            if "CUDA_VISIBLE_DEVICES" in os.environ
            else "None"
        )
    )
    return logger


class Logger:
    def __init__(
        self,
        log_dir: str,
        seed: str | int,
        # create_model_dir: bool = True,
        exp_name: str = "",
        search_space: str = "",
        run_time: str | None = None,
        last_run: bool = False,
        is_discrete: bool = False,
    ) -> None:
        self.is_discrete = is_discrete
        """Create a summary writer logging to log_dir."""
        if last_run:
            run_time = self.load_last_run(log_dir, exp_name, search_space, str(seed))
        elif run_time is None:
            run_time = time.strftime("%Y-%d-%h-%H:%M:%S", time.gmtime(time.time()))
        else:
            print("format not correct")

        self.log_dir = Path(log_dir) / exp_name / search_space / str(seed) / run_time
        self.seed = int(seed)

        if is_discrete:
            self.log_dir = self.log_dir / "discrete_net"
        else:
            self.log_dir = self.log_dir / "supernet"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        (Path(self.log_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.log_dir / (
            "tensorboard-{:}".format(time.strftime("%d-%h", time.gmtime(time.time())))
        )

        self.logger_path = self.log_dir / "log"
        self.logger_file = open(self.logger_path, "w")  # noqa: SIM115
        self.writer = None

    def set_up_new_run(self, is_discrete: bool = False) -> None:
        log_dir, exp_name, search_space, seed, run_time, net = self.log_dir.parts
        self.is_discrete = is_discrete
        run_time = time.strftime("%Y-%d-%h-%H:%M:%S", time.gmtime(time.time()))
        self.save_last_run(
            run_time=run_time,
            log_dir=log_dir,
            exp_name=exp_name,
            search_space=search_space,
            seed=seed,
        )
        self.log_dir = Path(log_dir) / exp_name / search_space / seed / run_time
        if is_discrete:
            self.log_dir = self.log_dir / "discrete_net"
        else:
            self.log_dir = self.log_dir / "supernet"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        (Path(self.log_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = self.log_dir / (
            "tensorboard-{:}".format(time.strftime("%d-%h", time.gmtime(time.time())))
        )

        self.logger_path = self.log_dir / "log"
        self.logger_file = open(self.logger_path, "w")  # noqa: SIM115
        self.writer = None

    def load_last_run(
        self, log_dir: str, exp_name: str, search_space: str, seed: str
    ) -> str:
        file_path = Path(log_dir) / exp_name / search_space / seed
        if self.is_discrete:
            file_path = file_path / "last_run_discrete_net"
        else:
            file_path = file_path / "last_run_supernet"
        with open(file_path) as f:
            run_time = f.read().strip()
        return run_time

    def save_last_run(
        self, run_time: str, log_dir: str, exp_name: str, search_space: str, seed: str
    ) -> str:
        file_path = Path(log_dir) / exp_name / search_space / seed
        if self.is_discrete:
            file_path = file_path / "last_run_discrete_net"
        else:
            file_path = file_path / "last_run_supernet"
        with open(file_path, "w") as f:
            f.write(run_time)
        return run_time

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dir={self.log_dir}, writer={self.writer})"

    def path(self, mode: str | None) -> Path:
        valids = (
            "best_model",  # checkpoint containing the best model
            "checkpoints",  # checkpoint of all the checkpoints (periodic)
            "log",  # path to the logger file
            "last_checkpoint",  # return the last checkpoint in the checkpoints folder
            None,
        )

        if mode not in valids:
            raise TypeError(f"Unknow mode = {mode}, valid modes = {valids}")
        if mode == "best_model":
            return self.log_dir / (mode + ".pth")
        if mode == "last_checkpoint":
            last_checkpoint_path = self.log_dir / "checkpoints" / "last_checkpoint"
            with open(last_checkpoint_path) as f:
                return self.log_dir / "checkpoints" / f.read().strip()
        if mode is None:
            return self.log_dir
        return self.log_dir / mode

    def extract_log(self) -> IO[Any]:
        return self.logger_file

    def close(self) -> None:
        self.logger_file.close()
        if self.writer is not None:
            self.writer.close()

    def log(self, string: str, save: bool = True, stdout: bool = False) -> None:
        if stdout:
            sys.stdout.write(string)
            sys.stdout.flush()
        else:
            print(string)
        if save:
            self.logger_file.write(f"{string}\n")
            self.logger_file.flush()

    def log_metrics(
        self,
        title: str,
        metrics: NamedTuple,
        epoch_str: str = "",
        totaltime: float | None = None,
    ) -> None:
        msg = "[{:}] {} : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
            epoch_str,
            title,
            metrics.loss,  # type: ignore
            metrics.acc_top1,  # type: ignore
            metrics.acc_top5,  # type: ignore
        )

        if totaltime is not None:
            msg += f", time-cost={totaltime:.1f} s"

        self.log(msg)

    def wandb_log_metrics(
        self,
        title: str,
        metrics: NamedTuple,
        epoch: int,
        totaltime: float | None = None,
    ) -> None:
        log_metrics = {
            f"{title}/epochs": epoch,
            f"{title}/loss": metrics.loss,  # type: ignore
            f"{title}/acc_top1": metrics.acc_top1,  # type: ignore
            f"{title}/acc_top5": metrics.acc_top5,  # type: ignore
        }
        if totaltime is not None:
            log_metrics.update({f"{title}/time": f"{totaltime:.1f}"})

        wandb.log(log_metrics)  # type: ignore
