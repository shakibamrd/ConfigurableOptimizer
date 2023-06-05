from __future__ import annotations

from typing import Any, Callable, Literal

import torch
from torch.optim.optimizer import Optimizer, required

from confopt.oneshot.archsampler import BaseSampler


class SDARTSSampler(BaseSampler):
    def __init__(
        self,
        arch_parameters: list[torch.Tensor],
        sample_frequency: Literal["epoch", "step"] = "step",
        tau_min: float = 0.1,
        tau_max: float = 10,
        total_epochs: int = 250,
    ) -> None:
        super().__init__(
            arch_parameters=arch_parameters, sample_frequency=sample_frequency
        )

        self.tau_min = torch.Tensor([tau_min])
        self.tau_max = torch.Tensor([tau_max])
        self.total_epochs = total_epochs
        self.tau_curr = self.tau_max - (self.tau_max - self.tau_min) * self._epoch / (
            self.total_epochs - 1
        )

    def set_taus(self, tau_min: float, tau_max: float) -> None:
        self.tau_min = torch.Tensor([tau_min])  # type: ignore
        self.tau_max = torch.Tensor([tau_max])  # type: ignore

    def set_total_epochs(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs

    def sample_alphas(
        self,
        arch_parameters: list[torch.Tensor],
        epsilon: float,
        attack_type: Literal["random", "adverserial"] = "random",
        model: torch.nn.Module = None,
        loss_criterion: torch.nn.Module = None,
        X: torch.Tensor = None,
        target: torch.Tensor = None,
        steps: int = 7,
        random_start: bool = True,
    ) -> list[torch.Tensor]:
        assert attack_type in [
            "random",
            "adverserial",
        ], "attack_type must be either 'random' or 'adverserial'"

        if attack_type == "random":
            sampled_alphas = []
            for alpha in arch_parameters:
                sampled_alphas.append(self.sample_random(alpha.clone(), epsilon))

            for alpha in sampled_alphas:
                for line in alpha:
                    max_index = line.argmax()
                    line.data.clamp_(0, 1)
                    if line.sum() == 0.0:
                        line.data[max_index] = 1.0
                    line.data.div_(line.sum())
            return sampled_alphas

        return self.sample_linf_pgd_alpha(
            model,
            loss_criterion,
            X,
            target,
            arch_parameters,
            epsilon,
            steps,
            random_start,
        )

    def sample_random(self, alpha: torch.Tensor, epsilon: float) -> torch.Tensor:
        alpha.data.add_(torch.zeros_like(alpha).uniform_(-epsilon, epsilon))
        return alpha

    def sample_linf_pgd_alpha(
        self,
        model: torch.nn.Module,
        loss_criterion: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        alphas: list[torch.Tensor],
        epsilon: float,
        steps: int = 7,
        random_start: bool = True,
    ) -> list[torch.Tensor]:
        saved_params = [p.clone() for p in alphas]
        optimizer = LinfSGD(alphas, lr=2 * epsilon / steps)
        with torch.no_grad():
            _, logits = model(X)
            loss_before = loss_criterion(logits, y)

        if random_start:
            for p in alphas:
                p.data.add_(torch.zeros_like(p).uniform_(-epsilon, epsilon))
            self.clip(model)

        for _ in range(steps):
            optimizer.zero_grad()
            model.zero_grad()
            __, logits = model(X)
            loss = loss_criterion(logits, y)
            loss.backward()
            optimizer.step()
            diff = [
                (alphas[i] - saved_params[i]).clamp_(-epsilon, epsilon)
                for i in range(len(saved_params))
            ]
            for i, p in enumerate(alphas):
                p.data.copy_(diff[i] + saved_params[i])
            self.clip(model)

        optimizer.zero_grad()
        model.zero_grad()
        with torch.no_grad():
            _, logits = model(X)
            loss_after = loss_criterion(logits, y)
        if loss_before > loss_after:
            for i, p in enumerate(alphas):
                p.data.copy_(saved_params[i])

        return alphas

    def clip(self, model: torch.nn.Module) -> None:
        for p in model.arch_parameters:
            for line in p:
                max_index = line.argmax()
                line.data.clamp_(0, 1)
                if line.sum() == 0.0:
                    line.data[max_index] = 1.0
                line.data.div_(line.sum())

    def new_epoch(self, *args, **kwargs) -> None:  # type: ignore
        self.tau_curr = self.tau_max - (self.tau_max - self.tau_min) * self._epoch / (
            self.total_epochs - 1
        )

        super().new_epoch(*args, **kwargs)


class LinfSGD(Optimizer):
    def __init__(
        self,
        params: Any,
        lr: Any = required,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure: Callable | None = None) -> torch.Tensor:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = torch.sign(p.grad.data)
                # d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    d_p = d_p.add(momentum, buf) if nesterov else buf

                p.data.add_(-group["lr"], d_p)

        return loss
