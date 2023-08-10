from __future__ import annotations

import unittest

import torch
from torch import nn
from torch import optim

from confopt.oneshot.archsampler import DARTSSampler
from confopt.searchspace import DARTSSearchSpace
from confopt.train import ConfigurableTrainer, BaseProfile
from confopt.dataset import CIFAR10Data
from confopt.utils import prepare_logger


class TestConfTrainer(unittest.TestCase):
    def test_basic_darts(self) -> None:
        epochs = 10

        searchspace = DARTSSearchSpace()
        sampler = DARTSSampler(searchspace.arch_parameters)
        m_optimizer = optim.SGD(searchspace.arch_parameters, lr=1e-3, momentum=0.0, weight_decay=0.1)
        a_optimizer = optim.Adam(searchspace.arch_parameters, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(m_optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        data = CIFAR10Data("~/Documents/dev/ConfigurableOptimizer/data", 0, 0.5)
        logger = prepare_logger(
            save_dir="~/Documents/dev/ConfigurableOptimizer/logs",
            seed=0,
            exp_name="Test"
        )

        profile = BaseProfile(sampler=sampler)

        trainer = ConfigurableTrainer(
            model=searchspace,
            data=data,
            model_optimizer=m_optimizer,
            arch_optimizer=a_optimizer,
            scheduler=lr_scheduler,
            criterion=criterion,
            batchsize=16,
            logger=logger
            )

        trainer.train(profile, epochs)
        
        assert True

if __name__ == "__main__":
    unittest.main()