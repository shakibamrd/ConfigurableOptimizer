from __future__ import annotations

import unittest

import torch
from torch import nn, optim

from confopt.dataset import CIFAR10Data
from confopt.oneshot.archsampler import DARTSSampler
from confopt.searchspace import DARTSSearchSpace
from confopt.train import ConfigurableTrainer
from confopt.utils import BaseProfile, prepare_logger


class TestConfTrainer(unittest.TestCase):
    def test_basic_darts(self) -> None:
        epochs = 1

        searchspace = DARTSSearchSpace()
        sampler = DARTSSampler(searchspace.arch_parameters)
        m_optimizer = optim.SGD(searchspace.arch_parameters, lr=1e-3, momentum=0.0, weight_decay=0.1)
        a_optimizer = optim.Adam(searchspace.arch_parameters, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(m_optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        data = CIFAR10Data("datasets", 0, 0.5)
        logger = prepare_logger(
            save_dir="logs",
            seed=0,
            exp_name="Test"
        )

        profile = BaseProfile(samplers=[sampler])

        # FIXME This is hacky but a quick way to check
        # Add a break statement after each loop over loaders (train_func, valid_func)
        # to speed up the process

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

        logger.close()

if __name__ == "__main__":
    unittest.main()