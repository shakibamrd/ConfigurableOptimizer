import os
from pathlib import Path
import shutil
import unittest

import torch

from confopt.utils import Logger, prepare_logger
from confopt.utils.channel_shuffle import channel_shuffle


class TestUtils(unittest.TestCase):
    def test_prepare_logger(self) -> None:
        save_dir = Path(".") / "tests" / "logs"
        exp_name = "test_exp"
        logger = prepare_logger(save_dir=save_dir, seed=9001, exp_name=exp_name)
        checkpoints_path = logger.path(mode="checkpoints")
        assert os.path.exists(checkpoints_path)
        assert os.path.exists(logger.logger_path)

        shutil.rmtree(save_dir, ignore_errors=True)
        logger.close()

    def test_logger(self) -> None:
        save_dir = Path(".") / "tests" / "logs"
        exp_name = "test_exp"
        logger = Logger(
            log_dir=str(save_dir), seed="22", exp_name=exp_name, search_space="nb201"
        )

        checkpoints_path = logger.path(mode="checkpoints")
        assert os.path.exists(checkpoints_path)
        assert os.path.exists(logger.logger_path)

        shutil.rmtree(save_dir, ignore_errors=True)
        logger.close()

    def test_channel_shuffle(self) -> None:
        k = 4
        x = torch.randn(2, 8, 3, 5)
        original_x = x.detach().clone()
        shuffled_x = channel_shuffle(x, groups=k)
        assert x.shape == shuffled_x.shape
        assert not torch.allclose(original_x, shuffled_x)


if __name__ == "__main__":
    unittest.main()
