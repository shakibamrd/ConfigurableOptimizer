import os
from pathlib import Path
import shutil
import unittest

from confopt.utils import prepare_logger


class TestUtils(unittest.TestCase):
    def test_prepare_logger(self) -> None:
        save_dir = Path(".")/"tests"/"temp"
        exp_name = "test_exp"
        logger = prepare_logger(
            save_dir=save_dir,
            seed=9001,
            exp_name=exp_name
        )
        assert logger.log_dir == save_dir / exp_name
        assert os.path.exists(save_dir / "checkpoint" / exp_name)
        assert os.path.exists(logger.logger_path)

        shutil.rmtree(save_dir)

if __name__ == "__main__":
    unittest.main()
