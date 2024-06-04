from __future__ import annotations

import unittest

import pytest

from confopt.searchspace.darts.core.genotypes import Genotype as NB301Genotype

nb301_genotype = NB301Genotype(
    normal=[
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 1),
        ("skip_connect", 0),
        ("skip_connect", 0),
        ("dil_conv_3x3", 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ("max_pool_3x3", 0),
        ("max_pool_3x3", 1),
        ("skip_connect", 2),
        ("max_pool_3x3", 1),
        ("max_pool_3x3", 0),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("max_pool_3x3", 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)

nb301_genotype_fail = NB301Genotype(
    normal=[
        ("skip_connect", 0),
        ("skip_connect", 1),
        ("skip_connect", 0),
        ("skip_connect", 1),
        ("skip_connect", 1),
        ("skip_connect", 0),
        ("skip_connect", 0),
        ("skip_connect", 2),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ("skip_connect", 0),
        ("skip_connect", 1),
        ("skip_connect", 2),
        ("skip_connect", 1),
        ("skip_connect", 0),
        ("skip_connect", 2),
        ("skip_connect", 2),
        ("skip_connect", 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)

test_nb301_acc = 94.166695
test_nb301_fail_acc = 88.165985


class TestBenchmarks(unittest.TestCase):
    @pytest.mark.benchmark  # type: ignore
    def test_nb301_benchmark(self) -> None:
        from confopt.benchmarks import NB301Benchmark

        api = NB301Benchmark()
        query_result = api.query(nb301_genotype, with_noise=False)

        self.assertAlmostEqual(query_result[-1], test_nb301_acc, 4)

    @pytest.mark.benchmark  # type: ignore
    def test_nb301_benchmark_fail_genotype(self) -> None:
        from confopt.benchmarks import NB301Benchmark

        api = NB301Benchmark()
        query_result = api.query(nb301_genotype_fail, with_noise=False)

        self.assertGreater(89.0, query_result[-1])


if __name__ == "__main__":
    unittest.main()
