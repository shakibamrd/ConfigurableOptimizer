from __future__ import annotations

import unittest

import torch

from confopt.oneshot.archsampler import (
    BaseSampler,
    DARTSSampler,
    DRNASSampler,
    GDASSampler,
    SNASSampler
)
from confopt.searchspace import NASBench201SearchSpace


class TestArchSamplers(unittest.TestCase):

    def _sampler_new_step_or_epoch(self,
        sampler: BaseSampler,
        sample_frequency: str
        ) -> None:
        if sample_frequency == "epoch":
            sampler.new_epoch()
        elif sample_frequency == "step":
            sampler.new_step()
        else:
            raise ValueError(f"Unknown sample_frequency: {sample_frequency}")

    def assert_rows_one_hot(self, alphas: list[torch.Tensor]) -> None:
        for row in alphas:
            assert torch.sum(row == 1.0) == 1
            assert torch.sum(row == 0.0) == row.numel() - 1

    def test_darts_sampler(self) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = DARTSSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        # assert that the tensors are close
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert torch.allclose(arch_param_before, arch_param_after)

    def _test_darts_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = DARTSSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        # assert that the tensors are close
        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert torch.allclose(arch_param_before, arch_param_after)

    def test_darts_sampler_new_step(self) -> None:
       self._test_darts_sampler_new_step_epoch(sample_frequency="step")

    def test_darts_sampler_new_epoch(self) -> None:
        self._test_darts_sampler_new_step_epoch(sample_frequency="epoch")

    def test_gdas_sampler(self) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = GDASSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
            self.assert_rows_one_hot(arch_param_after)

    def _test_gdas_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = GDASSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
            self.assert_rows_one_hot(arch_param_after)

    def test_gdas_sampler_new_step(self) -> None:
        self._test_gdas_sampler_new_step_epoch(sample_frequency="step")

    def test_gdas_sampler_new_epoch(self) -> None:
        self._test_gdas_sampler_new_step_epoch(sample_frequency="epoch")

    def test_drnas_sampler(self) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = DRNASSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

            for row in arch_param_after:
                assert torch.allclose(torch.sum(row), torch.Tensor([1.]))

    def _test_drnas_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = DRNASSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

            for row in arch_param_after:
                assert torch.allclose(torch.sum(row), torch.Tensor([1.]))

    def test_drnas_sampler_new_step(self) -> None:
        self._test_drnas_sampler_new_step_epoch(sample_frequency="step")

    def test_drnas_sampler_new_epoch(self) -> None:
        self._test_drnas_sampler_new_step_epoch(sample_frequency="epoch")

    def test_snas_sampler(self) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = SNASSampler(arch_parameters=searchspace.arch_parameters)

        alphas_before = searchspace.arch_parameters
        alphas_after = sampler.sample_alphas(alphas_before)

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)
    
    def _test_snas_sampler_new_step_epoch(self, sample_frequency: str) -> None:
        searchspace = NASBench201SearchSpace(N=1)
        sampler = SNASSampler(
            arch_parameters=searchspace.arch_parameters,
            sample_frequency=sample_frequency
        )

        alphas_before = searchspace.arch_parameters
        self._sampler_new_step_or_epoch(sampler, sample_frequency)
        alphas_after = sampler.sampled_alphas

        for arch_param_before, arch_param_after in zip(alphas_before, alphas_after):
            assert not torch.allclose(arch_param_before, arch_param_after)

    def test_snas_sampler_new_step(self) -> None:
        self._test_snas_sampler_new_step_epoch(sample_frequency="step")

    def test_snas_sampler_new_epoch(self) -> None:
        self._test_snas_sampler_new_step_epoch(sample_frequency="epoch")
    
    def test_illegal_sample_frequency(self) -> None:
        arch_parameters = [torch.randn(5, 5)]
        with self.assertRaises(AssertionError):
            DARTSSampler(
                arch_parameters=arch_parameters,
                sample_frequency="illegal"
            )

        with self.assertRaises(AssertionError):
            GDASSampler(
                arch_parameters=arch_parameters,
                sample_frequency="illegal"
            )

        with self.assertRaises(AssertionError):
            DRNASSampler(
                arch_parameters=arch_parameters,
                sample_frequency="illegal"
            )

        with self.assertRaises(AssertionError):
            SNASSampler(
                arch_parameters=arch_parameters,
                sample_frequency="illegal"
            )

if __name__ == "__main__":
    unittest.main()