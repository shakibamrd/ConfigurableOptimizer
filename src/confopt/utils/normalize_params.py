from __future__ import annotations

import torch


def normalize_params(
    params: torch.Tensor | list[torch.Tensor], mask: torch.Tensor
) -> torch.Tensor:
    if isinstance(params, list):
        params = torch.stack(params)
    normalized_params = torch.zeros_like(params)
    normalized_params[mask] = params[mask]
    normalized_params_sum = torch.sum(normalized_params, dim=-1, keepdim=True)
    normalized_params_sum[normalized_params_sum == 0] = 1
    normalized_params /= normalized_params_sum
    assert (normalized_params[~mask] == 0.0).all()
    return normalized_params
