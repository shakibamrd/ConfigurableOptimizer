from __future__ import annotations

from typing import Any
import wandb
import pandas as pd


def fetch_runs(
    filter_criteria: dict,
    entity: str = "confopt-team",
    project: str = "iclr-experiments",
) -> list:
    # Initialize wandb API
    api = wandb.Api(timeout=120)  # type: ignore

    # Fetch all runs that match the filter
    runs = api.runs(f"{entity}/{project}", filters=filter_criteria)

    return runs


def make_wandb_filters(
    state: str,
    *,
    meta_info: str | None = None,
    lora_rank: int | None = None,
    lora_warmup: int | None = None,
    oles: bool | None = None,
    oles_threshold: float | None = None,
    prune_epochs: int | None = None,
    prune_fractions: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    filters: dict[str, Any] = {
        "state": state,
    }

    if meta_info is not None:
        filters.update(
            {
                "config.extra:meta-info": meta_info,
            }
        )

    if lora_rank is not None:
        filters.update(
            {
                "config.lora.r": lora_rank,
            }
        )

    if lora_warmup is not None:
        filters.update(
            {
                "config.lora_extra.warm_epochs": lora_warmup,
            }
        )

    if oles is not None:
        filters.update(
            {
                "config.oles.oles": oles,
            }
        )

    if oles_threshold is not None:
        filters.update(
            {
                "config.oles.threshold": oles_threshold,
            }
        )

    if prune_epochs is not None:
        filters.update(
            {
                "config.pruner.prune_epochs": prune_epochs,
            }
        )

    if prune_fractions is not None:
        filters.update(
            {
                "config.pruner.prune_fractions": prune_fractions,
            }
        )

    if seed is not None:
        filters.update(
            {
                "config.trainer.seed": seed,
            }
        )

    return filters


def get_columns(df: pd.DataFrame, str_filters: list[str]) -> list:
    columns = [c for c in df.columns if all(f in c for f in str_filters)]
    return sorted(columns)


cell_type_to_idx = {"normal": 0, "reduce": 1}


def get_df_with_columns(df: pd.DataFrame, str_filters: list[str]) -> pd.DataFrame:
    columns = get_columns(df, str_filters)
    return df[columns]


def get_normalized_arch_values_by_edge(
    df: pd.DataFrame, cell_type: str, edge_idx: int
) -> pd.DataFrame:
    str_filters = [
        "arch_values",
        f"alpha_{cell_type_to_idx[cell_type]}",
        f"edge_{edge_idx}",
    ]

    return get_df_with_columns(df, str_filters)


def get_normalized_arch_values_by_op(
    df: pd.DataFrame, cell_type: str, op_idx: int
) -> pd.DataFrame:
    str_filters = [
        "arch_values",
        f"alpha_{cell_type_to_idx[cell_type]}",
        f"op_{op_idx}",
    ]

    return get_df_with_columns(df, str_filters)


def get_cell_grad_norm(df: pd.DataFrame, cell_idx: int) -> pd.DataFrame:
    str_filters = [
        "gradient_stats/",
        f"cell_{cell_idx}_grad_norm",
    ]

    return get_df_with_columns(df, str_filters)


def get_arch_param_grad_norm(df: pd.DataFrame, cell_type: str) -> pd.DataFrame:
    str_filters = [
        "gradient_stats/",
        f"arch_param_{cell_type_to_idx[cell_type]}_grad_norm",
    ]

    return get_df_with_columns(df, str_filters)


def get_arch_param_grad_norm_by_edge(
    df: pd.DataFrame, cell_type: str, edge_idx: int
) -> pd.DataFrame:
    str_filters = [
        "gradient_stats/",
        f"arch_param_{cell_type_to_idx[cell_type]}_row_{edge_idx}_grad_norm",
    ]

    return get_df_with_columns(df, str_filters)


def get_skip_connections(df: pd.DataFrame, cell_type: str) -> pd.DataFrame:
    str_filters = [
        f"skip_connections/{cell_type}",
    ]

    return get_df_with_columns(df, str_filters)


def get_mean_gradient_matching_score(df: pd.DataFrame) -> pd.DataFrame:
    str_filters = [
        "gm_scores/mean_gm",
    ]

    return get_df_with_columns(df, str_filters)


def get_benchmark_test_acc(df: pd.DataFrame) -> pd.DataFrame:
    str_filters = [
        "benchmark/test_top1",
    ]

    return get_df_with_columns(df, str_filters)


def get_layer_alignment_scores_all_cells(
    df: pd.DataFrame, cell_type: str
) -> pd.DataFrame:
    str_filters = [
        "layer_alignment_scores/mean/",
        cell_type,
    ]

    return get_df_with_columns(df, str_filters)


def get_layer_alignment_scores_first_and_last_cells(
    df: pd.DataFrame, cell_type: str
) -> pd.DataFrame:
    str_filters = [
        "layer_alignment_scores/first_last/",
        cell_type,
    ]

    return get_df_with_columns(df, str_filters)
