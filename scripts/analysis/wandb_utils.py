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
