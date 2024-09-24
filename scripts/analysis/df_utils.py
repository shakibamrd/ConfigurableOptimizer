from typing import Any
import numpy as np
import pandas as pd
import torch

from confopt.searchspace.darts.core.genotypes import DARTSGenotype

def calculate_mean_std(*dfs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Check if all dataframes have the same shape
    if not all(df.shape == dfs[0].shape for df in dfs):
        raise ValueError("All dataframes must have the same shape.")

    # Check if all dataframes have the same index and columns
    if not all(df.index.equals(dfs[0].index) for df in dfs):
        raise ValueError("All dataframes must have the same index.")

    if not all(df.columns.equals(dfs[0].columns) for df in dfs):
        raise ValueError("All dataframes must have the same columns.")

    # Stack the dataframes together to compute mean and std at the cell level
    stacked = np.stack([df.values for df in dfs], axis=-1)

    # Calculate mean across all dataframes
    mean_df = pd.DataFrame(
        np.mean(stacked, axis=-1), index=dfs[0].index, columns=dfs[0].columns
    )

    # Calculate std deviation across all dataframes
    std_df = pd.DataFrame(
        np.std(stacked, axis=-1), index=dfs[0].index, columns=dfs[0].columns
    )

    return mean_df, std_df


def drop_nan_columns(df: pd.DataFrame) -> pd.DataFrame:
    dropped_columns = df.columns[df.isna().any()].tolist()
    df_clean = df.dropna(axis=1)
    return df_clean, sorted(dropped_columns)


def sort_order_of_columns(*dfs: Any) -> pd.DataFrame:
    # Verify that all dataframes have the same columns (but not necessarily in the same order)
    if not all(set(df.columns) == set(dfs[0].columns) for df in dfs):
        raise ValueError("All dataframes must have the same columns.")

    # Get the union of all columns
    all_columns = set().union(*[df.columns for df in dfs])

    # Reorder columns in each dataframe
    dfs_reordered = [df.reindex(columns=sorted(all_columns)) for df in dfs]

    return dfs_reordered


def clean_dfs(dfs: Any) -> list[pd.DataFrame]:
    """
    Remove columns with NaN values - these are usually the mask columns
    Also remove arch_parameters/0 and arch_parameters/1 because they're dictionaries,
    since we want to find the mean and std dev of different dfs.
    """
    clean_dfs = []

    for df in dfs:
        df_clean, dropped_columns = drop_nan_columns(df)
        df_clean = df_clean.drop(columns=["arch_parameters/0"])

        if "arch_parameters/1" in df_clean.columns:
            df_clean = df_clean.drop(columns=["arch_parameters/1"])

        clean_dfs.append(df_clean)
        print(f"Dropped {len(dropped_columns)} columns have {dropped_columns}")
        # print(df_clean)

    clean_dfs = sort_order_of_columns(*clean_dfs)
    return clean_dfs


def concat_dfs_with_column_prefixes(dfs: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    assert len(dfs) == len(names)
    dfs_ = [df.add_prefix(name + "/") for df, name in zip(dfs, names)]
    return pd.concat(dfs_, axis=1)

def get_alphas(df: pd.DataFrame, cell_idx: int, expected_last_epoch: int=49) -> torch.Tensor:
    last_epoch_df = df.tail(1)

    assert last_epoch_df["_step"].values[0] == expected_last_epoch, \
        f"The last epoch {last_epoch_df['_step'].values[0]} is not the expected one ({expected_last_epoch})"

    alphas = torch.ones(14, 8)
    for edge_idx in range(14):
        for op_idx in range(8):
            alpha_values = f"arch_values/alpha_{cell_idx}.edge_{edge_idx}_op_{op_idx}"
            alphas[edge_idx, op_idx] = last_epoch_df[alpha_values].values[0]

    return alphas

def get_arch_parameters(df: pd.DataFrame, has_reduction_cell: bool, expected_last_epoch: int) -> list[torch.Tensor]:
    arch_params = []

    alphas_normal = get_alphas(df, 0, expected_last_epoch)
    arch_params.append(alphas_normal)

    if has_reduction_cell:
        alphas_reduction = get_alphas(df, 1, expected_last_epoch)
        arch_params.append(alphas_reduction)

    return arch_params

def get_darts_genotype(alphas_normal: torch.Tensor, alphas_reduce: torch.Tensor) -> DARTSGenotype:
    from confopt.searchspace.darts.core.genotypes import PRIMITIVES as primitives

    _steps = 4
    _multiplier = 4

    def _parse(weights: list[torch.Tensor]) -> list[tuple[str, int]]:
        gene = []
        n = 2
        start = 0
        for i in range(_steps):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(
                range(i + 2),
                key=lambda x: -max(
                    W[x][k]
                    for k in range(len(W[x]))  # type: ignore
                    if k != primitives.index("none")
                ),
            )[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != primitives.index("none") and (
                        k_best is None or W[j][k] > W[j][k_best]
                    ):
                        k_best = k
                gene.append((primitives[k_best], j))  # type: ignore
            start = end
            n += 1
        return gene

    gene_normal = _parse(alphas_normal.data.cpu().numpy())
    gene_reduce = _parse(alphas_reduce.data.cpu().numpy())

    concat = range(2 + _steps - _multiplier, _steps + 2)
    genotype = DARTSGenotype(
        normal=gene_normal,
        normal_concat=concat,
        reduce=gene_reduce,
        reduce_concat=concat,
    )
    return genotype.tostr()