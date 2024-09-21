from typing import Any
import numpy as np
import pandas as pd


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
        df_clean = df_clean.drop(columns=["arch_parameters/0", "arch_parameters/1"])
        clean_dfs.append(df_clean)
        print(f"Dropped {len(dropped_columns)} columns have {dropped_columns}")
        # print(df_clean)

    clean_dfs = sort_order_of_columns(*clean_dfs)
    return clean_dfs
