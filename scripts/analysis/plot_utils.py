import matplotlib.pyplot as plt
import pandas as pd

from wandb_utils import (
    get_arch_param_grad_norm_by_edge_df,
    get_arch_param_grad_norm_df,
    get_benchmark_test_acc_df,
    get_cell_grad_norm_df,
    get_layer_alignment_scores_all_cells_df,
    get_layer_alignment_scores_first_and_last_cells_df,
    get_mean_gradient_matching_score_df,
    get_skip_connections_df,
)


### Plotting functions ###
def plot_line_chart(df: pd.DataFrame) -> None:
    # Plotting the line chart
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Plot each column as a separate line
    for column in df.columns:
        plt.plot(
            df.index, df[column], label=column, linewidth=2
        )  # Customize line width

    # Add chart title and labels
    plt.title("Line Chart", fontsize=16)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Values", fontsize=12)

    # Customize ticks on the x and y axes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add a legend to identify different columns
    plt.legend(title="Columns", fontsize=10, title_fontsize=12)

    # Add a grid for better readability
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Display the chart
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def plot_line_chart_with_std_dev(
    df_mean: pd.DataFrame, df_std: pd.DataFrame, title: str, start_epoch: int = 0
) -> None:
    # Plotting the line chart
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Plot each column as a separate line with confidence intervals
    for column in df_mean.columns:
        mean = df_mean[column]
        std_dev = df_std[column]

        # Calculate the confidence intervals
        lower_bound = mean - std_dev
        upper_bound = mean + std_dev
        index = df_mean.index

        if start_epoch > 0:
            index = df_mean.index[start_epoch:]
            mean = mean[start_epoch:]
            lower_bound = lower_bound[start_epoch:]
            upper_bound = upper_bound[start_epoch:]

        # Plot the mean line
        plt.plot(index, mean, label=f"{column}", linewidth=2)  # Customize line width

        # Fill the area between the confidence intervals
        plt.fill_between(index, lower_bound, upper_bound, alpha=0.2)

    # Add chart title and labels
    plt.title(title, fontsize=16)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Values", fontsize=12)

    # Customize ticks on the x and y axes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add a legend to identify different columns
    plt.legend(title="Columns", fontsize=10, title_fontsize=12)

    # Add a grid for better readability
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Display the chart
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def plot_everything(
    mean_df: pd.DataFrame, std_df: pd.DataFrame, meta_info: str, start_epoch: int = 0
) -> None:
    # Cell gradients
    mean_cell_grads_dfs = [get_cell_grad_norm_df(mean_df, idx) for idx in range(8)]
    std_cell_grads_dfs = [get_cell_grad_norm_df(std_df, idx) for idx in range(8)]
    mean_cell_grads_df = pd.concat(mean_cell_grads_dfs, axis=1)
    std_cell_grads_df = pd.concat(std_cell_grads_dfs, axis=1)
    plot_line_chart_with_std_dev(
        mean_cell_grads_df, std_cell_grads_df, meta_info, start_epoch
    )

    # Arch gradients
    arch_grad_mean_dfs = [
        get_arch_param_grad_norm_df(mean_df, idx) for idx in ("normal", "reduce")
    ]
    arch_grad_std_dfs = [
        get_arch_param_grad_norm_df(std_df, idx) for idx in ("normal", "reduce")
    ]
    arch_grad_mean_df = pd.concat(arch_grad_mean_dfs, axis=1)
    arch_grad_std_df = pd.concat(arch_grad_std_dfs, axis=1)
    plot_line_chart_with_std_dev(
        arch_grad_mean_df, arch_grad_std_df, meta_info, start_epoch
    )

    # Arch param gradients by edge
    arch_param_row_mean_dfs = [
        get_arch_param_grad_norm_by_edge_df(mean_df, "normal", idx) for idx in range(14)
    ]
    arch_param_row_std_dfs = [
        get_arch_param_grad_norm_by_edge_df(std_df, "normal", idx) for idx in range(14)
    ]
    arch_param_row_mean_df = pd.concat(arch_param_row_mean_dfs, axis=1)
    arch_param_row_std_df = pd.concat(arch_param_row_std_dfs, axis=1)
    plot_line_chart_with_std_dev(
        arch_param_row_mean_df, arch_param_row_std_df, meta_info, start_epoch
    )

    # Skip connections
    skip_connections_mean_dfs = [
        get_skip_connections_df(mean_df, idx) for idx in ("normal", "reduce")
    ]
    skip_connections_std_dfs = [
        get_skip_connections_df(std_df, idx) for idx in ("normal", "reduce")
    ]
    skip_connections_mean_df = pd.concat(skip_connections_mean_dfs, axis=1)
    skip_connections_std_df = pd.concat(skip_connections_std_dfs, axis=1)
    plot_line_chart_with_std_dev(
        skip_connections_mean_df, skip_connections_std_df, meta_info
    )

    # Gradient matching scores
    gm_scores_mean_df = get_mean_gradient_matching_score_df(mean_df)
    gm_scores_std_df = get_mean_gradient_matching_score_df(std_df)
    plot_line_chart_with_std_dev(gm_scores_mean_df, gm_scores_std_df, meta_info)

    # Layer alignment scores
    for cell_type in ("normal", "reduce"):
        mean_layer_alignment_df = get_layer_alignment_scores_all_cells_df(
            mean_df, cell_type
        )
        std_layer_alignment_df = get_layer_alignment_scores_all_cells_df(
            std_df, cell_type
        )
        plot_line_chart_with_std_dev(
            mean_layer_alignment_df, std_layer_alignment_df, meta_info
        )

        mean_layer_alignment_df = get_layer_alignment_scores_first_and_last_cells_df(
            mean_df, cell_type
        )
        std_layer_alignment_df = get_layer_alignment_scores_first_and_last_cells_df(
            std_df, cell_type
        )
        plot_line_chart_with_std_dev(
            mean_layer_alignment_df, std_layer_alignment_df, meta_info
        )

    # Benchmark test accuracy
    mean_test_acc_df = get_benchmark_test_acc_df(mean_df)
    std_test_acc_df = get_benchmark_test_acc_df(std_df)
    plot_line_chart_with_std_dev(mean_test_acc_df, std_test_acc_df, meta_info)
