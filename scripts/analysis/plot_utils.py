import matplotlib.pyplot as plt
import pandas as pd


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
    df_mean: pd.DataFrame, df_std: pd.DataFrame, start_epoch:int=0
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
    plt.title("Line Chart with Confidence Bands", fontsize=16)
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
