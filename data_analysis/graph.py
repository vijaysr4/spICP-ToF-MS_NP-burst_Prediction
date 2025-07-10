import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_elements_time_series(
    df,
    time_col="Time (ms)",
    cols=4,
    figsize_base=(30, 3),
    colormap="tab20",
    save_path="elements_time_series.png"
):
    """
    Plot numeric columns against time and save the figure.

    Parameters:
    df : DataFrame with time and numeric columns.
    time_col : str, name of the time column (default: "Time (ms)").
    cols : int, number of columns in the subplot grid (default: 4).
    figsize_base : tuple, (width, height) per row in inches (default: (25, 3)).
    colormap : str, matplotlib colormap name (default: "tab20").
    save_path : str, path to save the PNG image (default: "elements_time_series.png").
    """
    elements = df.select_dtypes(include="number").columns.drop(time_col, errors='ignore')
    n = len(elements)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_base[0], rows * figsize_base[1]),
        sharex=True
    )
    axes = axes.flatten()
    colors = plt.cm.get_cmap(colormap).colors
    for ax, el, color in zip(axes, elements, colors):
        ax.plot(df[time_col], df[el], color=color, linewidth=1)
        ax.set_title(el, fontsize=11)
        ax.set_xlabel(time_col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.5)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)



# Load & sort
# df = pd.read_csv("NPs_BHVO_Oct23_full.csv").sort_values("Time (ms)")

# function call
# plot_elements_time_series(
#     df=df,
#     time_col="Time (ms)",
#     cols=4,
#     figsize_base=(30, 3),
#     colormap="tab20",
#     save_path="Individual_Isotops_plot.png"
# )
