import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def detect_peaks_all_channels(
    df: pd.DataFrame,
    time_col: str = "Time (ms)",
    threshold_std: float = 2.0,
    min_distance: int = 50,
    prominence_std: float = 1.0
) -> tuple[dict, dict]:
    """
    Detect peaks in each numeric channel of the DataFrame.

    Returns two dicts:
      - peaks_dict: column -> peak indices
      - properties_dict: column -> find_peaks properties
    """
    peaks_dict = {}
    properties_dict = {}
    numeric_cols = df.select_dtypes(include="number").columns.drop(time_col, errors='ignore')

    for col in numeric_cols:
        data = df[col].values
        mu, sigma = data.mean(), data.std()
        height = mu + threshold_std * sigma
        prom = prominence_std * sigma

        peaks, props = find_peaks(
            data,
            height=height,
            distance=min_distance,
            prominence=prom
        )
        peaks_dict[col] = peaks
        properties_dict[col] = props

    return peaks_dict, properties_dict


def plot_channels_with_peaks(
    df: pd.DataFrame,
    peaks_dict: dict,
    time_col: str = "Time (ms)",
    cols: int = 4,
    figsize_base: tuple = (30, 3),
    colormap: str = "tab20",
    save_path: str = "elements_with_peaks.png"
) -> None:
    """
    Plot each numeric channel with detected peaks and save the figure.
    Also annotates the total count of peaks across all channels.
    """
    elements = df.select_dtypes(include="number").columns.drop(time_col, errors='ignore')
    n = len(elements)
    rows = (n + cols - 1) // cols

    # Compute total peaks
    total_peaks = sum(len(peaks) for peaks in peaks_dict.values())

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_base[0], rows * figsize_base[1]),
        sharex=True
    )
    axes = axes.flatten()
    colors = plt.cm.get_cmap(colormap).colors

    for ax, col, color in zip(axes, elements, colors):
        t = df[time_col].values
        y = df[col].values
        peaks = peaks_dict.get(col, [])

        ax.plot(t, y, color=color, lw=0.8, alpha=0.9)
        ax.scatter(t[peaks], y[peaks], color='red', marker='x', s=20)
        ax.set_title(f"{col} ({len(peaks)} peaks)", fontsize=11)
        ax.set_xlabel(time_col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.5)

    for ax in axes[n:]:
        ax.axis('off')

    # Overall annotation
    fig.suptitle(f"Detected Peaks Across All Channels: {total_peaks}", fontsize=16, y=1.02)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {save_path} with total {total_peaks} peaks")

