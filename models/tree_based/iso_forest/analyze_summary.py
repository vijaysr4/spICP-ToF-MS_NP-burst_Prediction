from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Directories & file paths
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'
SUMMARY_CSV = RESULTS_DIR / 'evaluation_summary_mono_iso_forest.csv'
ANALYSIS_DIR = RESULTS_DIR / 'analysis_plots'


def load_summary(path: Path) -> pd.DataFrame:
    """Load the evaluation summary CSV into a DataFrame."""
    return pd.read_csv(path)


def plot_heatmap(df: pd.DataFrame, out_dir: Path, n_bins: int = 100) -> None:
    """Generate a heatmap of metrics over time and save to out_dir."""
    metrics = [
        "precision", "recall", "dice_f1", "iou",
        "gt_peak_err_abs", "gt_peak_err_norm", "composite"
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bin and pivot
    df["time_bin"] = pd.cut(df["gt_peak_time"], bins=n_bins)
    pivot = df.groupby("time_bin")[metrics].mean().T

    # Prepare labels and ticks
    bin_edges = pivot.columns.categories
    x_labels = [f"{interval.left:.1f}" for interval in bin_edges]

    # Dynamically size figure: width scales with bins, height with metrics
    fig_width = max(18, n_bins * 0.15)
    fig_height = max(8, len(metrics) * 1.0)
    plt.figure(figsize=(fig_width, fig_height))

    # Plot heatmap with nearest interpolation
    plt.imshow(pivot.values, aspect="auto", origin="lower", interpolation="nearest")
    plt.yticks(range(len(metrics)), metrics)

    # Show xticks sparsely to avoid overcrowding
    step = max(1, n_bins // 10)
    tick_positions = list(range(0, n_bins, step))
    tick_labels = [x_labels[i] for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=90)

    plt.xlabel("Ground‑Truth Peak Time (s)")
    plt.ylabel("Metrics")
    plt.title("Heatmap of Mean Metrics Over Time")
    cbar = plt.colorbar()
    cbar.set_label("Mean Metric Value")
    plt.tight_layout()

    # Save as high-resolution PNG (increase dpi) and vector PDF
    plt.savefig(out_dir / "heatmap_metrics_over_time.png", dpi=300)
    plt.savefig(out_dir / "heatmap_metrics_over_time.pdf")
    plt.close()(out_dir / "heatmap_metrics_over_time.png", dpi=150)
    plt.close()


def main() -> None:
    # Load and print head of the summary
    df = load_summary(SUMMARY_CSV)
    print("=== evaluation_summary.csv head ===")
    print(df.head(), "\n")

    # Plot and save heatmap
    plot_heatmap(df, ANALYSIS_DIR)
    print(f"Heatmap saved to {ANALYSIS_DIR / 'heatmap_metrics_over_time.png'}")


if __name__ == "__main__":
    main()
