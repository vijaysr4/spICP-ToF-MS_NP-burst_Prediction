#!/usr/bin/env python
"""
vis_fingerprints_barplot.py

Sample N windows and draw a bar‐chart of ion‐percentage composition for each,
laid out in a compact grid of subplots for better space usage.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot colourful bar‐charts of ion percentages for sampled windows in a grid"
    )
    parser.add_argument(
        "--fingerprints_csv", type=Path, required=True,
        help="CSV with columns ['start_ms','end_ms',<ion channels...>]"
    )
    parser.add_argument(
        "--sample_n", type=int, default=6,
        help="Number of windows to sample (default: 6)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("./plots"),
        help="Directory to save output PNG"
    )
    parser.add_argument(
        "--out_file", type=str, default="fingerprints_bars_grid.png",
        help="Filename for the output PNG"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution in DPI"
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.fingerprints_csv)
    feature_cols = [c for c in df.columns if c not in ("start_ms", "end_ms")]

    # Sample windows
    sampled = df.sample(n=args.sample_n, random_state=args.seed).reset_index(drop=True)

    # Generate distinct colours for each channel
    cmap = plt.get_cmap("tab20")
    colors = {col: cmap(i / len(feature_cols)) for i, col in enumerate(feature_cols)}

    # Determine grid size (nearly square)
    n = args.sample_n
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))

    # Create subplots grid
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 4),
        sharey=True,
        constrained_layout=True
    )
    # flatten axes for easy indexing
    axes = np.array(axes).reshape(-1)

    # Plot each sampled window
    for idx, (_, row) in enumerate(sampled.iterrows()):
        ax = axes[idx]
        vals = [row[col] for col in feature_cols]
        bars = ax.bar(feature_cols, vals, color=[colors[col] for col in feature_cols], edgecolor="black")
        ax.set_title(f"{row['start_ms']:.1f}–{row['end_ms']:.1f} ms", fontsize=10, pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # Style spines
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Colour‐code tick labels
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
        for label in ax.get_xticklabels():
            label.set_color(colors[label.get_text()])

        # Annotate bar values
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    h + 0.005,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7
                )

    # Turn off any unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    # Global styling
    fig.suptitle("Ion‐Percentage Profiles of Sampled Windows", fontsize=16, y=1.02)
    axes[0].set_ylabel("Fraction of Total Counts", fontsize=12)

    # Save figure
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_file
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved grid bar‐plot figure to {out_path}")


if __name__ == "__main__":
    main()


# python vis_fingerprints_barplot.py \
#   --fingerprints_csv fingerprints/raw_data_ion_fingerprints.csv \
#   --sample_n 6 \
#   --seed 42 \
#   --out_dir plots \
#   --out_file fingerprints_bars_pretty.png \
#   --dpi 300
