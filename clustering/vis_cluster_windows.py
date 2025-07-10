#!/usr/bin/env python
"""
vis_cluster_windows.py

For each cluster label, randomly sample 3 windows and plot their ion-percentage profiles
as bar charts in a grid. Saves a single high-resolution PNG.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Plot 3 random windows per cluster, showing ion composition"
    )
    parser.add_argument(
        "--labels_csv", type=Path, required=True,
        help="CSV with ion channels + 'cluster_label'"
    )
    parser.add_argument(
        "--sample_n", type=int, default=3,
        help="Number of windows to sample per cluster (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("./"),
        help="Directory to save the output PNG"
    )
    parser.add_argument(
        "--out_file", type=str, default="cluster_windows.png",
        help="Filename for the output PNG"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution (dpi) for saved PNG"
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.labels_csv)
    if "cluster_label" not in df.columns:
        raise ValueError("labels_csv must contain a 'cluster_label' column")
    feature_cols = [c for c in df.columns if c not in ("start_ms", "end_ms", "cluster_label")]

    # Prepare sampling
    rng = np.random.RandomState(args.seed)
    clusters = sorted(df["cluster_label"].unique())
    n_clusters = len(clusters)
    n_cols = args.sample_n

    # Create subplots grid
    fig, axes = plt.subplots(
        n_clusters, n_cols,
        figsize=(n_cols * 4, n_clusters * 4),
        sharey=True
    )
    if n_clusters == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Plot each sampled window
    for i, cl in enumerate(clusters):
        subdf = df[df.cluster_label == cl]
        if len(subdf) < args.sample_n:
            raise ValueError(f"Cluster {cl} has only {len(subdf)} windows (<{args.sample_n})")
        sampled = subdf.sample(n=args.sample_n, random_state=args.seed)

        for j, (_, row) in enumerate(sampled.iterrows()):
            ax = axes[i, j]
            values = row[feature_cols].values.astype(float)
            bars = ax.bar(feature_cols, values)
            ax.set_xticks(range(len(feature_cols)))
            ax.set_xticklabels(feature_cols, rotation=45, ha="right")
            ax.set_title(f"Cluster {cl} – idx {row.name}", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            # Annotate bar values
            for bar in bars:
                h = bar.get_height()
                if h > 0.01:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.005,
                        f"{h:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7
                    )

    # Global title and layout
    fig.suptitle("Sample Ion-Percentage Windows per Cluster", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_file
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved cluster-window visualization to {out_path}")

if __name__ == "__main__":
    main()

# python vis_cluster_windows.py \
#   --labels_csv ./results/kmeans/raw_data_kmeans_labels.csv \
#   --sample_n 3 \
#   --seed 42 \
#   --out_dir ./results/kmeans/plots \
#   --out_file sample_windows_per_cluster.png \
#   --dpi 300
