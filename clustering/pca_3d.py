import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create high-resolution PNG plots for K-Means clustering"
    )
    parser.add_argument(
        "--labels_csv",
        type=Path,
        required=True,
        help="CSV containing ion-channel columns plus 'cluster_label'"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./plots"),
        help="Directory to save PNG figures"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Resolution (dots per inch) for PNG files"
    )
    return parser.parse_args()


def load_data(labels_csv: Path):
    df = pd.read_csv(labels_csv)
    feature_cols = [c for c in df.columns if c not in ('start_ms', 'end_ms', 'cluster_label')]
    X = df[feature_cols]
    labels = df['cluster_label'].astype(str)
    return feature_cols, X, labels


def plot_boxplot(feature_cols, X, labels, out_path: Path, dpi: int):
    # unchanged for brevity
    ...


def plot_pca(X, labels, out_path: Path, dpi: int):
    # unchanged for brevity
    ...


def plot_pca_3d(X, labels, out_path: Path, dpi: int, elev: float = 30, azim: float = 45):
    """
    Create a high-definition 3D PCA scatter of windows colored by cluster,
    saved as a standard 2D PNG. Improves aesthetics with larger markers,
    white edges, annotated explained variance, and cleaner panes.
    """
    # Fit PCA to 3 components and get explained variance
    pca = PCA(n_components=3, random_state=0)
    coords = pca.fit_transform(X.values)
    var_ratio = pca.explained_variance_ratio_

    # Set up a larger, high-DPI figure
    fig = plt.figure(figsize=(12, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    clusters = sorted(labels.unique(), key=lambda x: int(x))
    cmap = plt.get_cmap('tab10')

    for i, cl in enumerate(clusters):
        idx = labels == cl
        ax.scatter(
            coords[idx, 0], coords[idx, 1], coords[idx, 2],
            s=80, alpha=0.8,
            label=f"Cluster {cl}",
            color=cmap(i),
            edgecolors='w', linewidth=0.5
        )

    # Customize view, labels, and title with variance info
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(f"PC 1 ({var_ratio[0]*100:.1f}% var)", fontsize=14)
    ax.set_ylabel(f"PC 2 ({var_ratio[1]*100:.1f}% var)", fontsize=14)
    ax.set_zlabel(f"PC 3 ({var_ratio[2]*100:.1f}% var)", fontsize=14)
    ax.set_title("3D PCA Projection of Windows by Cluster", fontsize=16)

    # Tidy up the panes and grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    # Legend outside plot
    ax.legend(title='Cluster', fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    save_path = out_path / "pca_clusters_3d.png"
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved high-definition 3D PCA scatter to {save_path}")


def main():
    args = parse_args()
    feature_cols, X, labels = load_data(args.labels_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_boxplot(feature_cols, X, labels, args.out_dir, args.dpi)
    plot_pca(X, labels, args.out_dir, args.dpi)
    plot_pca_3d(X, labels, args.out_dir, args.dpi)


if __name__ == '__main__':
    main()


# python pca_3d.py \
#   --labels_csv ./results/kmeans/raw_data_kmeans_labels.csv \
#   --out_dir    ./results/kmeans/plots \
#   --dpi        300