import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
        default=300,
        help="Resolution (dots per inch) for PNG files"
    )
    return parser.parse_args()


def load_data(labels_csv: Path):
    df = pd.read_csv(labels_csv)
    # Identify ion-channel columns (exclude time & label)
    feature_cols = [c for c in df.columns if c not in ('start_ms', 'end_ms', 'cluster_label')]
    X = df[feature_cols]
    labels = df['cluster_label'].astype(str)
    return feature_cols, X, labels


def plot_boxplot(feature_cols, X, labels, out_path: Path, dpi: int):
    """
    Create a box plot for each ion channel showing distribution of values per cluster.
    """
    clusters = sorted(labels.unique(), key=lambda x: int(x))
    n_channels = len(feature_cols)
    n_clusters = len(clusters)
    width = 0.8 / n_clusters

    plt.figure(figsize=(14, 6))
    # Define a color palette
    cmap = plt.get_cmap('tab10')

    for i, cl in enumerate(clusters):
        # Positions for this cluster's boxes
        positions = np.arange(n_channels) - 0.4 + i * width + width / 2
        data = [X[channel][labels == cl] for channel in feature_cols]
        bp = plt.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            manage_ticks=False
        )
        color = cmap(i)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for element in ['whiskers', 'caps', 'medians', 'fliers']:
            for item in bp[element]:
                item.set_color(color)
                item.set_linewidth(1)

    plt.xticks(np.arange(n_channels), feature_cols, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Ion Channel", fontsize=12)
    plt.ylabel("Percentage", fontsize=12)
    plt.title("Ion-Percentage Distribution per Channel by Cluster", fontsize=14)
    # Legend proxies
    handles = [plt.Line2D([0], [0], color=plt.get_cmap('tab10')(i), lw=4, alpha=0.6)
               for i in range(n_clusters)]
    plt.legend(handles, [f"Cluster {cl}" for cl in clusters], title='Cluster', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = out_path / "boxplot_channels.png"
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved box-plot to {save_path}")


def plot_pca(X, labels, out_path: Path, dpi: int):
    """
    Create a PCA scatter of windows colored by cluster.
    """
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X.values)

    plt.figure(figsize=(8, 6))
    clusters = sorted(labels.unique(), key=lambda x: int(x))
    cmap = plt.get_cmap('tab10')

    for i, cl in enumerate(clusters):
        idx = labels == cl
        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=30,
            alpha=0.7,
            label=f"Cluster {cl}",
            color=cmap(i)
        )

    plt.grid(alpha=0.3)
    plt.xlabel("PC 1", fontsize=12)
    plt.ylabel("PC 2", fontsize=12)
    plt.title("PCA Projection of Windows by Cluster", fontsize=14)
    plt.legend(title='Cluster', fontsize=10)
    plt.tight_layout()

    save_path = out_path / "pca_clusters.png"
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA scatter to {save_path}")


def main():
    args = parse_args()
    feature_cols, X, labels = load_data(args.labels_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_boxplot(feature_cols, X, labels, args.out_dir, args.dpi)
    plot_pca(X, labels, args.out_dir, args.dpi)


if __name__ == '__main__':
    main()



# python visualization.py \
#   --labels_csv ./results/kmeans/raw_data_kmeans_labels.csv \
#   --out_dir ./results/kmeans/plots \
#   --dpi 300

