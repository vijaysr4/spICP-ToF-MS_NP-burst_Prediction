import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from cluster_models import (
    resample_series,
    cluster_kshape,
    cluster_time_series_kmeans,
    cluster_hierarchical_dtw,
    cluster_hdbscan_dtw,
)


def load_windows(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'start_ms' not in df.columns or 'end_ms' not in df.columns:
        raise ValueError("Input CSV must contain 'start_ms' and 'end_ms' columns.")
    return df


def prepare_series(df: pd.DataFrame) -> np.ndarray:
    feature_cols = [c for c in df.columns if c not in ('start_ms', 'end_ms')]
    data = df[feature_cols].values.astype(float)
    return data.reshape((data.shape[0], data.shape[1], 1))


def save_results(df: pd.DataFrame, labels: np.ndarray, centroids: np.ndarray,
                 out_dir: str, labels_file: str, cents_file: str):
    """Save clustering labels and centroids with custom filenames."""
    os.makedirs(out_dir, exist_ok=True)

    # Labels
    df_out = df.copy()
    df_out['cluster_label'] = labels
    labels_path = os.path.join(out_dir, labels_file)
    df_out.to_csv(labels_path, index=False)
    print(f"Saved labels to {labels_path}")

    # Centroids (if given)
    if centroids is not None and cents_file:
        cents_path = os.path.join(out_dir, cents_file)
        np.save(cents_path, centroids)
        print(f"Saved centroids to {cents_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster ion-percentage windows using time-series clustering models"
    )
    parser.add_argument("--input", required=True,
                        help="Path to CSV file with windows data")
    parser.add_argument("--model", choices=["kshape","kmeans","hierarchical","hdbscan"],
                        default="kshape", help="Clustering model to use")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters (ignored for HDBSCAN)")
    parser.add_argument("--min_cluster_size", type=int, default=10,
                        help="Min cluster size for HDBSCAN")
    parser.add_argument("--resample_sz", type=int, default=None,
                        help="Resample series to this length if set")
    parser.add_argument("--output_dir", default="./cluster_results",
                        help="Directory to save clustering outputs")
    parser.add_argument("--labels_file", default=None,
                        help="Filename for saving labels CSV (defaults to <model>_labels.csv)")
    parser.add_argument("--centroids_file", default=None,
                        help="Filename for saving centroids (.npy). Omit to skip centroids")
    args = parser.parse_args()

    # Derive default filenames if not provided
    default_labels = f"{args.model}_labels.csv"
    default_cents = f"{args.model}_centroids.npy" if args.model in ("kshape","kmeans") else None
    labels_file = args.labels_file or default_labels
    cents_file = args.centroids_file or default_cents

    # Load & prepare
    print("Loading data...")
    df = load_windows(args.input)
    ts_data = prepare_series(df)
    print(f"Data contains {ts_data.shape[0]} series of length {ts_data.shape[1]}")

    # Optional resampling with progress bar
    if args.resample_sz is not None:
        print(f"Resampling all series to length {args.resample_sz}...")
        series_list = []
        for i in tqdm(range(ts_data.shape[0]), desc="Resampling series"):
            series_list.append(ts_data[i, :, 0])
        ts_data = resample_series(series_list, sz=args.resample_sz)
        print("Resampling complete.")

    # Cluster
    print(f"Running {args.model} clustering...")
    if args.model == "kshape":
        labels, centroids = cluster_kshape(
            ts_data, n_clusters=args.n_clusters
        )
    elif args.model == "kmeans":
        labels, centroids = cluster_time_series_kmeans(
            ts_data, n_clusters=args.n_clusters
        )
    elif args.model == "hierarchical":
        labels = cluster_hierarchical_dtw(
            ts_data, n_clusters=args.n_clusters
        )
        centroids = None
    else:  # hdbscan
        labels = cluster_hdbscan_dtw(
            ts_data, min_cluster_size=args.min_cluster_size
        )
        centroids = None
    print("Clustering complete.")

    # Save outputs
    save_results(df, labels, centroids, args.output_dir, labels_file, cents_file)


if __name__ == "__main__":
    main()



# # 1) K-Shape clustering (5 clusters)
# python clustering_similar_windows.py \
#   --input fingerprints/raw_data_ion_fingerprints.csv \
#   --model kshape \
#   --n_clusters 5 \
#   --output_dir ./results/kshape \
#   --labels_file raw_data_kshape_labels.csv \
#   --centroids_file raw_data_kshape_centroids.npy
#
# # 2) TimeSeriesKMeans (Euclidean, 5 clusters)
# python clustering_similar_windows.py \
#   --input fingerprints/raw_data_ion_fingerprints.csv \
#   --model kmeans \
#   --n_clusters 5 \
#   --output_dir ./results/kmeans \
#   --labels_file raw_data_kmeans_labels.csv \
#   --centroids_file raw_data_kmeans_centroids.npy
#
# # 3) Hierarchical clustering (DTW + average linkage, 5 clusters)
# python clustering_similar_windows.py \
#   --input fingerprints/raw_data_ion_fingerprints.csv \
#   --model hierarchical \
#   --n_clusters 5 \
#   --output_dir ./results/hierarchical \
#   --labels_file raw_data_hierarchical_labels.csv
#
# # 4) HDBSCAN (DTW, min cluster size 5)
# python clustering_similar_windows.py \
#   --input fingerprints/raw_data_ion_fingerprints.csv \
#   --model hdbscan \
#   --min_cluster_size 5 \
#   --output_dir ./results/hdbscan \
#   --labels_file raw_data_hdbscan_labels.csv
