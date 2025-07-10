import numpy as np
from typing import List, Tuple, Optional

from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.metrics import dtw
from sklearn.cluster import AgglomerativeClustering
import hdbscan
from tqdm import tqdm


def resample_series(
    series_list: List[np.ndarray],
    sz: int = 100
) -> np.ndarray:
    """
    Resample a list of variable-length 1D time series to a common length.

    Shows a progress bar.
    """
    ts_data = to_time_series_dataset(series_list)
    ts_resampled = TimeSeriesResampler(sz=sz).fit_transform(ts_data)
    return ts_resampled


def cluster_kshape(
    ts_data: np.ndarray,
    n_clusters: int = 5,
    n_init: int = 10,
    random_state: int = 0,
    verbose: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply k-Shape clustering to univariate time series data.

    Parameters:
        verbose: Level of verbosity (0: silent, >0: progress)
    """
    model = KShape(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
        verbose=verbose
    )
    labels = model.fit_predict(ts_data)
    centroids = np.squeeze(model.cluster_centers_, axis=2)
    return labels, centroids


def cluster_time_series_kmeans(
    ts_data: np.ndarray,
    n_clusters: int = 5,
    metric: str = "euclidean",
    gamma: float = 0.5,
    max_iter: int = 10,
    random_state: int = 0,
    verbose: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply TimeSeriesKMeans clustering (Euclidean or soft-DTW) to time series.

    Internal iterations are logged by tslearn when verbose>0.
    """
    if metric not in ("euclidean", "softdtw"):
        raise ValueError("Metric must be 'euclidean' or 'softdtw'.")

    params = {
        "n_clusters": n_clusters,
        "metric": metric,
        "max_iter": max_iter,
        "random_state": random_state,
        "verbose": verbose
    }
    if metric == "softdtw":
        params["metric_params"] = {"gamma": gamma}

    model = TimeSeriesKMeans(**params)
    labels = model.fit_predict(ts_data)
    centroids = np.squeeze(model.cluster_centers_, axis=2)
    return labels, centroids


def compute_dtw_matrix(
    flattened: np.ndarray
) -> np.ndarray:
    """
    Compute full pairwise DTW distance matrix with tqdm progress bar.
    flattened: (n_series, sz)
    """
    n_series = flattened.shape[0]
    dist_mat = np.zeros((n_series, n_series))
    for i in tqdm(range(n_series), desc="DTW distance rows"):
        for j in range(i + 1, n_series):
            d = dtw(flattened[i], flattened[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat


def cluster_hierarchical_dtw(
    ts_data: np.ndarray,
    n_clusters: int = 5,
    linkage: str = "average",
    show_progress: bool = True
) -> np.ndarray:
    """
    Perform agglomerative clustering on time series using DTW distance.

    Shows a progress bar while building the distance matrix if show_progress=True.
    """
    n_series, sz, _ = ts_data.shape
    flattened = ts_data.reshape(n_series, sz)
    if show_progress:
        distance_matrix = compute_dtw_matrix(flattened)
    else:
        # no progress bar
        distance_matrix = np.zeros((n_series, n_series))
        for i in range(n_series):
            for j in range(i + 1, n_series):
                d = dtw(flattened[i], flattened[j])
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        linkage=linkage
    )
    labels = model.fit_predict(distance_matrix)
    return labels


def cluster_hdbscan_dtw(
    ts_data: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Density-based clustering (HDBSCAN) on DTW distances.

    Shows a progress bar for distance computation if show_progress=True.
    """
    n_series, sz, _ = ts_data.shape
    flattened = ts_data.reshape(n_series, sz)
    if show_progress:
        distance_matrix = compute_dtw_matrix(flattened)
    else:
        distance_matrix = np.zeros((n_series, n_series))
        for i in range(n_series):
            for j in range(i + 1, n_series):
                d = dtw(flattened[i], flattened[j])
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(distance_matrix)
    return labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    series = [
        np.sin(np.linspace(0, 2 * np.pi, np.random.randint(80, 120)))
        for _ in range(50)
    ]
    ts = resample_series(series, sz=100)
    labels, centroids = cluster_kshape(ts, n_clusters=3)

    for centroid in centroids:
        plt.plot(centroid)
    plt.title("k-Shape Centroids")
    plt.xlabel("Resampled Time Index")
    plt.ylabel("Amplitude")
    plt.show()
