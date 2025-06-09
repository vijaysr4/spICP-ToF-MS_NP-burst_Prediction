import numpy as np
import pandas as pd
import hdbscan
from tqdm import tqdm
from typing import List

def index_of_time(times: np.ndarray, t: float) -> int:
    """
    Find the index in 'times' whose value is closest to t.
    """
    return int(np.abs(times - t).argmin())

def extract_fingerprints(
    raw_df: pd.DataFrame,
    bursts_df: pd.DataFrame,
    time_col: str,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    For each detected burst, sum raw counts over its time span.
    Shows a progress bar over the bursts.
    """
    times = raw_df[time_col].values
    fps = []

    # wrap the iteration in tqdm so we can see progress
    for _, burst in tqdm(bursts_df.iterrows(),
                         total=len(bursts_df),
                         desc="Extracting fingerprints"):
        start_t, end_t = burst["start_ms"], burst["end_ms"]
        i0 = index_of_time(times, start_t)
        i1 = index_of_time(times, end_t)
        sums = raw_df.iloc[i0:i1+1][feature_cols].sum().to_dict()
        sums.update({
            "start_ms": start_t,
            "end_ms":   end_t,
            "error":    burst["error"]
        })
        fps.append(sums)

    return pd.DataFrame(fps)

def cluster_peaks(
    fingerprints: pd.DataFrame,
    feature_cols: List[str],
    min_cluster_size: int = 5
) -> pd.DataFrame:
    """
    Run HDBSCAN on the fingerprint vectors and append the cluster labels.
    Displays progress during clustering.
    """
    # HDBSCAN itself doesn't expose an inner loop, but we'll at least
    # let the user know this step is happening.
    print("Clustering peaks with HDBSCAN…")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                gen_min_span_tree=True)
    labels = clusterer.fit_predict(fingerprints[feature_cols].values)
    fingerprints["cluster"] = labels
    print("  → Found %d clusters (excluding outliers)" %
          (len(set(labels)) - (1 if -1 in labels else 0)))
    return fingerprints

def main(
    raw_csv: str,
    bursts_csv: str,
    output_csv: str = "models/processed_data/burst_clusters.csv",
    time_col: str = "Time (ms)"
):
    # Load data
    raw_df    = pd.read_csv(raw_csv)
    bursts_df = pd.read_csv(bursts_csv)
    feature_cols = [c for c in raw_df.columns if c != time_col]

    # Extract elemental fingerprints
    fps = extract_fingerprints(raw_df, bursts_df, time_col, feature_cols)

    # Cluster the peaks
    fps_clustered = cluster_peaks(fps, feature_cols, min_cluster_size=5)

    # Save results
    fps_clustered.to_csv(output_csv, index=False)
    print(f"Clustered bursts saved to {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python cluster_bursts.py RAW_DATA.csv DETECTED_BURSTS.csv [OUTPUT.csv]")
        sys.exit(1)
    raw_csv    = sys.argv[1]
    bursts_csv = sys.argv[2]
    out_csv    = sys.argv[3] if len(sys.argv) > 3 else "burst_clusters.csv"
    main(raw_csv, bursts_csv, out_csv)


# python -m models.cluster_bursts data/NPs_BHVO_Oct23_full.csv models/processed_data/detected_bursts.csv models/processed_data/burst_clusters.csv