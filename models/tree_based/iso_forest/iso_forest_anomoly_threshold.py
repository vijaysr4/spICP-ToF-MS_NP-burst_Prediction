import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

# Parameters
WINDOW_SIZE    = 8        # samples per window
STRIDE         = 2        # step size between windows
QUANTILE       = 0.9739     # quantile threshold for anomaly scores
RANDOM_STATE   = 42       # reproducibility

# Load data
def load_data(path):
    return pd.read_csv(path, names=['Time_s', 'Intensity'], header=0)

# Build overlapping windows
def build_windows(series, window_size, stride):
    X, starts = [], []
    n = len(series)
    for start in tqdm(range(0, n - window_size + 1, stride), desc='Building windows'):
        X.append(series.iloc[start:start+window_size].values)
        starts.append(start)
    return np.vstack(X), np.array(starts)

# Extract and merge peak windows
def extract_peak_windows(starts, flags, time_array, window_size):
    peak_windows = []
    overlaps = 0
    i = 0
    N = len(starts)
    while i < N:
        if not flags[i]:
            i += 1
            continue
        # count overlapping within this group
        j = i + 1
        while j < N and flags[j]:
            overlaps += 1
            j += 1
        # determine group interval
        group_start = starts[i]
        group_end = starts[j-1] + window_size - 1
        group_end = min(group_end, len(time_array)-1)
        t0 = time_array[group_start]
        t1 = time_array[group_end]
        peak_windows.append([t0, t1])
        i = j
    return peak_windows, overlaps

# Plot helper
def save_score_plot(t_mid, scores, path, log_scale=False):
    # Title includes algorithm and channel
    title_base = 'Mono-Channel IsolationForest Anomaly Scores'
    title = title_base + (' (Log Y-scale)' if log_scale else '')

    plt.figure(figsize=(16, 4), dpi=150)
    ax = plt.gca()
    ax.scatter(t_mid, scores, s=20, alpha=0.6, edgecolors='none')
    ax.set_xlim(t_mid.min(), t_mid.max())
    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Anomaly score', fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, prune='both'))
    plt.xticks(rotation=45, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# Main logic: compute scores, detect, merge peaks, save plots and CSV
def main():
    # Determine paths
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(
        script_dir, '..', '..', '..', 'data_files',
        'mono_channel_and_peak_period'
    ))
    data_file = os.path.join(data_dir, 'cleaned_data.csv')
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load and window data
    df = load_data(data_file)
    X, starts = build_windows(df['Intensity'], WINDOW_SIZE, STRIDE)

    # Fit model and compute continuous scores
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination='auto',
        random_state=RANDOM_STATE
    )
    model.fit(X)
    scores = -model.decision_function(X)

    # Determine threshold by quantile
    thresh = np.quantile(scores, QUANTILE)
    flags = scores >= thresh
    total_windows = len(flags)
    total_peak_windows = flags.sum()
    print(f"Anomaly threshold (quantile={QUANTILE*100:.2f}%): {thresh:.4f}")
    print(f"Total windows: {total_windows}")
    print(f"Total peak windows detected: {total_peak_windows}")

    # Extract and merge peaks, counting overlaps
    peak_windows, overlaps = extract_peak_windows(starts, flags, df['Time_s'].values, WINDOW_SIZE)
    print(f"Total overlapping windows within peaks: {overlaps}")
    print(f"Total merged peak intervals: {len(peak_windows)}")

    # Save peak intervals to CSV
    percent = QUANTILE * 100
    csv_name = f"peak_windows_iso_forest_mono_channel_anomaly_{percent:.2f}.csv"
    csv_path = os.path.join(results_dir, csv_name)
    df_out = pd.DataFrame(peak_windows, columns=['Peak_start', 'Peak_end'])
    df_out.to_csv(csv_path, index=False)
    print(f"Peak intervals saved to {csv_path}")

    # Compute midpoints for plotting
    times = df['Time_s'].values
    t_mid = (times[starts] + times[np.minimum(starts + WINDOW_SIZE - 1, len(times)-1)]) / 2

    # Save score distribution plots
    linear_name = 'Anomaly_score_iso_forest_mono_channel_scores.png'
    linear_path = os.path.join(results_dir, linear_name)
    save_score_plot(t_mid, scores, linear_path, log_scale=False)
    print(f"Linear anomaly score plot saved to {linear_path}")

    log_name = 'Anomaly_score_iso_forest_mono_channel_scores_log.png'
    log_path = os.path.join(results_dir, log_name)
    save_score_plot(t_mid, scores, log_path, log_scale=True)
    print(f"Log-scale anomaly score plot saved to {log_path}")

if __name__ == '__main__':
    main()