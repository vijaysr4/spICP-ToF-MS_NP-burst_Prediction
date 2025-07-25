
#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from tqdm import tqdm

# Utility to merge detected peaks that are closer than a threshold (in seconds)
def merge_close_peaks(times, min_gap):
    if len(times) == 0:
        return times
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] <= min_gap:
            # skip or average
            merged[-1] = (merged[-1] + t) / 2
        else:
            merged.append(t)
    return np.array(merged)

def load_data():
    # script dir
    script_dir   = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    DATA_DIR     = os.path.join(project_root, 'data_files', 'mono_channel_and_peak_period')
    OUTDIR       = os.path.join(script_dir, 'results')
    os.makedirs(OUTDIR, exist_ok=True)

    df = pd.read_csv(
        os.path.join(DATA_DIR, 'cleaned_data.csv'),
        names=['Time_s', 'Intensity'],
        header=0
    )
    w_df = pd.read_csv(
        os.path.join(DATA_DIR, 'events20240515_e.csv'),
        names=['idx', 'start', 'end'],
        header=0
    )
    return df, w_df, OUTDIR

def evaluate_peaks(detected_times, intervals):
    starts = intervals['start'].values
    ends   = intervals['end'].values
    tp = sum(any((t >= s) and (t <= e) for s, e in zip(starts, ends)) for t in detected_times)
    fp = len(detected_times) - tp
    fn = sum(not any((detected_times >= s) & (detected_times <= e)) for s, e in zip(starts, ends))
    return tp, fp, fn

def main():
    df, w_df, OUTDIR = load_data()
    t = df['Time_s'].values
    y = df['Intensity'].values

    # Sweep window lengths from 100 to 1000 samples, stepping by 50 samples
    windows = list(range(100, 1001, 50))  # e.g., 100,150,...,1000
    total_counts, f1_scores = [], []

    # choose a minimum gap to merge peaks (e.g., average raw interval span)
    avg_span = np.mean(w_df['end'] - w_df['start'])

    for w in tqdm(windows, desc="Eval window sizes", unit="window"):
        if w > len(y):
            total_counts.append(np.nan)
            f1_scores.append(np.nan)
            continue

        # smoothing
        smooth = savgol_filter(y, window_length=w, polyorder=2)
        peaks, _ = find_peaks(smooth)
        detected = t[peaks]

        # merge overlapping or very close detections
        detected = merge_close_peaks(detected, min_gap=avg_span)
        total_counts.append(len(detected))

        tp, fp, fn = evaluate_peaks(detected, w_df)
        prec = tp/(tp+fp) if tp+fp else 0
        rec  = tp/(tp+fn) if tp+fn else 0
        f1_scores.append(2*prec*rec/(prec+rec) if prec+rec else 0)

    # Identify best window
    best_idx = int(np.nanargmax(f1_scores))
    best_w, best_f1 = windows[best_idx], f1_scores[best_idx]

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(windows, total_counts, marker='o', label='Peak Count')
    ax1.set_xlabel('Window Length (samples)', fontsize=12)
    ax1.set_ylabel('Detected Peaks', fontsize=12)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(windows, f1_scores, marker='s', linestyle='--', label='F₁ Score')
    ax2.set_ylabel('F₁ Score', fontsize=12)
    ax2.set_ylim(0, 1.05)

    ax2.axvline(best_w, linestyle=':', color='grey')
    ax2.annotate(f'Best F₁={best_f1:.2f}\nat w={best_w}',
                 xy=(best_w, best_f1), xytext=(best_w+50, best_f1-0.1),
                 arrowprops=dict(arrowstyle='->'))

    # Legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1+handles2, labels1+labels2, loc='upper center')

    plt.title('Window Size (100–1000) vs Peak Count & F₁ Score', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(OUTDIR, 'window_selection_large_range.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()

