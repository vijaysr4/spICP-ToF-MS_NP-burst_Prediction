#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm

# Use seaborn style for prettier plots
sns.set(style="whitegrid", context="talk")

# --------------------------------------------------------------------------- #
# Configuration (paths relative to this script)
# --------------------------------------------------------------------------- #
SCRIPT_DIR  = Path(__file__).resolve().parent
NANO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = NANO_ROOT / 'data_files' / 'mono_channel_and_peak_period'
CLEANED_CSV = DATA_DIR / 'cleaned_data.csv'
RESULTS_DIR = SCRIPT_DIR / 'results'
SUMMARY_CSV = RESULTS_DIR / '97.39_evaluation_summary_mono_iso_forest.csv'
PLOTS_DIR   = RESULTS_DIR / 'analysis_plots'
PRED_CSV    = RESULTS_DIR / 'peak_windows_iso_forest_mono_channel_anomaly_97.39.csv'
GT_CSV      = DATA_DIR / 'events20240515_e_merged_0.0003.csv'

# --------------------------------------------------------------------------- #
# Main visualization routine
# --------------------------------------------------------------------------- #
def main() -> None:
    # Ensure output directory exists
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load summary and raw time-series
    summary = pd.read_csv(SUMMARY_CSV)
    ts_df   = pd.read_csv(CLEANED_CSV)

    # Identify false positives: predicted windows without a GT peak
    fp = summary[summary['has_gt_peak'] == 0]
    if fp.empty:
        print("No false positives to visualize.")
        return

    # Randomly sample up to 6 false positives
    sample = fp.sample(n=min(6, len(fp)), random_state=42)

    # Load all GT and pred windows once
    raw_gt = pd.read_csv(GT_CSV)
    raw_gt = raw_gt.rename(columns={'Peak_end':'orig_end', 'Trimmed_Peak_end':'Peak_end'})
    gt_df  = raw_gt[['Peak_start','Peak_end']]
    pred_all = pd.read_csv(PRED_CSV)

    # Plot each sample
    for idx, row in tqdm(sample.iterrows(), total=len(sample), desc='Saving FP plots'):
        p_start = row['pred_window_start']
        p_end   = row['pred_window_end']

        # Define ±0.1 second around predicted window
        t0, t1 = p_start - 0.1, p_end + 0.1
        segment = ts_df[(ts_df['Time [s]'] >= t0) & (ts_df['Time [s]'] <= t1)]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(segment['Time [s]'], segment['intensity'], linewidth=2, label='Intensity')

        # Highlight primary predicted window
        ax.axvspan(p_start, p_end, color='orange', alpha=0.3, label='Primary Pred')

        # Highlight overlapping GT windows
        for _, g in gt_df.iterrows():
            if not (g.Peak_end < t0 or g.Peak_start > t1):
                ax.axvspan(g.Peak_start, g.Peak_end, color='blue', alpha=0.2, label='GT Window')

        # Highlight other predicted windows
        for _, p2 in pred_all.iterrows():
            # skip outside and primary
            if p2['Peak_end'] < t0 or p2['Peak_start'] > t1:
                continue
            if p2['Peak_start'] == p_start and p2['Peak_end'] == p_end:
                continue
            ax.axvspan(p2['Peak_start'], p2['Peak_end'], color='green', alpha=0.2, label='Other Pred')

        # Avoid duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        # Formatting
        ax.set_title(f"False Positive #{idx} — Zoomed Pred Window ±0.1s")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Intensity')
        ax.set_xlim(t0, t1)
        # Provide more ticks for visualization
        ax.xaxis.set_major_locator(ticker.MaxNLocator(10))

        # Save figure with descriptive naming
        filename = f"97.39_win_8s2_fp_mono_iso_forest_peak{idx}.png"
        fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(sample)} false-positive context plots to {PLOTS_DIR}")

if __name__ == '__main__':
    main()
