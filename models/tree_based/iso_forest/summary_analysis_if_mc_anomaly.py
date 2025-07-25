#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths (all relative to this script’s location)
# --------------------------------------------------------------------------- #
SCRIPT_DIR  = Path(__file__).resolve().parent
NANO_ROOT   = SCRIPT_DIR.parents[2]

DATA_DIR    = NANO_ROOT / 'data_files' / 'mono_channel_and_peak_period'
RESULTS_DIR = SCRIPT_DIR / 'results'
SUMMARY_CSV = RESULTS_DIR / '97.39_evaluation_summary_mono_iso_forest.csv'

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Context info for titles and filenames
# --------------------------------------------------------------------------- #
model_name  = "Mono‑channel Isolation Forest"
threshold   = 97.39
context_str = f"{model_name} (threshold={threshold}%)"

# --------------------------------------------------------------------------- #
# Load detailed evaluation summary
# --------------------------------------------------------------------------- #
summary = pd.read_csv(SUMMARY_CSV)

# --------------------------------------------------------------------------- #
# Pretty HD plot settings
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.5,
    'grid.color': '#cccccc',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
})

# --------------------------------------------------------------------------- #
# 1. Histogram of signed start-time errors
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1,1,1)
ax.hist(summary['start_err'], bins=40, edgecolor='k')
ax.set_title(f"Signed Start‑Time Errors — {context_str}")
ax.set_xlabel('Predicted start – True start (s)')
ax.set_ylabel('Count')
ax.grid(True)

# annotate summary stats
text = (
    f"Mean error: {summary['start_err'].mean():.4f}s\n"
    f"Median error: {summary['start_err'].median():.4f}s\n"
    f"Std dev: {summary['start_err'].std():.4f}s"
)
ax.text(0.98, 0.95, text, transform=ax.transAxes,
        fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

hist_path = RESULTS_DIR / f"hist_signed_error_mono_iso_thresh_{threshold}.png"
fig.tight_layout()
fig.savefig(hist_path)
plt.close(fig)

# --------------------------------------------------------------------------- #
# 2. Scatter IoU vs. absolute start‑time error
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1,1,1)
ax.scatter(
    summary['start_err_abs'],
    summary['iou'],
    s=30, alpha=0.7, edgecolors='k'
)
ax.set_title(f"IoU vs. Absolute Start‑Time Error — {context_str}")
ax.set_xlabel('Absolute start error (s)')
ax.set_ylabel('IoU')
ax.grid(True)

# force axes to start at zero to use full plot area
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

# annotate correlation
corr = summary[['start_err_abs','iou']].corr().iloc[0,1]
ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
        fontsize=10, va='top', ha='left',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

scatter_path = RESULTS_DIR / f"scatter_iou_vs_abs_error_mono_iso_thresh_{threshold}.png"
fig.tight_layout()
fig.savefig(scatter_path)
plt.close(fig)

print("Plots saved to:")
print(f" • {hist_path}")
print(f" • {scatter_path}")
