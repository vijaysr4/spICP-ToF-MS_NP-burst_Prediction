import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# events_span_histogram_with_points.py lives in project_root/data_analysis/
THIS_DIR = os.path.dirname(__file__)
# Ensure a results directory next to this script
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = os.path.join(
    THIS_DIR,
    os.pardir,
    'data_files',
    'mono_channel_and_peak_period'
)

# 1. Load the augmented windows file
in_path = os.path.join(DATA_DIR, 'events20240515_e_with_spans.csv')
w_df = pd.read_csv(in_path)
spans = w_df['span_pre-trim']  # window sizes in seconds

# 2. Compute histogram for time-based spans
bins = 120
counts, edges = np.histogram(spans, bins=bins, range=(spans.min(), spans.max()))
centers = (edges[:-1] + edges[1:]) / 2
width = edges[1] - edges[0]

# 3. Summary statistics
stats = {
    'Min': spans.min(),
    'Max': spans.max(),
    'Mean': spans.mean(),
    'Median': spans.median()
}
colors = {
    'Min': 'tab:blue',
    'Max': 'tab:orange',
    'Mean': 'tab:green',
    'Median': 'tab:red'
}

# Map stats to bins
stat_bin = {label: np.clip(np.digitize(val, edges) - 1, 0, bins - 1)
            for label, val in stats.items()}

# 4. Plot histogram of window sizes (seconds)
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(centers, counts, width=width * 0.9)
for i, bar in enumerate(bars):
    label = next((lbl for lbl, idx in stat_bin.items() if idx == i), None)
    if label:
        bar.set_facecolor(colors[label])
        bar.set_alpha(1.0)
    else:
        bar.set_facecolor('lightgray')
        bar.set_alpha(0.9)

# Styling
for spine in ax.spines.values(): spine.set_visible(False)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
ax.xaxis.set_major_locator(MultipleLocator((spans.max()) / 10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax.set_xlabel('Window Size (seconds)')
ax.set_ylabel('Number of Windows')
ax.set_title('Pre-Trim Window Size Distribution (seconds)')
legend_handles = [Patch(facecolor=colors[l], label=f"{l} = {stats[l]:.6f}s") for l in stats]
ax.legend(handles=legend_handles, frameon=False)
plt.tight_layout()
# Save to results folder
out_path = os.path.join(RESULTS_DIR, 'pretrim_window_size_histogram_seconds.png')
fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved seconds-based histogram to: {out_path}")

# 5. Convert span (s) to number of datapoints
cleaned_path = os.path.join(DATA_DIR, 'cleaned_data.csv')
df = pd.read_csv(cleaned_path)
dt = np.median(np.diff(df['Time [s]']))
spans_pts = np.round(spans / dt).astype(int)

# 6. Histogram for datapoint-based spans
counts_pts, edges_pts = np.histogram(spans_pts, bins=120,
                                     range=(spans_pts.min(), spans_pts.max()))
centers_pts = (edges_pts[:-1] + edges_pts[1:]) / 2
width_pts = edges_pts[1] - edges_pts[0]

# 7. Plot histogram of window sizes (data points)
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(centers_pts, counts_pts, width=width_pts * 0.9)
stat_pts = {label: int(round(val / dt)) for label, val in stats.items()}
stat_bin_pts = {label: np.clip(np.digitize(val, edges_pts) - 1, 0, len(centers_pts)-1)
                for label, val in stat_pts.items()}
for i, bar in enumerate(bars):
    label = next((lbl for lbl, idx in stat_bin_pts.items() if idx == i), None)
    if label:
        bar.set_facecolor(colors[label])
        bar.set_alpha(1.0)
    else:
        bar.set_facecolor('lightgray')
        bar.set_alpha(0.9)

# Styling
for spine in ax.spines.values(): spine.set_visible(False)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
ax.xaxis.set_major_locator(MultipleLocator(max(1, (spans_pts.max()) // 10)))
ax.set_xlabel('Window Size (number of datapoints)')
ax.set_ylabel('Number of Windows')
ax.set_title('Pre-Trim Window Size Distribution (datapoints)')
legend_handles_pts = [Patch(facecolor=colors[l],
                          label=f"{l} ≈ {stat_pts[l]} pts") for l in stats]
ax.legend(handles=legend_handles_pts, frameon=False)
plt.tight_layout()
# Save to results folder
out_path_pts = os.path.join(RESULTS_DIR, 'pretrim_window_size_histogram_datapoints.png')
fig.savefig(out_path_pts, dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved datapoints-based histogram to: {out_path_pts}")
