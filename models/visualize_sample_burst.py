# visualize_single_burst_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# --- User parameters ---
RAW_CSV        = "../data/NPs_BHVO_Oct23_full.csv"
BURSTS_CSV     = "processed_data/detected_bursts.csv"
CLUSTERED_CSV  = "processed_data/burst_clusters.csv"
TIME_COL       = "Time (ms)"
BURST_IDX      = 4        # which burst to visualize
ZOOM_MARGIN_MS = 0.5      # ms padding around the burst
Y_MARGIN       = 1.1      # scale factor above the max to give headroom
RESULTS_DIR    = "results"

# Ensure output directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load data ---
raw_df       = pd.read_csv(RAW_CSV)
bursts_df    = pd.read_csv(BURSTS_CSV)
clustered_df = pd.read_csv(CLUSTERED_CSV)

# Determine ion channels and fingerprint columns
isotope_cols = [c for c in raw_df.columns if c != TIME_COL]
fp_cols      = [c for c in clustered_df.columns if c not in {"start_ms","end_ms","error","cluster"}]

# Extract the burst window
t0 = bursts_df.loc[BURST_IDX, "start_ms"]
t1 = bursts_df.loc[BURST_IDX, "end_ms"]

# Prepare colormap
cmap = get_cmap("tab20")

# Subset for zoom window to compute y-limits
mask = (raw_df[TIME_COL] >= t0 - ZOOM_MARGIN_MS) & (raw_df[TIME_COL] <= t1 + ZOOM_MARGIN_MS)
window_df = raw_df.loc[mask, isotope_cols]
max_peak = window_df.max().max()
ymax     = max_peak * Y_MARGIN

# --- 1) Original data plot ---
fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=150)
for i, col in enumerate(isotope_cols):
    ax1.plot(raw_df[TIME_COL], raw_df[col],
             color=cmap(i), linewidth=1.5, alpha=0.8, label=col)
ax1.set_xlim(t0 - ZOOM_MARGIN_MS, t1 + ZOOM_MARGIN_MS)
ax1.set_ylim(0, ymax)
ax1.set_xlabel("Time (ms)", fontsize=12)
ax1.set_ylabel("Ion Count", fontsize=12)
ax1.set_title(f"Burst {BURST_IDX} – Raw Ion Signals", fontsize=14, pad=12)
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax1.legend(loc="upper right", ncol=4, fontsize=8, frameon=True)
fig1.tight_layout()
fig1.savefig(os.path.join(RESULTS_DIR, f"burst_{BURST_IDX}_original.png"), bbox_inches="tight")
plt.close(fig1)

# --- 2) Highlighted peak region plot ---
fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=150)
for i, col in enumerate(isotope_cols):
    ax2.plot(raw_df[TIME_COL], raw_df[col],
             color=cmap(i), linewidth=1.5, alpha=0.8)
# use a distinct highlight color (e.g. light red)
ax2.axvspan(t0, t1, color="salmon", alpha=0.3, label="Detected Peak")
ax2.set_xlim(t0 - ZOOM_MARGIN_MS, t1 + ZOOM_MARGIN_MS)
ax2.set_ylim(0, ymax)
ax2.set_xlabel("Time (ms)", fontsize=12)
ax2.set_ylabel("Ion Count", fontsize=12)
ax2.set_title(f"Burst {BURST_IDX} – Detected Peak Highlighted", fontsize=14, pad=12)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax2.legend(loc="upper right", fontsize=8, frameon=True)
fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, f"burst_{BURST_IDX}_peak.png"), bbox_inches="tight")
plt.close(fig2)

# --- 3) Cluster fingerprint bar chart ---
row = clustered_df.iloc[BURST_IDX]
fig3, ax3 = plt.subplots(figsize=(10, 4), dpi=150)
values = row[fp_cols]
bar_colors = [cmap(i) for i in range(len(fp_cols))]
ax3.bar(fp_cols, values, color=bar_colors, edgecolor="black")
ax3.set_xlabel("Isotope", fontsize=12)
ax3.set_ylabel("Summed Ion Count", fontsize=12)
ax3.set_title(f"Burst {BURST_IDX} – Elemental Fingerprint (Cluster {int(row['cluster'])})", fontsize=14, pad=12)
ax3.tick_params(axis='x', rotation=45, labelsize=10)
ax3.grid(axis='y', linestyle="--", linewidth=0.5, alpha=0.6)
fig3.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR, f"burst_{BURST_IDX}_cluster.png"), bbox_inches="tight")
plt.close(fig3)
