import os
import pandas as pd

# --- Setup paths ---
THIS_DIR     = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data_files', 'mono_channel_and_peak_period')

CLEANED_PATH       = os.path.join(DATA_DIR, 'cleaned_data.csv')
EVENTS_PATH        = os.path.join(DATA_DIR, 'events20240515_e.csv')
MERGED_EVENTS_PATH = os.path.join(DATA_DIR, 'events20240515_e_merged_0.0003.csv')

# --- Load data ---
df       = pd.read_csv(CLEANED_PATH)    # full time‑series
peaks_df = pd.read_csv(EVENTS_PATH)     # original peak windows

# --- Build lookup from timestamp → row index in df ---
time_to_idx = pd.Series(df.index.values, index=df['Time [s]']).to_dict()

# --- Sort peaks by start time ---
peaks_df = peaks_df.sort_values('Peak_start').reset_index(drop=True)

# --- Merge only when gap_rows < 4 ---
merged = []
current_start = peaks_df.loc[0, 'Peak_start']
current_end   = peaks_df.loc[0, 'Peak_end']
current_idx1  = time_to_idx.get(current_start)

for i in range(1, len(peaks_df)):
    p1_start, p1_end = current_start, current_end
    p2_start = peaks_df.loc[i, 'Peak_start']
    p2_end   = peaks_df.loc[i, 'Peak_end']
    idx2      = time_to_idx.get(p2_start)

    # Compute gap in rows (if both indices exist)
    if current_idx1 is not None and idx2 is not None:
        gap_rows = idx2 - current_idx1 - 1
    else:
        gap_rows = float('inf')  # treat as non‑mergeable if lookup fails

    # Merge if gap_rows < 4
    if gap_rows < 4:
        print(f"Merging windows: [{p1_start}, {p1_end}] + [{p2_start}, {p2_end}]"
              f" → [{p1_start}, {max(p1_end, p2_end)}] (gap_rows={gap_rows})")
        current_end = max(current_end, p2_end)
        # current_start and current_idx1 remain
    else:
        # flush current span
        merged.append({'Peak_start': current_start,
                       'Peak_end':   current_end})
        # start new span
        current_start, current_end, current_idx1 = p2_start, p2_end, idx2

# Append the final span
merged.append({'Peak_start': current_start, 'Peak_end': current_end})

# --- Save merged windows to CSV ---
merged_df = pd.DataFrame(merged)
merged_df.to_csv(MERGED_EVENTS_PATH, index=False)
print(f"\nMerged events written to: {MERGED_EVENTS_PATH}")

# --- Print counts ---
print(f"Original number of windows: {len(peaks_df)}")
print(f"Number of windows after merging: {len(merged_df)}")


# Merging windows: [17.8303, 17.8364] + [17.8305, 17.8493] → [17.8303, 17.8493] (gap_rows=1)
# Merging windows: [21.9652, 21.9785] + [21.9653, 21.972] → [21.9652, 21.9785] (gap_rows=0)
# Merging windows: [23.1617, 23.1754] + [23.1619, 23.1657] → [23.1617, 23.1754] (gap_rows=1)
# Merging windows: [25.6541, 25.6783] + [25.6543, 25.6585] → [25.6541, 25.6783] (gap_rows=1)
# Merging windows: [30.6318, 30.6435] + [30.632, 30.6475] → [30.6318, 30.6475] (gap_rows=1)
# Merging windows: [38.6509, 38.6702] + [38.6511, 38.6657] → [38.6509, 38.6702] (gap_rows=1)
#

# Original number of windows: 923
# Number of windows after merging: 917
