import os
import pandas as pd

# --- Setup paths ---
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data_files', 'mono_channel_and_peak_period')

CLEANED_PATH = os.path.join(DATA_DIR, 'cleaned_data.csv')
MERGED_PATH = os.path.join(DATA_DIR, 'events20240515_e_merged_0.0003.csv')

# --- Load data ---
df = pd.read_csv(CLEANED_PATH)   # full time‑series (needed only if you want row‑index lookups)
merged_df = pd.read_csv(MERGED_PATH)    # merged peak windows

# --- Total windows ---
total_windows = len(merged_df)

# --- Count overlapping windows in the merged list ---
overlap_count = 0
for i in range(total_windows - 1):
    end_i = merged_df.loc[i,   'Peak_end']
    start_i1 = merged_df.loc[i+1, 'Peak_start']
    if end_i > start_i1:
        overlap_count += 1

# --- Output ---
print(f"Total number of windows: {total_windows}")
print(f"Total number of overlapping windows: {overlap_count}")

# --- Build lookup from timestamp → row index in df ---
time_to_idx = pd.Series(df.index.values, index=df['Time [s]']).to_dict()

# --- Check overlapping windows for at least 2 rows between their starts ---
two_row_ok_count = 0
for i in range(total_windows - 1):
    p1_start = merged_df.loc[i,   'Peak_start']
    p1_end = merged_df.loc[i,   'Peak_end']
    p2_start = merged_df.loc[i+1, 'Peak_start']

    # only consider true overlaps
    if p1_end > p2_start:
        idx1 = time_to_idx.get(p1_start)
        idx2 = time_to_idx.get(p2_start)
        if idx1 is None or idx2 is None:
            print(f"Could not find indices for starts {p1_start} or {p2_start}")
            continue

        gap_rows = idx2 - idx1 - 1  # rows strictly between the two starts
        if gap_rows >= 2:
            two_row_ok_count += 1
        else:
            print(f"Overlap between [{p1_start}, …] and [{p2_start}, …] has only {gap_rows} intermediate row(s)")

print(f"Overlapping windows with at least 2 timestamps between starts: {two_row_ok_count}")
