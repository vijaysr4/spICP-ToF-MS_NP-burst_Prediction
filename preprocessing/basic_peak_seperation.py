import os
import pandas as pd

# --- Setup paths ---
THIS_DIR     = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data_files', 'mono_channel_and_peak_period')

CLEANED_PATH = os.path.join(DATA_DIR, 'cleaned_data.csv')
MERGED_PATH  = os.path.join(DATA_DIR, 'events20240515_e_merged_0.0003.csv')

# --- Load data ---
df        = pd.read_csv(CLEANED_PATH)   # full time‑series (needed only if you want row‑index lookups)
merged_df = pd.read_csv(MERGED_PATH)    # merged peak windows

# --- Total windows ---
total_windows = len(merged_df)

# --- Count overlapping windows in the merged list ---
overlap_count = 0
for i in range(total_windows - 1):
    end_i    = merged_df.loc[i,   'Peak_end']
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
    p1_end   = merged_df.loc[i,   'Peak_end']
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

# --- Add Trimmed_Peak_end column and apply trimming logic ---
merged_df['Trimmed_Peak_end'] = merged_df['Peak_end']  # default copy

for i in range(total_windows - 1):
    p1_start = merged_df.loc[i,   'Peak_start']
    p1_end   = merged_df.loc[i,   'Peak_end']
    p2_start = merged_df.loc[i+1, 'Peak_start']

    # only trim true overlaps
    if p1_end <= p2_start:
        continue

    idx2 = time_to_idx.get(p2_start)
    if idx2 is None:
        print(f"[SKIP] No index for P2_start={p2_start}, window #{i} ({p1_start}–{p1_end})")
        continue

    # target two full timestamps between trimmed_end and p2_start => trimmed_idx = idx2 - 3
    trimmed_idx = idx2 - 3

    # guard: ensure trimmed_idx > idx1
    idx1 = time_to_idx.get(p1_start)
    if idx1 is None:
        print(f"[WARN] No index for P1_start={p1_start}, cannot apply guard logic.")
    else:
        if trimmed_idx <= idx1:
            print(f"[ADJUST] Window #{i} ({p1_start}–{p1_end}) → not enough room; "
                  f"idx1={idx1}, trimmed_idx={trimmed_idx}. "
                  f"Setting trimmed_idx=idx2-1 ({idx2-1}).")
            trimmed_idx = idx2 - 1

    # assign trimmed timestamp if valid
    if 0 <= trimmed_idx < len(df):
        new_end = df.loc[trimmed_idx, 'Time [s]']
        print(f"[TRIM] Window #{i} ({p1_start}–{p1_end}) trimmed to {new_end} "
              f"(2 rows before P2_start={p2_start}).")
        merged_df.at[i, 'Trimmed_Peak_end'] = new_end
    else:
        print(f"[OOB] trimmed_idx={trimmed_idx} out of bounds, skipping.")

# --- Save updated DataFrame (overwrites original) ---
merged_df.to_csv(MERGED_PATH, index=False)
print(f"\nUpdated file saved to: {MERGED_PATH}")

# --- New Summary Checks ---
# 1. Total peaks with non-null Trimmed_Peak_end
non_null_trimmed = merged_df['Trimmed_Peak_end'].notna().sum()
print(f"Total peaks with non-null Trimmed_Peak_end: {non_null_trimmed} out of {total_windows}")

# 2. Count overlaps using Trimmed_Peak_end > next Peak_start
trimmed_overlap_count = 0
for i in range(total_windows - 1):
    trimmed_end_i = merged_df.loc[i, 'Trimmed_Peak_end']
    next_start    = merged_df.loc[i+1, 'Peak_start']
    if pd.notna(trimmed_end_i) and trimmed_end_i > next_start:
        trimmed_overlap_count += 1

print(f"Total number of overlapping peaks (Trimmed_Peak_end > next Peak_start): {trimmed_overlap_count}")
