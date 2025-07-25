import os
import pandas as pd

# Script to trim overlapping peak windows, generate labels, and report statistics.

# 1) Setup paths
script_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
data_dir = os.path.join(
    project_root,
    'data_files',
    'mono_channel_and_peak_period'
)

data_fpath = os.path.join(data_dir, 'cleaned_data.csv')
window_fpath = os.path.join(data_dir, 'events20240515_e.csv')

# 2) Load data
print("Loading data...")
df = pd.read_csv(data_fpath)  # ['Time [s]', 'intensity']
windows = (
    pd.read_csv(window_fpath)
      .sort_values('Peak_start')
      .reset_index(drop=True)
)

# 3) Compute sampling interval and gap
dt = round(df['Time [s]'].diff().median(), 6)  # time per sample
gap = round(2 * dt, 6)  # two-sample gap
print(f"Sampling interval dt={dt}, gap={gap}")

# 4) Trim windows with gap rules
two_gap_count = 0
one_gap_count = 0
zero_gap_count = 0
trimmed_ends = []
for i, row in windows.iterrows():
    start = row['Peak_start']
    end = row['Peak_end']
    trimmed_end = end

    if i < len(windows) - 1:
        next_start = windows.at[i + 1, 'Peak_start']
        if next_start < end:
            # try two-sample gap
            two_gap = round(next_start - gap, 6)
            if two_gap > start:
                trimmed_end = two_gap
                two_gap_count += 1
            else:
                # try one-sample gap
                one_gap = round(next_start - dt, 6)
                if one_gap > start:
                    trimmed_end = one_gap
                    one_gap_count += 1
                else:
                    # no gap possible, zero-length window
                    trimmed_end = start
                    zero_gap_count += 1

    trimmed_ends.append(trimmed_end)

windows['trimmed_end'] = trimmed_ends

# Report trimming summary
print(f"Windows trimmed with 2-sample gap: {two_gap_count}")
print(f"Windows trimmed with 1-sample gap: {one_gap_count}")
print(f"Windows collapsed to zero span:   {zero_gap_count}")

# 5) Generate labels
df['Label'] = 0
for _, row in windows.iterrows():
    s, te = row['Peak_start'], row['trimmed_end']
    mask = (df['Time [s]'] >= s) & (df['Time [s]'] <= te)
    df.loc[mask, 'Label'] = 1

# 6) Compute statistics
total_windows = len(windows)
total_clusters = int(((df['Label'] == 1) & (df['Label'].shift(fill_value=0) == 0)).sum())
print(f"Total peaks in windows file:         {total_windows}")
print(f"Total peak clusters in labels:       {total_clusters}")

# 7) Identify windows not fully labeled
mismatches = []
for _, row in windows.iterrows():
    s, te = row['Peak_start'], row['trimmed_end']
    mask = (df['Time [s]'] >= s) & (df['Time [s]'] <= te)
    if (df.loc[mask, 'Label'] == 0).any():
        mismatches.append((s, te))
print(f"Windows not fully labeled:           {len(mismatches)}")

# 8) Identify negative trimmed spans
windows['trimmed_span'] = windows['trimmed_end'] - windows['Peak_start']
negatives = windows[windows['trimmed_span'] < 0]
print(f"Windows with negative trimmed span:  {len(negatives)}")
if not negatives.empty:
    print(negatives[['Peak_start', 'trimmed_end', 'trimmed_span']].to_string(index=False))

# 9) Compute spans for reporting
windows['orig_span'] = windows['Peak_end'] - windows['Peak_start']

# 10) Top-10 smallest spans (including zero-length)
print("\nTop 10 by original span:")
print(windows.nsmallest(10, 'orig_span')[['Peak_start', 'Peak_end', 'orig_span']].to_string(index=False))
print("\nTop 10 by trimmed span:")
print(windows.nsmallest(10, 'trimmed_span')[['Peak_start', 'trimmed_end', 'trimmed_span']].to_string(index=False))
