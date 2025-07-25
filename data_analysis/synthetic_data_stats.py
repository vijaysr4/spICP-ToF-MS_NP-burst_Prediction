import os
import pandas as pd

# synthetic_data_stats.py lives in project_root/data_analysis/
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT,
                        'data_files',
                        'mono_channel_and_peak_period')

data_fpath = os.path.join(DATA_DIR, 'cleaned_data.csv')
window_fpath = os.path.join(DATA_DIR, 'events20240515_e_trimmed_eps_0.001.csv')

# Load data
df = pd.read_csv(data_fpath)
w_df = pd.read_csv(window_fpath)

print("First rows of intensity data:")
print(df.head())
print("\nFirst rows of detected peaks:")
print(w_df.head())

# 1. Total number of peaks
n_peaks = len(w_df)

# 2. Compute each peak’s span
spans = w_df['end_trimmed'] - w_df['start']
avg_span = spans.mean()
min_span = spans.min()
max_span = spans.max()

# 3. Merge overlapping windows to get total peak time
intervals = w_df[['start', 'end_trimmed']].sort_values('start').to_numpy()
merged = []
for start, end in intervals:
    if not merged or start > merged[-1][1]:
        merged.append([start, end])
    else:
        merged[-1][1] = max(merged[-1][1], end)
total_peak_time = sum(end - start for start, end in merged)

# 4. Compute the total time span of your original data
t_min, t_max = df['Time [s]'].min(), df['Time [s]'].max()
total_time = t_max - t_min

# 5. Percentage of time spent in peaks
peak_pct = 100 * total_peak_time / total_time

# 6. Top 10 shortest peaks + original end + next window
w2 = w_df.copy()
w2['span'] = spans
# sort by start to identify the "next" window
w2 = w2.sort_values('start').reset_index(drop=True)

# shift to get the next window in time
w2['next_start'] = w2['start'].shift(-1)
w2['next_end_trimmed'] = w2['end_trimmed'].shift(-1)

# pick the 10 rows with the smallest span
min10 = w2.nsmallest(10, 'span')[[
    'start',
    'end',            # original end
    'end_trimmed',
    'span',
    'next_start',
    'next_end_trimmed'
]]

# --- Print everything neatly ---
print("\n=== Peak Statistics ===")
print(f"Total number of peaks:        {n_peaks}")
print(f"Minimum peak span (seconds):  {min_span:.6f}")
print(f"Maximum peak span (seconds):  {max_span:.6f}")
print(f"Average peak span (seconds):  {avg_span:.6f}")
print(f"Total peak time (seconds):    {total_peak_time:.6f}")
print(f"Full data time span (s):      {total_time:.6f}")
print(f"Percentage of time in peaks:  {peak_pct:.2f}%")

print("\nTop 10 shortest peaks and the following window:")
print(min10.to_string(index=False))
