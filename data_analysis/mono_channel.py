import os
import pandas as pd

# 1. Build paths relative to this script
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

csv_time = os.path.join(
    PROJECT_ROOT,
    "data_files",
    "mono_channel_and_peak_period",
    "synthTRA+20240515_e.csv"
)
csv_peaks = os.path.join(
    PROJECT_ROOT,
    "data_files",
    "mono_channel_and_peak_period",
    "events20240515_e.csv"
)

# 2. Load and clean the time series (seconds)
df = pd.read_csv(csv_time, header=0)
df = df.dropna(how='all').reset_index(drop=True)
df['Time_s'] = df['Time [s]']  # already in seconds

# 3. Load the peak windows (seconds)
#    Assume columns: peak_id, start_s, end_s
df_peaks = pd.read_csv(csv_peaks, header=0, names=['peak_id', 'start_s', 'end_s'])

# 4. Merge overlapping intervals (in seconds)
intervals = df_peaks[['start_s', 'end_s']].sort_values('start_s').to_numpy()
merged = []
for start, end in intervals:
    if not merged or start > merged[-1][1]:
        merged.append([start, end])
    else:
        merged[-1][1] = max(merged[-1][1], end)

# 5. Calculate total duration and covered duration (seconds)
total_duration_s = df['Time_s'].max() - df['Time_s'].min()
covered_s = sum(end - start for start, end in merged)

# 6. Print results
pct_time = covered_s / total_duration_s * 100
print(f"Total time span:      {total_duration_s:.6f} s")
print(f"Time covered by peaks:{covered_s:.6f} s")
print(f"Percentage in peaks:   {pct_time:.2f}%")
