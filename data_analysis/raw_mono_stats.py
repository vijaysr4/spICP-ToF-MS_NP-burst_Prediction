import os
import pandas as pd

# Paths (as you already have them)
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT,
                        'data_files',
                        'mono_channel_and_peak_period')

data_fpath = os.path.join(DATA_DIR, 'cleaned_data.csv')
window_fpath = os.path.join(DATA_DIR, 'events20240515_e.csv')

# Read in the two files
df = pd.read_csv(data_fpath)
window_df = pd.read_csv(window_fpath)

# Print the first 5 rows of each
print("=== cleaned_data.csv head ===")
print(df.head(), "\n")

print("=== events20240515_e.csv head ===")
print(window_df.head())


# Compute each peak’s duration
durations = window_df['Peak_end'] - window_df['Peak_start']

# Calculate summary stats
num_peaks = len(durations)
min_peak  = durations.min()
max_peak  = durations.max()
avg_peak  = durations.mean()

# Print them out
print(f"Total peaks:           {num_peaks}")
print(f"Min peak duration:   {min_peak:.6f} s")
print(f"Max peak duration:   {max_peak:.6f} s")
print(f"Avg peak duration:   {avg_peak:.6f} s")
