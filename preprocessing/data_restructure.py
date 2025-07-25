import os
import pandas as pd

# Build paths relative to this script
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# load + clean as before
csv_peaks = os.path.join(
    PROJECT_ROOT,
    "data_files",
    "mono_channel_and_peak_period",
    "events20240515_e.csv"
)
df = pd.read_csv(csv_peaks, header=None, names=['idx', 'Peak_start', 'Peak_end'])
df = df.dropna(how='all').drop('idx', axis=1).reset_index(drop=True)

# build output path
output_path = os.path.join(
    PROJECT_ROOT,
    "data_files",
    "mono_channel_and_peak_period",
    "original.csv"
)

# save without the pandas index
df.to_csv(csv_peaks, index=False)

print(f"Saved cleaned data to {output_path}")
