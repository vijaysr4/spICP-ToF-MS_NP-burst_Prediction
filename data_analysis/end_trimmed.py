import pandas as pd
import os

# Locate your data directory
script_dir   = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
DATA_DIR     = os.path.join(project_root, 'data_files', 'mono_channel_and_peak_period')

# Load the CSV
infile   = 'events20240515_e_trimmed_eps_0.001.csv'
csv_path = os.path.join(DATA_DIR, infile)
df       = pd.read_csv(csv_path)

print(df.head(10))


intervals = df[['start', 'end_trimmed']].sort_values('start').reset_index(drop=True)

# 2. Compute the running max of end_trimmed up *to* the previous row
intervals['max_prev_end'] = intervals['end_trimmed'].cummax().shift(1).fillna(0)

# 3. Flag any start that begins before that running max
intervals['overlap_any'] = intervals['start'] < intervals['max_prev_end']

# 4. Extract the overlaps
overlaps = intervals[intervals['overlap_any']]
print(overlaps)

# Compute spans
df['span_end'] = df['end'] - df['start']
df['span_end_trimmed'] = df['end_trimmed'] - df['start']

# Get the 20 smallest spans for each
top20_untrimmed = df.nsmallest(20, 'span_end')[['start', 'end', 'span_end']]
top20_trimmed   = df.nsmallest(20, 'span_end_trimmed')[['start', 'end_trimmed', 'span_end_trimmed']]

print("Top 20 intervals by smallest (end - start):")
print(top20_untrimmed.to_string(index=False))

print("\nTop 20 intervals by smallest (end_trimmed - start):")
print(top20_trimmed.to_string(index=False))