import os
import pandas as pd

# events_top_pretrim.py lives in project_root/data_analysis/
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT,
                        'data_files',
                        'mono_channel_and_peak_period')

# Input file with spans already added
in_fname = 'events20240515_e_with_spans.csv'
in_path  = os.path.join(DATA_DIR, in_fname)

# 1. Load the data
w_df = pd.read_csv(in_path)

# 2. Sort by pre‑trim span and take top 20 smallest
top20 = w_df.nsmallest(30, 'span_pre-trim')[[
    'start',
    'end',
    'span_pre-trim'
]]

# 3. Print results
print("Top 30 windows with smallest pre‑trim span:")
print(top20.to_string(index=False))
