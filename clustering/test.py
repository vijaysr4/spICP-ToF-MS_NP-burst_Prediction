import pandas as pd
import os
BASE = os.path.dirname(__file__)
fpath = os.path.join(BASE, 'fingerprints', 'raw_data_ion_fingerprints.csv')

df = pd.read_csv(fpath, index_col=False)

print(df.head())

print("Number of windows:", len(df))