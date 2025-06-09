import os
import pandas as pd
import numpy as np

ION_CHANNELS = [
    '23Na', '24Mg', '26Mg', '27Al', '28Si', '29Si',
    '39K', '40Ca', '48Ti', '54Fe', '55Mn', '56Fe', '60Ni'
]

def assign_particle(df, times):
    df_idx = df.set_index('Time (ms)')
    records = []
    for t in np.unique(times):
        idx = df_idx.index.get_indexer([t], method='nearest')[0]
        time_key = df_idx.index[idx]
        row = df_idx.iloc[idx]
        top_elem = row.idxmax()
        records.append({'Time (ms)': time_key, 'TopElement': top_elem})
    return pd.DataFrame(records)

def annotate_all(input_csv, results_dir, output_csv):
    df = pd.read_csv(input_csv)
    all_tables = []
    for ch in ION_CHANNELS:
        pkl_file = os.path.join(results_dir, f'peaks_{ch}.pkl')
        results: dict = pd.read_pickle(pkl_file)
        for method, times in results.items():
            tbl = assign_particle(df, times)
            tbl.insert(0, 'Method', method)
            tbl.insert(1, 'Channel', ch)
            all_tables.append(tbl)
    out = pd.concat(all_tables, ignore_index=True)
    out.to_csv(output_csv, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input",   required=True, help="CSV with Time (ms) + channels")
    parser.add_argument("-r","--results", required=True, help="Dir with peaks_*.pkl files")
    parser.add_argument("-o","--output",  required=True, help="CSV to save annotated peaks")
    args = parser.parse_args()

    annotate_all(args.input, args.results, args.output)


# python assign_particles_all.py \
#   --input ../data/NPs_BHVO_Oct23_full.csv \
#   --results results/ \
#   --output annotated_peaks_all.csv
