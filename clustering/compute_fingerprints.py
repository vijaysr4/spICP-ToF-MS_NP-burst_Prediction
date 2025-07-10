import pandas as pd
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def load_data(
    data_path: Path,
    windows_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the ion-count time series and the peak windows files.

    Args:
        data_path: Path to the CSV file with columns ['Time (ms)', ion columns...].
        windows_path: Path to the CSV file with columns ['start_ms', 'end_ms', ...].

    Returns:
        A tuple (df, windows_df):
            - df: DataFrame with a 'time' column and one column per ion.
            - windows_df: DataFrame with 'start_ms' and 'end_ms' columns.
    """
    df = pd.read_csv(data_path)
    df = df.rename(columns={'Time (ms)': 'time'})
    windows_df = pd.read_csv(windows_path)
    return df, windows_df


def compute_fingerprints(
    df: pd.DataFrame,
    windows_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each window, sum the ion-counts over the time interval and normalize
    to fractions (summing to 1), displaying progress.

    Args:
        df: Time-series DataFrame with 'time' and ion columns.
        windows_df: DataFrame with 'start_ms' and 'end_ms' defining each window.

    Returns:
        A DataFrame with one row per window, columns:
            ['start_ms', 'end_ms', <ion columns...>]
        where each ion column is the fraction of counts in that window.
    """
    ion_cols: List[str] = [c for c in df.columns if c != 'time']
    results: List[dict] = []

    for w in tqdm(
        windows_df.itertuples(index=False),
        total=len(windows_df),
        desc="Computing fingerprints"
    ):
        # using attribute access on namedtuple
        mask = (df['time'] >= w.start_ms) & (df['time'] <= w.end_ms)
        sub = df.loc[mask, ion_cols]

        sums = sub.sum()
        total = sums.sum()
        if total == 0:
            fractions = sums * 0.0
        else:
            fractions = sums / total

        row = {
            'start_ms': w.start_ms,
            'end_ms': w.end_ms,
            **fractions.to_dict()
        }
        results.append(row)

    return pd.DataFrame(results)


def save_fingerprints(
    fingerprints: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save the fingerprints DataFrame to CSV.

    Args:
        fingerprints: DataFrame returned by compute_fingerprints().
        output_path: Path where to write the CSV.
    """
    fingerprints.to_csv(output_path, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute ion-count fingerprints for peak windows."
    )
    parser.add_argument(
        '--data', type=Path, required=True,
        help="CSV file with time-series ion counts (has 'Time (ms)' column)."
    )
    parser.add_argument(
        '--windows', type=Path, required=True,
        help="CSV file with detected windows (columns: 'start_ms','end_ms')."
    )
    parser.add_argument(
        '--output', type=Path, default=Path('ion_fingerprints.csv'),
        help="Path to save the normalized fingerprints CSV."
    )

    args = parser.parse_args()

    # Load
    df, windows_df = load_data(args.data, args.windows)

    # Compute with progress bar
    fingerprints = compute_fingerprints(df, windows_df)

    # Save
    save_fingerprints(fingerprints, args.output)

    print(f"Saved normalized ion fingerprints to {args.output}")


# python clustering/compute_fingerprints.py \
#   --data data_files/NPs_BHVO_Oct23_full.csv \
#   --windows models/processed_data/sg_filter_detected_bursts.csv \
#   --output clustering/fingerprints/sg_filter_ion_fingerprints.csv
