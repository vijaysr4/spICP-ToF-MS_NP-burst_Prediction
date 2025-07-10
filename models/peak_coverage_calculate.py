import argparse
import pandas as pd


def merge_intervals(intervals):
    """
    Merge a list of intervals (start, end) and return merged list.
    """
    if not intervals:
        return []

    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    merged = [intervals_sorted[0]]

    for current in intervals_sorted[1:]:
        last = merged[-1]
        # If overlapping or contiguous, merge
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged


def calculate_coverage(windows, total_duration):
    """
    Given list of (start, end) windows and total duration in ms,
    compute covered duration and percentage.
    Returns: (covered_ms, percentage)
    """
    merged = merge_intervals(windows)
    covered = sum(end - start for start, end in merged)
    percentage = covered / total_duration * 100 if total_duration > 0 else 0
    return covered, percentage


def main():
    parser = argparse.ArgumentParser(
        description='Calculate percentage of time covered by detected peak windows.')
    parser.add_argument(
        '--windows', required=True,
        help='CSV file with columns: start_ms, end_ms')
    parser.add_argument(
        '--data', required=True,
        help='Original data CSV file (must contain a time column in ms)')
    args = parser.parse_args()

    # Load predicted windows
    wdf = pd.read_csv(args.windows)
    if not {'start_ms', 'end_ms'}.issubset(wdf.columns):
        raise ValueError("Windows CSV must contain 'start_ms' and 'end_ms' columns.")
    windows = list(wdf[['start_ms', 'end_ms']].itertuples(index=False, name=None))

    # Load original data to determine total duration
    ddf = pd.read_csv(args.data)
    # Identify time column (assumes 'Time' in name)
    time_cols = [col for col in ddf.columns if 'time' in col.lower()]
    if not time_cols:
        raise ValueError("Data CSV must contain a time column (e.g. 'Time (ms)')")
    time_col = time_cols[0]
    times = ddf[time_col].astype(float)
    total_duration = times.max() - times.min()

    # Calculate coverage
    covered_ms, pct = calculate_coverage(windows, total_duration)

    print(f"Total covered duration: {covered_ms:.5f} ms")
    print(f"Total duration: {total_duration:.5f} ms (from '{time_col}' column)")
    print(f"Coverage: {pct:.5f}%")

if __name__ == '__main__':
    main()


# python models/peak_coverage_calculate.py --windows models/processed_data/sg_filter_detected_bursts.csv --data data_files/NPs_BHVO_Oct23_full.csv
