import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from data import *
from data.graph import plot_elements_time_series
from typing import List

def preprocess_timeseries(
    df: pd.DataFrame,
    time_col: str,
    isotope_cols: List[str],
    smooth_window: int = 5,
    smooth_poly: int = 2,
    roll_window: int = 500
) -> pd.DataFrame:
    """
    Smooth and baseline‐normalize multivariate spICP‐ToF‐MS data,
    while preserving the time column for downstream peak mapping.

    Steps:
      1. Copy the time column.
      2. Apply Savitzky–Golay smoothing to each isotope channel.
      3. Compute rolling mean and std for baseline.
      4. Compute residuals = (smoothed – mean) / std.
      5. Replace infinities (from zero‐std) with 0.

    Args:
        df: Raw DataFrame containing time_col and isotope_cols.
        time_col: Name of the time column (e.g. "Time (ms)").
        isotope_cols: List of isotope channel column names.
        smooth_window: Window length for SG filter (must be odd).
        smooth_poly: Polynomial order for SG smoothing.
        roll_window: Window size (in rows) for rolling statistics.

    Returns:
        A DataFrame with:
          - time_col unchanged,
          - one column per isotope containing the cleaned normalized residuals.
    """
    # Preserve time column
    result = pd.DataFrame({time_col: df[time_col].values}, index=df.index)

    # Smooth isotope channels to suppress single-bin noise
    smooth = df[isotope_cols].apply(
        lambda col: savgol_filter(col, smooth_window, smooth_poly, mode='interp'),
        axis=0,
        result_type='broadcast'
    )

    # Rolling baseline statistics
    rolling_mean = smooth.rolling(window=roll_window, center=True, min_periods=1).mean()
    rolling_std  = smooth.rolling(window=roll_window, center=True, min_periods=1).std(ddof=0)

    # Compute normalized residuals
    residuals = (smooth - rolling_mean) / rolling_std

    # Remove infinities and NaNs from zero-std divisions
    residuals = residuals.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Attach residuals back to result
    for col in isotope_cols:
        result[col] = residuals[col].values

    return result



df = pd.read_csv("data/NPs_BHVO_Oct23_full.csv")
isotopes = [c for c in df.columns if c != "Time (ms)"]
norm_df = preprocess_timeseries(df, "Time (ms)", isotopes)
print(norm_df.head())
plot_elements_time_series(
    df=norm_df,
    time_col="Time (ms)",
    cols=4,
    figsize_base=(30, 3),
    colormap="tab20",
    save_path="models/Norm_Individual_Isotops_plot.png"
)

# execute from root directory
# python -m models.preprocess_ts