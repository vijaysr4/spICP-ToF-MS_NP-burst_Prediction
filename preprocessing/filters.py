import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def savgol_smooth(df: pd.DataFrame, cols: list, window_length: int = 5, polyorder: int = 2, mode: str = 'interp') -> pd.DataFrame:
    """
    Apply Savitzky-Golay smoothing to specified columns in-place.

    Args:
        df: DataFrame containing raw signals.
        cols: List of column names to smooth.
        window_length: Window size (odd integer).
        polyorder: Polynomial order for filter.
        mode: Boundary mode for filter ('interp', 'mirror', etc.).

    Returns:
        DataFrame with smoothed columns.
    """
    smoothed = df[cols].copy()
    for col in cols:
        smoothed[col] = savgol_filter(df[col].values,
                                      window_length,
                                      polyorder,
                                      mode=mode)
    return smoothed


def rolling_baseline_norm(smoothed: pd.DataFrame,
                          window: int = 500,
                          center: bool = True,
                          min_periods: int = 1) -> pd.DataFrame:
    """
    Compute baseline-normalized residuals for smoothed signals.

    Args:
        smoothed: DataFrame of smoothed signals.
        window: Rolling window length.
        center: Center the window (True/False).
        min_periods: Minimum observations for rolling stats.

    Returns:
        DataFrame of normalized residuals (z-scores).
    """
    rolling_mean = smoothed.rolling(window=window,
                                   center=center,
                                   min_periods=min_periods).mean()
    rolling_std  = smoothed.rolling(window=window,
                                   center=center,
                                   min_periods=min_periods).std(ddof=0)
    residuals = (smoothed - rolling_mean) / rolling_std
    return residuals.replace([np.inf, -np.inf], np.nan).fillna(0)


def preprocess_timeseries(df: pd.DataFrame,
                          time_col: str,
                          isotope_cols: list,
                          smooth_window: int = 5,
                          smooth_poly: int = 2,
                          roll_window: int = 500) -> pd.DataFrame:
    """
    Full pipeline: SG smooth + rolling baseline normalization.
    """
    # Preserve time
    result = pd.DataFrame({time_col: df[time_col].values}, index=df.index)

    # Smooth
    smooth_df = savgol_smooth(df, isotope_cols,
                              window_length=smooth_window,
                              polyorder=smooth_poly)
    # Baseline normalize
    norm_df = rolling_baseline_norm(smooth_df,
                                    window=roll_window)
    for col in isotope_cols:
        result[col] = norm_df[col].values

    return result
