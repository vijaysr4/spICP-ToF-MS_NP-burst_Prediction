import os
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

# Configuration
WINDOW_SIZE: int = 100           # timesteps per window
STRIDE: int = 50                 # sliding step between windows
SCALER_TYPE: str = 'standard'    # 'standard' or 'robust'
OUTPUT_DIR: str = 'ae_input_windows'

# Load dataset once
RAW_CSV_PATH: str = 'ion_data.csv'
df_raw = (
    pd.read_csv(RAW_CSV_PATH)
      .sort_values('Time (ms)')
      .reset_index(drop=True)
)

# Define feature columns (excluding timestamp)
cols = df_raw.columns.drop('Time (ms)')

# Prepare data array for windowing (no baseline correction)
data_array = df_raw[cols].to_numpy()


def make_windows(
    data: np.ndarray,
    win_size: int = WINDOW_SIZE,
    stride: int = STRIDE
) -> np.ndarray:
    """
    Convert multivariate series into overlapping windows.

    Args:
        data: 2D array of shape (n_samples, n_features).
        win_size: Number of timesteps per window.
        stride: Step size between window starts.

    Returns:
        3D array of shape (n_windows, win_size, n_features).
    """
    n_samples, n_features = data.shape
    windows = []
    for start in range(0, n_samples - win_size + 1, stride):
        windows.append(data[start:start + win_size])
    return np.stack(windows)


def scale_windows(
    windows: np.ndarray,
    scaler_type: str = SCALER_TYPE
) -> Tuple[np.ndarray, object]:
    """
    Fit and apply a scaler to windowed data.

    Args:
        windows: 3D array (n_windows, win_size, n_features).
        scaler_type: 'standard' or 'robust'.

    Returns:
        Tuple of (scaled_windows, fitted_scaler).
    """
    n_w, w, f = windows.shape
    flat = windows.reshape(-1, f)

    if scaler_type == 'robust':
        scaler = RobustScaler().fit(flat)
    else:
        scaler = StandardScaler().fit(flat)

    flat_scaled = scaler.transform(flat)
    scaled = flat_scaled.reshape(n_w, w, f)
    return scaled, scaler


def save_windows(
    windows: np.ndarray,
    out_dir: str = OUTPUT_DIR
) -> None:
    """
    Save each window as a .npy file for training.

    Args:
        windows: 3D array of windows.
        out_dir: Directory to save .npy files.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, w in enumerate(windows):
        np.save(os.path.join(out_dir, f'window_{i:05d}.npy'), w)
    print(f"Saved {len(windows)} windows to '{out_dir}'")


# Execute preprocessing without baseline correction
windows = make_windows(data_array, WINDOW_SIZE, STRIDE)
windows_scaled, scaler = scale_windows(windows, SCALER_TYPE)
save_windows(windows_scaled, OUTPUT_DIR)
