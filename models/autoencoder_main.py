import numpy as np
import pandas as pd
from models.preprocess_ts import preprocess_timeseries
from models.autoencoder_1d_cnn import build_autoencoder
from typing import List

def create_windows(
        data: np.ndarray,
                   window_size: int,
                   stride: int = 1) -> np.ndarray:
    T, C = data.shape
    windows = np.lib.stride_tricks.sliding_window_view(data, (window_size, C))
    return windows[::stride, 0, :]

def main(
    csv_path: str,
    out_path: str = "models/processed_data/detected_bursts.csv",
    time_col: str = "Time (ms)",
    smooth_window: int = 5,
    roll_window: int = 500,
    window_size: int = 32,
    threshold_pct: float = 95.0,
    epochs: int = 20,
    batch_size: int = 128
):
    # Load and normalize
    df = pd.read_csv(csv_path)
    features: List[str] = [c for c in df.columns if c != time_col]
    df_norm = preprocess_timeseries(
        df, time_col, features,
        smooth_window=smooth_window, smooth_poly=2, roll_window=roll_window
    )

    # Windowing
    data_arr = df_norm[features].values
    windows = create_windows(data_arr, window_size)

    # Build & train
    ae = build_autoencoder(window_size, n_channels=data_arr.shape[1])
    ae.fit(windows, windows,
           epochs=epochs, batch_size=batch_size, validation_split=0.1)

    # Reconstruction error
    recon = ae.predict(windows)
    errors = np.mean((windows - recon)**2, axis=(1,2))

    # Threshold & collect bursts
    thresh = np.percentile(errors, threshold_pct)
    anomalous = np.where(errors > thresh)[0]

    # Map back to time and save
    times = df_norm[time_col].values
    bursts = []
    for idx in anomalous:
        bursts.append({
            "start_ms": times[idx],
            "end_ms":   times[idx + window_size - 1],
            "error":    float(errors[idx])
        })

    pd.DataFrame(bursts).to_csv(out_path, index=False)
    print(f"Detected {len(bursts)} bursts → saved to {out_path}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1], out_path=sys.argv[2] if len(sys.argv)>2 else "detected_bursts.csv")

# python -m models.autoencoder_main data/NPs_BHVO_Oct23_full.csv models/processed_data/detected_bursts.csv
