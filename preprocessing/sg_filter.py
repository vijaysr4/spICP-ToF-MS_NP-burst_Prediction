import pandas as pd
from scipy.signal import savgol_filter
from findpeaks_scipy import detect_peaks_all_channels, plot_channels_with_peaks


def main():
    # 1. Load and sort raw data
    data_path = "../data/NPs_BHVO_Oct23_full.csv"
    df_raw = pd.read_csv(data_path)
    df_raw.sort_values("Time (ms)", inplace=True)
    time_col = "Time (ms)"
    isotopes = [c for c in df_raw.columns if c != time_col]

    # 2. Detect peaks on raw data and plot
    peaks_raw, props_raw = detect_peaks_all_channels(
        df_raw,
        time_col=time_col,
        threshold_std=2.0,
        min_distance=50,
        prominence_std=1.0
    )
    plot_channels_with_peaks(
        df_raw,
        peaks_raw,
        time_col=time_col,
        cols=4,
        figsize_base=(30, 3),
        colormap="tab20",
        save_path="Raw_Isotopes_With_Peaks.png"
    )
    print("Raw peak detection complete. Plot saved to Raw_Isotopes_With_Peaks.png")

    # 3. Apply Savitzky–Golay smoothing (SG filter only)
    smooth_df = pd.DataFrame({time_col: df_raw[time_col]})
    for col in isotopes:
        smooth_df[col] = savgol_filter(
            df_raw[col].values,
            window_length=5,
            polyorder=2,
            mode='interp'
        )
    smooth_path = "sg_filter_NPs_BHVO_Oct23.csv"
    smooth_df.to_csv(smooth_path, index=False)
    print(f"Smoothed data saved to {smooth_path}")

    # 4. Detect peaks on smoothed data and plot (same order)
    peaks_smooth, props_smooth = detect_peaks_all_channels(
        smooth_df,
        time_col=time_col,
        threshold_std=2.0,
        min_distance=50,
        prominence_std=1.0
    )
    plot_channels_with_peaks(
        smooth_df,
        peaks_smooth,
        time_col=time_col,
        cols=4,
        figsize_base=(30, 3),
        colormap="tab20",
        save_path="Smoothed_Isotopes_With_Peaks.png"
    )
    print("Smoothed peak detection complete. Plot saved to Smoothed_Isotopes_With_Peaks.png")


if __name__ == "__main__":
    main()