import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from tqdm import tqdm

def multivariate_mahalanobis_threshold(
    df: pd.DataFrame,
    channels: list[str],
    time_col: str = 'Time (ms)',
    window: int = 500,
    alpha: float = 0.001
) -> (np.ndarray, np.ndarray, float):
    """
    Compute rolling-window Mahalanobis distances and threshold.

    Returns:
      times       : array of all timestamps
      D2          : array of Mahalanobis distances
      thr         : scalar chi-square threshold
    """
    d   = len(channels)
    thr = chi2.ppf(1 - alpha, df=d)
    n   = len(df)
    D2  = np.zeros(n)
    X   = df[channels].to_numpy(dtype=float)

    # show progress over the rolling-window computation
    for i in tqdm(range(window, n), desc='Computing D²'):
        W     = X[i-window:i, :]
        mu    = W.mean(axis=0)
        cov   = np.cov(W, rowvar=False) + 1e-6 * np.eye(d)
        inv   = np.linalg.inv(cov)
        delta = X[i, :] - mu
        D2[i] = float(delta @ inv @ delta)

    return df[time_col].to_numpy(), D2, thr

if __name__ == "__main__":
    # ensure output directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('processed_data', exist_ok=True)

    # Load your data
    df = pd.read_csv('../data/NPs_BHVO_Oct23_full.csv')

    # Define ion channels
    ION_CHANNELS = [
        '23Na','24Mg','26Mg','27Al','28Si','29Si',
        '39K','40Ca','48Ti','54Fe','55Mn','56Fe','60Ni'
    ]

    # Compute Mahalanobis distances and threshold (with progress bar)
    times, D2, threshold = multivariate_mahalanobis_threshold(
        df,
        channels=ION_CHANNELS,
        window=500,
        alpha=0.001
    )

    # Identify bursts
    mask = D2 > threshold
    burst_times = times[mask]
    burst_vals  = D2[mask]

    # Plot and save in /results
    plt.figure(figsize=(12, 4))
    plt.plot(times, D2, label='Mahalanobis D²')
    plt.hlines(threshold, times[0], times[-1],
               colors='r', linestyles='--',
               label=f'χ² threshold ({threshold:.1f})')
    plt.scatter(burst_times, burst_vals, c='r', s=10,
                label='Detected bursts')
    plt.xlabel('Time (ms)')
    plt.ylabel('D²')
    plt.title('Multivariate Mahalanobis Burst Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/mahalanobis_bursts.png', dpi=150)
    plt.close()

    # Save burst times in /processed_data
    pd.Series(burst_times, name='Time (ms)')\
      .to_csv('processed_data/mahalanobis_bursts.csv',
              index=False, header=True)
