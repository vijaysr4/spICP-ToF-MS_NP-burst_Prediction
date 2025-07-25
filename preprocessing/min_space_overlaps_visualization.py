import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

# --- Setup paths ---
THIS_DIR     = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data_files', 'mono_channel_and_peak_period')
PLOTS_DIR    = os.path.join(THIS_DIR, 'plots')

# ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

CLEANED_PATH = os.path.join(DATA_DIR, 'cleaned_data.csv')
EVENTS_PATH  = os.path.join(DATA_DIR, 'events20240515_e.csv')

# --- Load datasets ---
df       = pd.read_csv(CLEANED_PATH)
peaks_df = pd.read_csv(EVENTS_PATH)

# --- Define overlapping peak pairs with their gap_rows ---
# Format: (P1_start, P1_end, P2_start, P2_end, gap_rows)
overlaps = [
    (7.4282, 7.4354, 7.4287, 7.4487, 4),
    (17.8303,17.8364,17.8305,17.8493,1),
    (21.9652,21.9785,21.9653,21.9720,0),
    (23.1617,23.1754,23.1619,23.1657,1),
    (25.6541,25.6783,25.6543,25.6585,1),
    (26.2443,26.2614,26.2451,26.2590,7),
    (30.6318,30.6435,30.6320,30.6475,1),
    (35.9087,35.9237,35.9095,35.9294,7),
    (38.6509,38.6702,38.6511,38.6657,1),
    (39.3125,39.3267,39.3132,39.3250,6),
    (42.0018,42.0208,42.0024,42.0199,5),
    (43.0794,43.1168,43.0801,43.0977,6),
    (46.9643,46.9868,46.9651,46.9750,7),
    (55.4495,55.4652,55.4504,55.4671,8),
    (57.0621,57.0907,57.0626,57.0717,4),
    (59.6460,59.6766,59.6470,59.6515,9),
]

# --- Plot, highlight, and save each overlapping window ---
for i, (p1s, p1e, p2s, p2e, gap) in enumerate(overlaps, start=1):
    # determine window bounds
    window_start = p1s - 0.01
    window_end   = max(p2e, p1e) + 0.01

    # subset data
    mask = (df['Time [s]'] >= window_start) & (df['Time [s]'] <= window_end)
    seg  = df.loc[mask]

    fig, ax = plt.subplots(figsize=(14, 4), dpi=200)
    ax.plot(seg['Time [s]'], seg['intensity'], linewidth=1.2, color='gray', alpha=0.7)
    ax.axvspan(p1s, p1e, color='tab:orange', alpha=0.3)
    ax.axvspan(p2s, p2e, color='tab:green',  alpha=0.3)

    ax.grid(which='major', linestyle='--', linewidth=0.4, alpha=0.6)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.2, alpha=0.4)

    # Use gap_rows for title and filename
    ax.set_title(f"Overlap #{i}: gap_rows {gap} rows", fontsize=16, weight='bold')
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Intensity", fontsize=14)

    ax.set_xlim(window_start, window_end)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=0)

    ax.legend(['Signal', 'Peak 1', 'Peak 2'], fontsize=12, loc='upper right', frameon=False)
    fig.tight_layout()

    filename = f"monochannel_overlap_gap{gap}_{i}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
