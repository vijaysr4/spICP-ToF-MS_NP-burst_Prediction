import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# Custom styling without relying on external style packages
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 20,
    'axes.facecolor': '#f9f9f9',
    'figure.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'lines.linewidth': 1.8,
    'lines.color': '#1f77b4'
})

# Paths setup
THIS_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data directory relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data_files', 'mono_channel_and_peak_period')

# File paths
data_fpath = os.path.join(DATA_DIR, 'cleaned_data.csv')
window_fpath = os.path.join(DATA_DIR, 'events20240515_e_trimmed_eps_0.001.csv')

# Load data
df = pd.read_csv(data_fpath)
w_df = pd.read_csv(window_fpath)

# Determine axis limits from data
x_min, x_max = df['Time [s]'].min(), df['Time [s]'].max()
y_min, y_max = df['intensity'].min(), df['intensity'].max()

# Helper for formatting x-axis ticks
def format_xaxis(ax):
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='x', which='major', rotation=45, length=6)
    ax.tick_params(axis='x', which='minor', rotation=45, length=3)

# Plot 1: Cleaned Intensity vs Time
fig, ax = plt.subplots(figsize=(24, 6))
ax.plot(df['Time [s]'], df['intensity'], label='Intensity')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Intensity')
ax.set_title('Monochannel Intensity vs Time')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
format_xaxis(ax)
ax.grid(True, which='both', alpha=0.7)

# Save first plot
plot1_path = os.path.join(RESULTS_DIR, 'monochannel_intensity_vs_time.png')
fig.tight_layout(pad=2)
fig.savefig(plot1_path, dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved plot 1 to {plot1_path}")

# Plot 2: Intensity vs Time with Peak Windows
fig, ax = plt.subplots(figsize=(24, 6))
ax.plot(df['Time [s]'], df['intensity'], label='Intensity')

# Highlight each peak window
for _, peak in w_df.iterrows():
    ax.axvspan(peak['start'], peak['end_trimmed'], color='#ff7f0e', alpha=0.3)

# Add legend entry for peak windows
peak_patch = Patch(facecolor='#ff7f0e', alpha=0.3, label='Peak Window')

ax.set_xlabel('Time [s]')
ax.set_ylabel('Intensity')
ax.set_title('Monochannel Intensity with Peak Windows vs Time')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Combine legends
handles, labels = ax.get_legend_handles_labels()
handles.append(peak_patch)
ax.legend(handles=handles, labels=[*labels, 'Peak Window'], loc='upper right')
format_xaxis(ax)
ax.grid(True, which='both', alpha=0.7)

# Save second plot
plot2_path = os.path.join(RESULTS_DIR, 'monochannel_intensity_with_peak_windows_vs_time.png')
fig.tight_layout(pad=2)
fig.savefig(plot2_path, dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved plot 2 to {plot2_path}")
