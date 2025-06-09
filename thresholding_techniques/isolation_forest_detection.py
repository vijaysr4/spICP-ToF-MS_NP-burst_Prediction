import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

# ensure output directories exist
os.makedirs('results', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)

# load the data
df = pd.read_csv('../data/NPs_BHVO_Oct23_full.csv')

# list of all ion channels to include
ION_CHANNELS = [
    '23Na','24Mg','26Mg','27Al','28Si','29Si',
    '39K','40Ca','48Ti','54Fe','55Mn','56Fe','60Ni'
]

# prepare the feature matrix
X = df[ION_CHANNELS].to_numpy(dtype=float)

# train the isolation forest
clf = IsolationForest(contamination=0.001, random_state=42)
clf.fit(X)

# compute anomaly scores with a progress bar
n_samples = X.shape[0]
scores = np.empty(n_samples)
for i in tqdm(range(n_samples), desc="Scoring samples"):
    # decision_function returns an array of length 1 for a single sample
    scores[i] = clf.decision_function(X[i].reshape(1, -1))[0]

# predict which samples are anomalies (–1)
anomaly_mask = clf.predict(X) == -1

# extract the corresponding timestamps and scores
times = df['Time (ms)'].to_numpy()
burst_times = times[anomaly_mask]
burst_scores = scores[anomaly_mask]

# create plot
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(times, scores, color='steelblue', linewidth=1, alpha=0.8, label='Anomaly score')
ax.axhline(0, color='crimson', linestyle='--', linewidth=1.5, label='Decision threshold')
ax.scatter(burst_times, burst_scores, color='crimson', s=20, edgecolors='k', label='Detected bursts')
ax.set_title('Isolation Forest Burst Detection', fontsize=16, weight='bold')
ax.set_xlabel('Time (ms)', fontsize=14)
ax.set_ylabel('Anomaly Score', fontsize=14)
ax.legend(frameon=True, fontsize=12)
plt.tight_layout()

# save the figure
fig.savefig('results/isolation_forest_bursts.png', dpi=200)
plt.close(fig)

# save detected burst times
pd.Series(burst_times, name='Time (ms)') \
  .to_csv('processed_data/isolation_forest_bursts.csv', index=False, header=True)
