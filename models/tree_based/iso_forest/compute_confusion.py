#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths (all relative to this script’s location)
# --------------------------------------------------------------------------- #
SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'
SUMMARY_CSV = RESULTS_DIR / '97.39_evaluation_summary_mono_iso_forest.csv'
GT_CSV      = SCRIPT_DIR.parents[2] / 'data_files' / 'mono_channel_and_peak_period' / 'events20240515_e_merged_0.0003.csv'

# --------------------------------------------------------------------------- #
# Load evaluation summary and new ground-truth
# --------------------------------------------------------------------------- #
summary = pd.read_csv(SUMMARY_CSV)

# Ground‑truth file has columns:
#   - Peak_start
#   - Peak_end         (original)
#   - Trimmed_Peak_end (use this as the true end)
gt_raw = pd.read_csv(GT_CSV)
# Keep only the start and the trimmed end
gt = gt_raw[['Peak_start', 'Trimmed_Peak_end']].rename(
    columns={'Trimmed_Peak_end': 'Peak_end'}
)

# --------------------------------------------------------------------------- #
# Count TP, FP, FN
# --------------------------------------------------------------------------- #
# True Positives: predicted windows that contained the GT peak
TP = int(summary['has_gt_peak'].sum())

# False Positives: predicted windows that did not contain a GT peak
FP = int(len(summary) - TP)

# Total number of GT events
N_gt = len(gt)

# False Negatives: GT peaks not captured by any prediction
FN = N_gt - TP


# --------------------------------------------------------------------------- #
# Compute precision, recall, F1
# --------------------------------------------------------------------------- #
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1_score  = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

# --------------------------------------------------------------------------- #
# Print the confusion‐style matrix and metrics
# --------------------------------------------------------------------------- #
print("Confusion‐style counts:")
print(f"  TP: {TP}")
print(f"  FP: {FP}")
print(f"  FN: {FN}")

print("Derived metrics:")
print(f"  Precision: {precision:.6f}")
print(f"  Recall:    {recall:.6f}")
print(f"  F1 score:  {f1_score:.6f}")
