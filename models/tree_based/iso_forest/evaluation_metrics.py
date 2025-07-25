#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Paths (all relative to this script’s location)
# --------------------------------------------------------------------------- #
SCRIPT_DIR  = Path(__file__).resolve().parent
NANO_ROOT   = SCRIPT_DIR.parents[2]

DATA_DIR    = NANO_ROOT / 'data_files' / 'mono_channel_and_peak_period'
GT_CSV      = DATA_DIR / 'events20240515_e_merged_0.0003.csv'

RESULTS_DIR = SCRIPT_DIR / 'results'
PRED_CSV    = RESULTS_DIR / 'peak_windows_iso_forest_mono_channel_anomaly_97.39.csv'
SUMMARY_CSV = RESULTS_DIR / '97.39_evaluation_summary_mono_iso_forest.csv'
# --------------------------------------------------------------------------- #

def interval_metrics(g_s: float, g_e: float,
                     p_s: float, p_e: float) -> dict[str, float]:
    '''Compute overlap, durations, and IoU.'''
    overlap = max(0.0, min(p_e, g_e) - max(p_s, g_s))
    dur_gt   = g_e - g_s
    dur_pred = p_e - p_s
    union = dur_gt + dur_pred - overlap
    iou   = overlap / union if union else 0.0
    return {
        'overlap':  overlap,
        'dur_gt':   dur_gt,
        'dur_pred': dur_pred,
        'iou':      iou,
    }


def find_peak(df: pd.DataFrame, start: float, end: float
              ) -> tuple[float | None, float | None]:
    '''Return (time, intensity) of the maximum point within [start, end].'''
    window = df[(df['Time [s]'] >= start) & (df['Time [s]'] <= end)]
    if window.empty:
        return None, None
    row = window.loc[window['intensity'].idxmax()]
    return float(row['Time [s]']), float(row['intensity'])


def greedy_match(pred: pd.DataFrame, gt: pd.DataFrame
                 ) -> dict[int, int]:
    '''Greedy one-to-one matching: returns map from pred index to gt index.'''
    mapping: dict[int,int] = {}
    used_gt: set[int] = set()
    for p_idx, p in tqdm(pred.sort_values('Peak_start').iterrows(),
                         desc='Matching preds→GT', total=len(pred), unit='pred'):
        best_iou, best_g = 0.0, None
        for g_idx, g in gt.iterrows():
            if g_idx in used_gt:
                continue
            ov = max(0.0, min(p.Peak_end, g.Peak_end) - max(p.Peak_start, g.Peak_start))
            un = (p.Peak_end - p.Peak_start) + (g.Peak_end - g.Peak_start) - ov
            iou = ov / un if un else 0.0
            if iou > best_iou:
                best_iou, best_g = iou, g_idx
        if best_g is not None:
            mapping[p_idx] = best_g
            used_gt.add(best_g)
    return mapping


def main() -> None:
    # Load data
    ts_df   = pd.read_csv(DATA_DIR / 'cleaned_data.csv')
    raw_gt  = pd.read_csv(GT_CSV)
    raw_gt  = raw_gt.rename(columns={'Peak_end':'orig_end', 'Trimmed_Peak_end':'Peak_end'})
    gt_df   = raw_gt[['Peak_start','Peak_end']].astype(float)
    pred_df = pd.read_csv(PRED_CSV).astype(float)

    # Perform matching
    match_map = greedy_match(pred_df, gt_df)

    records: list[dict] = []
    # Loop over all predictions
    for p_idx, p in tqdm(pred_df.iterrows(), desc='Evaluating all preds', total=len(pred_df), unit='pred'):
        rec: dict[str,float] = {}
        # Signed start-time error initialized
        rec['start_err'] = np.nan
        rec['start_err_abs'] = np.nan
        # If matched, compute metrics
        if p_idx in match_map:
            g = gt_df.loc[match_map[p_idx]]
            # interval metrics
            rec.update(interval_metrics(g.Peak_start, g.Peak_end, p.Peak_start, p.Peak_end))
            # find true peak and flag
            gt_peak_t,_ = find_peak(ts_df, g.Peak_start, g.Peak_end)
            rec['has_gt_peak'] = int(gt_peak_t is not None and p.Peak_start <= gt_peak_t <= p.Peak_end)
            # timing error
            err = p.Peak_start - g.Peak_start
            rec['start_err'] = err
            rec['start_err_abs'] = abs(err)
            # record GT window
            rec['gt_window_start'] = g.Peak_start
            rec['gt_window_end']   = g.Peak_end
        else:
            # For unmatched preds, set defaults
            rec.update({
                'overlap': 0.0,
                'dur_gt': np.nan,
                'dur_pred': p.Peak_end-p.Peak_start,
                'iou': 0.0,
                'has_gt_peak': 0,
                'gt_window_start': np.nan,
                'gt_window_end': np.nan,
            })
        # Always include pred window
        rec['pred_window_start'] = p.Peak_start
        rec['pred_window_end']   = p.Peak_end
        records.append(rec)

    summary = pd.DataFrame(records)
    # Compute macro-averages across all predictions
    agg_cols = ['iou','has_gt_peak','start_err','start_err_abs']
    macro = summary[agg_cols].mean(numeric_only=True)

    # Print macro metrics
    print('\n=== Macro‑averaged metrics over all preds ===')
    for k,v in macro.items():
        print(f"{k:20s}: {v:.6f}")

    # Save full summary
    SUMMARY_CSV.parent.mkdir(exist_ok=True, parents=True)
    summary.to_csv(SUMMARY_CSV, index=False, float_format='%.6f')
    print(f"\nSaved detailed summary → {SUMMARY_CSV}")

if __name__=='__main__':
    main()
