# -*- coding: utf-8 -*-
"""
revision_step4_summarize.py
===========================
STEP 4 of 5 -- Reviewer Comments 8 + 11 (lean 12-run design).

Reads the 12 PSO output folders from STEP 3 and produces 3 Excel tables:

    06_Final_Tables/
        all_runs_raw.xlsx
            12-row audit trail (one per PSO run)
        Table_S2_random_seed_sensitivity.xlsx
            mean +- SD of AUC/AP/Acc/F1/Brier across 10 seeds at thr=70
        Table_S3_threshold_sensitivity.xlsx
            AUC/AP/Acc/F1/Brier vs threshold at seed=42

Run:
    conda activate wildfire
    python src/revision_step4_summarize.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, brier_score_loss)

SEEDS_AT_THR70    = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
THRESHOLDS_AT_S42 = [100, 200]
THR_MAIN          = 70
SEED_MAIN         = 42

def pick_folder(t):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=t)
    r.destroy(); return p

def calc(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "oof_roc_auc":     float(roc_auc_score(y_true, y_prob)),
        "oof_pr_auc":      float(average_precision_score(y_true, y_prob)),
        "oof_brier":       float(brier_score_loss(y_true, y_prob)),
        "oof_accuracy":    float(accuracy_score(y_true, y_pred)),
        "oof_f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "oof_precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "oof_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "oof_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "oof_npv":         float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan"),
    }

print("=" * 60)
print("STEP 4 / 5: Collect 12 runs and summarize")
print("=" * 60)

print("\nDialog 1: pick 05_Model_Results")
res_root = pick_folder("STEP 4 dialog 1")
if not res_root: sys.exit(1)

print("\nDialog 2: pick 06_Final_Tables")
out_dir = pick_folder("STEP 4 dialog 2")
if not out_dir: sys.exit(1)
os.makedirs(out_dir, exist_ok=True)

combos = [(THR_MAIN, s) for s in SEEDS_AT_THR70]
for thr in THRESHOLDS_AT_S42:
    combos.append((thr, SEED_MAIN))

rows = []
missing = []
for thr, s in combos:
    run = os.path.join(res_root, f"thr{thr}", f"seed{s}")
    sj  = os.path.join(run, "metrics_summary.json")
    if not os.path.exists(sj):
        missing.append((thr, s)); continue
    with open(sj, "r", encoding="utf-8") as f:
        summary = json.load(f)
    row = {
        "threshold_ha": thr, "seed": s,
        "cv_mean_pr_auc":  summary.get("cv_mean",  {}).get("pr_auc",  np.nan),
        "cv_mean_roc_auc": summary.get("cv_mean",  {}).get("roc_auc", np.nan),
        "cv_oof_pr_auc":   summary.get("cv_oof",   {}).get("pr_auc",  np.nan),
        "cv_oof_roc_auc":  summary.get("cv_oof",   {}).get("roc_auc", np.nan),
        "final_holdout_pr_auc":  summary.get("final_holdout", {}).get("pr_auc",  np.nan),
        "final_holdout_roc_auc": summary.get("final_holdout", {}).get("roc_auc", np.nan),
    }
    cv_csv = os.path.join(run, "cv_metrics_10fold.csv")
    if os.path.exists(cv_csv):
        cdf = pd.read_csv(cv_csv)
        row["fold_pr_auc_sd"]  = float(cdf["pr_auc"].std())
        row["fold_roc_auc_sd"] = float(cdf["roc_auc"].std())
    oof = os.path.join(run, "cv_oof_predictions.csv")
    if os.path.exists(oof):
        d = pd.read_csv(oof).dropna(subset=["y_true", "y_pred_oof"])
        row.update(calc(d["y_true"].values, d["y_pred_oof"].values))
    rows.append(row)

if not rows:
    print("No results. Did STEP 3 finish anything?"); sys.exit(2)

full = pd.DataFrame(rows)
raw_path = os.path.join(out_dir, "all_runs_raw.xlsx")
full.to_excel(raw_path, index=False)
full.to_csv(raw_path.replace(".xlsx", ".csv"), index=False)
print(f"\n[saved] {raw_path}  ({len(full)} rows)")
if missing:
    print(f"[warn] {len(missing)} runs missing: {missing}")

METRIC_COLS = [c for c in [
    "cv_mean_roc_auc", "cv_mean_pr_auc",
    "oof_roc_auc", "oof_pr_auc", "oof_brier",
    "oof_accuracy", "oof_f1", "oof_precision",
    "oof_sensitivity", "oof_specificity", "oof_npv",
] if c in full.columns]

# Table S2: random-seed sensitivity (across 10 seeds at thr=70)
s2_src = full[full["threshold_ha"] == THR_MAIN].copy()
s2_rows = [{"metric": "n_seeds", "value": int(len(s2_src))}]
for m in METRIC_COLS:
    v = s2_src[m].dropna()
    if len(v):
        s2_rows.append({
            "metric": m,
            "mean":   float(v.mean()),
            "sd":     float(v.std()),
            "min":    float(v.min()),
            "max":    float(v.max()),
            "mean_sd_text": f"{v.mean():.4f} +/- {v.std():.4f}",
        })
s2 = pd.DataFrame(s2_rows)
s2_path = os.path.join(out_dir, "Table_S2_random_seed_sensitivity.xlsx")
s2.to_excel(s2_path, index=False)
s2.to_csv(s2_path.replace(".xlsx", ".csv"), index=False)
print(f"[saved] {s2_path}")

# Table S3: threshold sensitivity at seed 42
s3_src = full[full["seed"] == SEED_MAIN].copy().sort_values("threshold_ha")
keep = ["threshold_ha"] + METRIC_COLS
s3 = s3_src[keep].copy()
s3_path = os.path.join(out_dir, "Table_S3_threshold_sensitivity.xlsx")
s3.to_excel(s3_path, index=False)
s3.to_csv(s3_path.replace(".xlsx", ".csv"), index=False)
print(f"[saved] {s3_path}")

print("\n" + "=" * 60)
print("HEADLINE  Table S2  (random-seed sensitivity at thr=70 ha)")
print("=" * 60)
print(s2.to_string(index=False))

print("\n" + "=" * 60)
print("HEADLINE  Table S3  (threshold sensitivity at seed=42)")
print("=" * 60)
print(s3.to_string(index=False))

print("\nNext: STEP 5 (assumption checks):")
print("  python src/revision_step5_assumption_checks.py")

try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo("STEP 4 done",
        f"Wrote 3 Excel tables.\nTable S2 = {len(s2_src)} seeds at thr={THR_MAIN}.\n"
        f"Table S3 = {len(s3_src)} thresholds at seed={SEED_MAIN}.")
    a.destroy()
except Exception:
    pass
