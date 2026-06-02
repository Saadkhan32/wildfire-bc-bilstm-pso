# -*- coding: utf-8 -*-
"""
revision_step4_summarize_all_models.py
======================================
STEP 4 of 5 (Option C lean = 4 models, 12 combinations)

Collects 48 PSO/non-PSO outputs and builds 4 Excel tables:

    06_Final_Tables/
        all_runs_raw.xlsx                       (48-row audit trail)
        Table_S2_random_seed_sensitivity_by_model.xlsx
            mean +/- SD per model at thr=70 across the 10 seeds
        Table_S3_threshold_sensitivity_by_model.xlsx
            metric vs threshold at seed 42, per model
        Table_S4_cross_model_comparison.xlsx
            pivot: rows = model, cols = threshold

Run:
    conda activate wildfire
    python src/revision_step4_summarize_all_models.py
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

MODELS = ["BiLSTM_PSO", "LSTM_PSO", "BiLSTM", "LSTM"]
SEEDS_AT_THR70    = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
THRESHOLDS_AT_S42 = [100, 200]
THR_MAIN          = 70
SEED_MAIN         = 42

def pick_folder(t):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=t); r.destroy(); return p

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
print("STEP 4 / 5: Collect 48 runs across all 4 models")
print("=" * 60)

print("\nDialog 1: pick 05_Model_Results")
res = pick_folder("STEP 4 dialog 1")
if not res: sys.exit(1)

print("\nDialog 2: pick 06_Final_Tables")
out_dir = pick_folder("STEP 4 dialog 2")
if not out_dir: sys.exit(1)
os.makedirs(out_dir, exist_ok=True)

combos = [(THR_MAIN, s) for s in SEEDS_AT_THR70]
for thr in THRESHOLDS_AT_S42:
    combos.append((thr, SEED_MAIN))

rows = []; missing = []
for model in MODELS:
    for thr, s in combos:
        run = os.path.join(res, model, f"thr{thr}", f"seed{s}")
        sj = os.path.join(run, "metrics_summary.json")
        if not os.path.exists(sj):
            missing.append((model, thr, s)); continue
        with open(sj, "r", encoding="utf-8") as f:
            summary = json.load(f)
        row = {
            "model": model, "threshold_ha": thr, "seed": s,
            "cv_mean_pr_auc":  summary.get("cv_mean",  {}).get("pr_auc",  np.nan),
            "cv_mean_roc_auc": summary.get("cv_mean",  {}).get("roc_auc", np.nan),
            "cv_oof_pr_auc":   summary.get("cv_oof",   {}).get("pr_auc",  np.nan),
            "cv_oof_roc_auc":  summary.get("cv_oof",   {}).get("roc_auc", np.nan),
        }
        cv = os.path.join(run, "cv_metrics_10fold.csv")
        if os.path.exists(cv):
            cdf = pd.read_csv(cv)
            row["fold_pr_auc_sd"]  = float(cdf["pr_auc"].std())
            row["fold_roc_auc_sd"] = float(cdf["roc_auc"].std())
        oof = os.path.join(run, "cv_oof_predictions.csv")
        if os.path.exists(oof):
            d = pd.read_csv(oof).dropna(subset=["y_true", "y_pred_oof"])
            row.update(calc(d["y_true"].values, d["y_pred_oof"].values))
        rows.append(row)

if not rows:
    print("No results found. Did STEP 3 produce anything?"); sys.exit(2)

full = pd.DataFrame(rows)
raw_path = os.path.join(out_dir, "all_runs_raw.xlsx")
full.to_excel(raw_path, index=False)
full.to_csv(raw_path.replace(".xlsx", ".csv"), index=False)
print(f"\n[saved] {raw_path}  ({len(full)} rows)")
if missing:
    by = {}
    for m, t, s in missing: by[m] = by.get(m, 0) + 1
    print(f"[warn] {len(missing)} runs missing:")
    for m, n in by.items():
        print(f"    {m}: {n}")

METRICS = [c for c in [
    "cv_mean_roc_auc", "cv_mean_pr_auc",
    "oof_roc_auc", "oof_pr_auc", "oof_brier",
    "oof_accuracy", "oof_f1", "oof_precision",
    "oof_sensitivity", "oof_specificity", "oof_npv",
] if c in full.columns]

# Table S2: random-seed sensitivity by model (at thr=70)
s2_src = full[full["threshold_ha"] == THR_MAIN].copy()
s2_rows = []
for mdl, g in s2_src.groupby("model"):
    row = {"model": mdl, "n_seeds": int(len(g))}
    for m in METRICS:
        v = g[m].dropna()
        mn = float(v.mean()) if len(v) else np.nan
        sd = float(v.std())  if len(v) else np.nan
        row[f"{m}_mean"]    = mn
        row[f"{m}_sd"]      = sd
        row[f"{m}_min"]     = float(v.min()) if len(v) else np.nan
        row[f"{m}_max"]     = float(v.max()) if len(v) else np.nan
        row[f"{m}_meanSD"]  = f"{mn:.4f} +/- {sd:.4f}" if not np.isnan(mn) else "n/a"
    s2_rows.append(row)
s2 = pd.DataFrame(s2_rows)
s2_path = os.path.join(out_dir, "Table_S2_random_seed_sensitivity_by_model.xlsx")
s2.to_excel(s2_path, index=False)
s2.to_csv(s2_path.replace(".xlsx", ".csv"), index=False)
print(f"[saved] {s2_path}")

# Table S3: threshold sensitivity by model (at seed=42)
s3_src = full[full["seed"] == SEED_MAIN].copy().sort_values(["model", "threshold_ha"])
s3 = s3_src[["model", "threshold_ha"] + METRICS].copy()
s3_path = os.path.join(out_dir, "Table_S3_threshold_sensitivity_by_model.xlsx")
s3.to_excel(s3_path, index=False)
s3.to_csv(s3_path.replace(".xlsx", ".csv"), index=False)
print(f"[saved] {s3_path}")

# Table S4: cross-model comparison (pivots)
if not s2.empty and "oof_roc_auc_meanSD" in s2.columns:
    # AUC mean+/-SD across seeds at thr=70
    auc_s2 = s2[["model", "oof_roc_auc_meanSD"]].rename(
        columns={"oof_roc_auc_meanSD": "AUC_thr70_seedsensitivity"})
    f1_s2 = s2[["model", "oof_f1_meanSD"]].rename(
        columns={"oof_f1_meanSD": "F1_thr70_seedsensitivity"})
    # AUC at each threshold at seed=42
    if not s3.empty:
        piv_auc = s3.pivot_table(index="model", columns="threshold_ha",
                                  values="oof_roc_auc")
        piv_auc.columns = [f"AUC_seed42_thr{c}" for c in piv_auc.columns]
        piv_f1  = s3.pivot_table(index="model", columns="threshold_ha",
                                  values="oof_f1")
        piv_f1.columns = [f"F1_seed42_thr{c}" for c in piv_f1.columns]
        cmp = auc_s2.merge(f1_s2, on="model") \
                    .merge(piv_auc.reset_index(), on="model") \
                    .merge(piv_f1.reset_index(),  on="model")
        cmp_path = os.path.join(out_dir, "Table_S4_cross_model_comparison.xlsx")
        cmp.to_excel(cmp_path, index=False)
        cmp.to_csv(cmp_path.replace(".xlsx", ".csv"), index=False)
        print(f"[saved] {cmp_path}")

print("\n" + "=" * 60)
print("HEADLINE  Table S2  (random-seed sensitivity by model at thr=70)")
print("=" * 60)
print(s2[["model", "n_seeds",
          "oof_roc_auc_meanSD", "oof_pr_auc_meanSD",
          "oof_accuracy_meanSD", "oof_f1_meanSD"]].to_string(index=False))

print("\n" + "=" * 60)
print("HEADLINE  Table S3  (threshold sensitivity by model at seed=42)")
print("=" * 60)
print(s3.to_string(index=False))

print("\nNext: STEP 5 (assumption checks):")
print("  python src/revision_step5_assumption_checks.py")

try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo("STEP 4 done",
        f"{len(full)} rows.  {s2.shape[0]} models in summary.")
    a.destroy()
except Exception: pass
