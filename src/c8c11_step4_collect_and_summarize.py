# -*- coding: utf-8 -*-
"""
c8c11_step4_collect_and_summarize.py  (Option C, all-4-models version)
======================================================================
STEP 4 of 5 -- Reviewer Comments 8 + 11.

What this does:
  Walks every (MODEL, threshold, seed) folder under 05_Model_Results,
  reads metrics_summary.json + cv_oof_predictions.csv + best_params.json,
  computes threshold metrics on OOF, and writes:

      06_Final_Tables/all_model_sensitivity_results.xlsx
      06_Final_Tables/random_seed_sensitivity_summary_clean.xlsx
          mean +- SD per (model, threshold) across the 10 seeds
      06_Final_Tables/threshold_sensitivity_seed42.xlsx
          metric vs threshold at seed 42 (per model)
      06_Final_Tables/cross_model_comparison.xlsx
          headline AUC/F1 table: rows = models, cols = thresholds

Run:
    conda activate wildfire
    python src/c8c11_step4_collect_and_summarize.py
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
                              recall_score, confusion_matrix,
                              brier_score_loss)

MODELS = ["BiLSTM_PSO", "LSTM_PSO", "BiLSTM", "LSTM"]
THRESHOLDS_HA = [70, 100, 200]
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]

def pick_folder(title):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=title)
    r.destroy(); return p

def calc_metrics(y_true, y_prob, thr=0.5):
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
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

print("=" * 60)
print("STEP 4 / 5: Collect all-4-models results")
print("=" * 60)

print("\nDialog 1: pick 05_Model_Results")
results_root = pick_folder("STEP 4: pick 05_Model_Results")
if not results_root: sys.exit(1)

print("\nDialog 2: pick 06_Final_Tables")
final_dir = pick_folder("STEP 4: pick 06_Final_Tables")
if not final_dir: sys.exit(1)
os.makedirs(final_dir, exist_ok=True)

rows = []
missing = []
for model in MODELS:
    for thr in THRESHOLDS_HA:
        for s in SEEDS:
            run_dir = os.path.join(results_root, model, f"thr{thr}", f"seed{s}")
            sj = os.path.join(run_dir, "metrics_summary.json")
            if not os.path.exists(sj):
                missing.append((model, thr, s)); continue
            with open(sj, "r", encoding="utf-8") as f:
                summary = json.load(f)
            row = {
                "model": model, "threshold_ha": thr, "seed": s,
                "cv_mean_pr_auc":           summary.get("cv_mean", {}).get("pr_auc", np.nan),
                "cv_mean_roc_auc":          summary.get("cv_mean", {}).get("roc_auc", np.nan),
                "cv_oof_pr_auc_summary":    summary.get("cv_oof",  {}).get("pr_auc", np.nan),
                "cv_oof_roc_auc_summary":   summary.get("cv_oof",  {}).get("roc_auc", np.nan),
                "final_holdout_pr_auc":     summary.get("final_holdout", {}).get("pr_auc", np.nan),
                "final_holdout_roc_auc":    summary.get("final_holdout", {}).get("roc_auc", np.nan),
            }
            cv_csv = os.path.join(run_dir, "cv_metrics_10fold.csv")
            if os.path.exists(cv_csv):
                cdf = pd.read_csv(cv_csv)
                row["fold_pr_auc_sd"]  = float(cdf["pr_auc"].std())
                row["fold_roc_auc_sd"] = float(cdf["roc_auc"].std())
            oof_csv = os.path.join(run_dir, "cv_oof_predictions.csv")
            if os.path.exists(oof_csv):
                oof = pd.read_csv(oof_csv).dropna(subset=["y_true", "y_pred_oof"])
                row.update(calc_metrics(oof["y_true"].values,
                                        oof["y_pred_oof"].values, thr=0.5))
            rows.append(row)

if not rows:
    print("No results found. Did STEP 3 produce anything?"); sys.exit(2)

full = pd.DataFrame(rows)
all_path = os.path.join(final_dir, "all_model_sensitivity_results.xlsx")
full.to_excel(all_path, index=False)
full.to_csv(all_path.replace(".xlsx", ".csv"), index=False)
print(f"\n[saved] {all_path}  ({len(full)} rows)")
if missing:
    print(f"[warn] {len(missing)} run(s) missing (not yet completed in STEP 3)")
    by_model = {}
    for m, t, s in missing:
        by_model.setdefault(m, 0); by_model[m] += 1
    for m, n in by_model.items():
        print(f"    {m}: {n} missing")

METRIC_COLS = [c for c in [
    "cv_mean_roc_auc", "cv_mean_pr_auc",
    "oof_roc_auc", "oof_pr_auc", "oof_brier",
    "oof_accuracy", "oof_f1", "oof_precision",
    "oof_sensitivity", "oof_specificity", "oof_npv",
] if c in full.columns]

# ---- Clean summary: mean +- SD per (model, threshold) across 10 seeds ----
clean_rows = []
for (mdl, thr), g in full.groupby(["model", "threshold_ha"]):
    row = {"model": mdl, "threshold_ha": thr, "n_seeds": int(len(g))}
    for m in METRIC_COLS:
        v = g[m].dropna()
        mn = float(v.mean()) if len(v) else np.nan
        sd = float(v.std())  if len(v) else np.nan
        row[f"{m}_mean"]   = mn
        row[f"{m}_sd"]     = sd
        row[f"{m}_min"]    = float(v.min()) if len(v) else np.nan
        row[f"{m}_max"]    = float(v.max()) if len(v) else np.nan
        row[f"{m}_meanSD"] = f"{mn:.4f} +/- {sd:.4f}" if not np.isnan(mn) else "n/a"
    clean_rows.append(row)
clean = pd.DataFrame(clean_rows)
clean_path = os.path.join(final_dir, "random_seed_sensitivity_summary_clean.xlsx")
clean.to_excel(clean_path, index=False)
clean.to_csv(clean_path.replace(".xlsx", ".csv"), index=False)
print(f"[saved] {clean_path}")

# ---- Threshold sensitivity at seed 42 (one row per (model, threshold)) ----
seed42 = full[full["seed"] == 42].copy()
if not seed42.empty:
    t_path = os.path.join(final_dir, "threshold_sensitivity_seed42.xlsx")
    seed42.to_excel(t_path, index=False)
    seed42.to_csv(t_path.replace(".xlsx", ".csv"), index=False)
    print(f"[saved] {t_path}")

# ---- Cross-model comparison (headline table: rows=model, cols=threshold) ----
if not clean.empty and "oof_roc_auc_meanSD" in clean.columns:
    pivot_auc = clean.pivot(index="model", columns="threshold_ha",
                            values="oof_roc_auc_meanSD")
    pivot_f1  = clean.pivot(index="model", columns="threshold_ha",
                            values="oof_f1_meanSD")
    cmp_path = os.path.join(final_dir, "cross_model_comparison.xlsx")
    with pd.ExcelWriter(cmp_path) as xw:
        pivot_auc.to_excel(xw, sheet_name="AUC_meanSD")
        pivot_f1.to_excel(xw,  sheet_name="F1_meanSD")
    print(f"[saved] {cmp_path}")

# ---- Headline print ----
print("\n" + "=" * 60)
print("HEADLINE: across-seed mean +/- SD per (model, threshold)")
print("=" * 60)
print(clean[["model", "threshold_ha", "n_seeds",
             "oof_roc_auc_meanSD", "oof_pr_auc_meanSD",
             "oof_accuracy_meanSD", "oof_f1_meanSD"]].to_string(index=False))

if not seed42.empty:
    print("\nThreshold-only effect at seed 42:")
    print(seed42[["model", "threshold_ha", "oof_roc_auc", "oof_pr_auc",
                  "oof_accuracy", "oof_f1"]].to_string(index=False))

print("\nNext: run STEP 5 (assumption checks):")
print("  python src/c8c11_step5_assumption_checks.py")

try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo("STEP 4 done",
        f"Wrote summary tables for {len(full)} runs across {full['model'].nunique()} models.")
    a.destroy()
except Exception:
    pass
