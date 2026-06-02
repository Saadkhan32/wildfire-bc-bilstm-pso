# -*- coding: utf-8 -*-
"""
c8_step5_assumption_checks.py
=============================
STEP 5 of 5 -- Reviewer Comment 8, sub-question 5: "assumptions met and tested?"

What this does:
  Runs three statistical checks on the seed-42 BiLSTM output:
    1. VIF (multicollinearity) on the 15 RFE-selected predictors
       -> tables/T_vif_predictors.csv
    2. Calibration (predicted probability vs observed fire rate)
       -> figs/Fig_S_calibration.png + tables/T_calibration_bins.csv
    3. Residual Moran's I (spatial independence of OOF errors)
       -> tables/T_residual_morans_i.csv

Why this matters for the reviewer:
  Huettmann asked 'assumptions met and tested?'. These three checks
  are the canonical assumption tests for a spatial ML classifier.

Pre-reqs:
    pip install statsmodels libpysal esda matplotlib

How to run:
    conda activate wildfire
    cd C:\\Users\\saadz\\Documents\\wildfire-bc-bilstm-pso
    python src\\c8_step5_assumption_checks.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

def pick_file(title, ft):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=title, filetypes=ft)
    r.destroy(); return p

def pick_folder(title):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=title)
    r.destroy(); return p

print("=" * 60)
print("STEP 5 / 5: Assumption checks (VIF, calibration, residual Moran's I)")
print("=" * 60)
print()

# Dialog 1 -- pick training_seed_42.csv
print("Dialog 1 of 3: Pick training_seed_42.csv from Wildfire_C8\\02_Seed_Tables.")
seed_csv = pick_file(
    "STEP 5 dialog 1: pick training_seed_42.csv",
    [("CSV files", "*.csv")],
)
if not seed_csv:
    print("CANCELLED."); sys.exit(1)
print(f"  seed CSV: {seed_csv}")

# Dialog 2 -- pick seed_42/BILSTM folder
print("\nDialog 2 of 3: Pick the seed_42\\BILSTM model folder")
print("(under Wildfire_C8\\03_Model_Results\\seed_42\\BILSTM\\)")
model_dir = pick_folder("STEP 5 dialog 2: pick seed_42\\BILSTM folder")
if not model_dir:
    print("CANCELLED."); sys.exit(1)
print(f"  model dir: {model_dir}")

# Dialog 3 -- pick the project repo root (so outputs go to tables/ + figs/)
print("\nDialog 3 of 3: Pick your project repo root")
print("(probably C:\\Users\\saadz\\Documents\\wildfire-bc-bilstm-pso)")
repo_root = pick_folder("STEP 5 dialog 3: pick repo root")
if not repo_root:
    print("CANCELLED."); sys.exit(1)
print(f"  repo: {repo_root}")

TABLES_DIR = os.path.join(repo_root, "tables")
FIGS_DIR   = os.path.join(repo_root, "figs")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

HOLDOUT_CSV    = os.path.join(model_dir, "holdout_predictions.csv")
SELECTED_FEATS = os.path.join(model_dir, "selected_features_final.csv")

for p in (HOLDOUT_CSV, SELECTED_FEATS):
    if not os.path.exists(p):
        print(f"\nERROR: missing {p}")
        print("Run STEP 4 first.")
        sys.exit(2)

# ============================================================
# Check 1: VIF
# ============================================================
print("\n========== Check 1 / 3: VIF (multicollinearity) ==========")
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: pip install statsmodels"); sys.exit(3)

df = pd.read_csv(seed_csv)
selected = pd.read_csv(SELECTED_FEATS)["selected_feature"].tolist()

def strip_prefix(s):
    for pre in ("num__", "cat__"):
        if s.startswith(pre): return s[len(pre):]
    return s

raw_names = []
for s in selected:
    b = strip_prefix(s)
    if b.lower().startswith("lulc"):
        b = b.split("_")[0]
    raw_names.append(b)
raw_names = list(dict.fromkeys(raw_names))

available = [c for c in raw_names if c in df.columns]
print(f"  {len(available)} of {len(raw_names)} selected features available as raw columns.")

X = df[available].select_dtypes(include=[np.number]).dropna().copy()
print(f"  VIF input: {X.shape[0]} rows, {X.shape[1]} predictors")
X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

vif_rows = []
for i, col in enumerate(X_std.columns):
    try:
        v = variance_inflation_factor(X_std.values, i)
    except Exception:
        v = float("nan")
    vif_rows.append({"predictor": col, "VIF": round(float(v), 3)})

vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
vif_df["flag"] = vif_df["VIF"].apply(
    lambda v: "N/A" if np.isnan(v) else
              ("OK (<5)" if v < 5 else
               ("Watch (5-10)" if v < 10 else "Problem (>=10)"))
)
vif_out = os.path.join(TABLES_DIR, "T_vif_predictors.csv")
vif_df.to_csv(vif_out, index=False)
print(f"  wrote {vif_out}")
print(vif_df.to_string(index=False))
max_vif = vif_df["VIF"].max()
n_over_5 = int((vif_df["VIF"] >= 5).sum())
print(f"  >>> max VIF = {max_vif:.2f};  predictors with VIF >= 5: {n_over_5}")

# ============================================================
# Check 2: Calibration
# ============================================================
print("\n========== Check 2 / 3: Calibration ==========")
try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: pip install matplotlib"); sys.exit(3)

pred = pd.read_csv(HOLDOUT_CSV)
y_true = pred["y_true"].astype(int).values
y_prob = pred["y_prob"].astype(float).values
brier = brier_score_loss(y_true, y_prob)
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

cal_df = pd.DataFrame({"mean_predicted_prob": mean_pred,
                       "observed_fire_rate": frac_pos})
cal_out = os.path.join(TABLES_DIR, "T_calibration_bins.csv")
cal_df.to_csv(cal_out, index=False)

fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
ax.plot(mean_pred, frac_pos, "o-", ms=8, lw=1.6, color="#c0392b",
        label="BiLSTM (seed 42)")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed fire rate")
ax.set_title(f"Reliability diagram\nBrier score = {brier:.4f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
cal_png = os.path.join(FIGS_DIR, "Fig_S_calibration.png")
plt.savefig(cal_png, dpi=300, bbox_inches="tight")
plt.savefig(cal_png.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"  wrote {cal_out}")
print(f"  wrote {cal_png}")
print(f"  >>> Brier score = {brier:.4f}")

# ============================================================
# Check 3: Residual Moran's I
# ============================================================
print("\n========== Check 3 / 3: Residual Moran's I ==========")
try:
    from libpysal.weights import KNN
    from esda.moran import Moran
except ImportError:
    print("ERROR: pip install libpysal esda"); sys.exit(3)

from sklearn.model_selection import train_test_split

y_all = pd.to_numeric(df["Status"], errors="coerce").fillna(0).astype(int).values

_, te_idx, _, _ = train_test_split(np.arange(len(y_all)), y_all,
                                    test_size=0.30, random_state=42, stratify=y_all)
if len(te_idx) != len(pred):
    print(f"  WARN: test-set size mismatch ({len(te_idx)} vs {len(pred)}).")
n = min(len(te_idx), len(pred))
test_idx = te_idx[:n]
coords = df.loc[test_idx, ["Longitude", "Latitude"]].dropna().values
residuals = (y_true[:n] - y_prob[:n])[:len(coords)]

if len(coords) < 30:
    print(f"  ERROR: only {len(coords)} coords with lat/lon; cannot compute.")
    sys.exit(4)

k = min(8, max(2, len(coords) // 50))
w = KNN.from_array(coords, k=k); w.transform = "r"
mi = Moran(residuals, w, permutations=999)

mi_row = {
    "model":            "BiLSTM (seed 42)",
    "n_test_residuals": int(len(residuals)),
    "knn_k":            int(k),
    "morans_I":         round(float(mi.I), 4),
    "expected_I":       round(float(mi.EI), 4),
    "z_score":          round(float(mi.z_sim), 3),
    "p_value":          round(float(mi.p_sim), 4),
    "interpretation":   ("Independent (low spatial autocorrelation)"
                         if mi.p_sim > 0.05
                         else "Spatial autocorrelation detected"),
}
mi_df = pd.DataFrame([mi_row])
mi_out = os.path.join(TABLES_DIR, "T_residual_morans_i.csv")
mi_df.to_csv(mi_out, index=False)
print(f"  wrote {mi_out}")
print(mi_df.to_string(index=False))

# ============================================================
# Summary JSON for rebuttal-letter pickup
# ============================================================
summary = {
    "comment": "Reviewer Comment 8 sub-question 5: assumptions met and tested",
    "seed_audited": 42, "model_audited": "BiLSTM",
    "vif": {"max": float(max_vif),
            "n_ge_5": n_over_5,
            "table": vif_out},
    "calibration": {"brier_score": float(brier),
                    "figure": cal_png, "bins_table": cal_out},
    "residual_morans_i": {"I": float(mi.I), "expected": float(mi.EI),
                          "z": float(mi.z_sim), "p": float(mi.p_sim),
                          "knn_k": int(k), "n_residuals": int(len(residuals)),
                          "interpretation": mi_row["interpretation"]},
}
summary_out = os.path.join(TABLES_DIR, "T_c8_assumption_summary.json")
with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("STEP 5 DONE.")
print("=" * 60)
print(f"  T_vif_predictors.csv         max VIF = {max_vif:.2f},  "
      f"VIF>=5: {n_over_5}")
print(f"  T_calibration_bins.csv       Brier = {brier:.4f}")
print(f"  Fig_S_calibration.png        reliability diagram")
print(f"  T_residual_morans_i.csv      Moran's I = {mi.I:.4f}, p = {mi.p_sim:.4f}")
print(f"  T_c8_assumption_summary.json  (rebuttal-letter pickup)")
print()
print("All five Comment-8 deliverables are now in your repo:")
print(f"  - {TABLES_DIR}\\T_sample_lineage.json")
print(f"  - {TABLES_DIR}\\T_reproducibility_audit.csv")
print(f"  - {TABLES_DIR}\\T_vif_predictors.csv")
print(f"  - {TABLES_DIR}\\T_calibration_bins.csv")
print(f"  - {TABLES_DIR}\\T_residual_morans_i.csv")
print(f"  - {TABLES_DIR}\\T_c8_assumption_summary.json")
print(f"  - <C8 output>\\random_seed_sensitivity_clean_summary.xlsx")
print()
print("Send these to me (or commit + tell me to pull) and I will write")
print("the v9 manuscript tracked changes + rebuttal letter section.")

try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo(
        "STEP 5 complete",
        f"All Comment-8 assumption checks done.\n\nMax VIF: {max_vif:.2f}\n"
        f"Brier: {brier:.4f}\nMoran's I: {mi.I:.4f}  (p = {mi.p_sim:.4f})\n\n"
        "Hand the outputs to Claude for the manuscript and rebuttal.",
    )
    a.destroy()
except Exception:
    pass
