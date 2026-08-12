import os
import sys
import json
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
def pick_file(t, ft):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=t, filetypes=ft)
    r.destroy(); return p
def pick_folder(t):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=t)
    r.destroy(); return p
print("=" * 60)
print("STEP 5 / 5: assumption checks (VIF + calibration + Moran I)")
print("=" * 60)
print("\nDialog 1: pick training_thr70_seed42.csv")
seed_csv = pick_file("STEP 5 dialog 1", [("CSV", "*.csv")])
if not seed_csv: sys.exit(1)
print("\nDialog 2: pick the 05_Model_Results/thr70/seed42 folder")
model_dir = pick_folder("STEP 5 dialog 2")
if not model_dir: sys.exit(1)
print("\nDialog 3: pick repo root (wildfire-bc-bilstm-pso)")
repo_root = pick_folder("STEP 5 dialog 3")
if not repo_root: sys.exit(1)
TABLES_DIR = os.path.join(repo_root, "tables")
FIGS_DIR   = os.path.join(repo_root, "figs")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,   exist_ok=True)
OOF_CSV   = os.path.join(model_dir, "cv_oof_predictions.csv")
SEL_FEATS = os.path.join(model_dir, "selected_features_final.csv")
for p in (OOF_CSV, SEL_FEATS):
    if not os.path.exists(p):
        print(f"ERROR: missing {p}"); sys.exit(2)
print("\n========== Check 1 / 3: VIF ==========")
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(seed_csv)
sel = pd.read_csv(SEL_FEATS)["selected_feature"].tolist()
def strip_prefix(s):
    for pre in ("num__", "cat__"):
        if s.startswith(pre): return s[len(pre):]
    return s
raw = []
for s in sel:
    b = strip_prefix(s)
    if b.lower().startswith("lulc"): b = b.split("_")[0]
    raw.append(b)
raw = list(dict.fromkeys(raw))
avail = [c for c in raw if c in df.columns]
X = df[avail].select_dtypes(include=[np.number]).dropna()
Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
vif_rows = []
for i, c in enumerate(Xs.columns):
    try:
        v = float(variance_inflation_factor(Xs.values, i))
    except Exception:
        v = float("nan")
    vif_rows.append({"predictor": c, "VIF": round(v, 3)})
vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
def vif_flag(v):
    if np.isnan(v): return "N/A"
    if v < 5:  return "OK"
    if v < 10: return "Watch"
    return "Problem"
vif_df["flag"] = vif_df["VIF"].apply(vif_flag)
vif_out = os.path.join(TABLES_DIR, "Table_S4_VIF.csv")
vif_df.to_csv(vif_out, index=False)
print(vif_df.to_string(index=False))
max_vif = float(vif_df["VIF"].max())
n_over_5 = int((vif_df["VIF"] >= 5).sum())
print(f"  >>> max VIF = {max_vif:.2f}; predictors VIF>=5: {n_over_5}")
print("\n========== Check 2 / 3: Calibration ==========")
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
oof = pd.read_csv(OOF_CSV).dropna(subset=["y_true", "y_pred_oof"])
y_true = oof["y_true"].astype(int).values
y_prob = oof["y_pred_oof"].astype(float).values
brier = float(brier_score_loss(y_true, y_prob))
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
cal_df = pd.DataFrame({"mean_predicted_prob": mean_pred,
                       "observed_fire_rate": frac_pos})
cal_out = os.path.join(TABLES_DIR, "Table_S5_calibration_bins.csv")
cal_df.to_csv(cal_out, index=False)
fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
ax.plot(mean_pred, frac_pos, "o-", ms=8, lw=1.6, color="#c0392b",
        label="BiLSTM-PSO (thr70 seed42)")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed fire rate")
ax.set_title(f"Reliability diagram (Brier = {brier:.4f})")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
cal_png = os.path.join(FIGS_DIR, "Fig_S_calibration.png")
plt.savefig(cal_png, dpi=300, bbox_inches="tight")
plt.savefig(cal_png.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"  Brier = {brier:.4f}")
print("\n========== Check 3 / 3: Residual Moran I ==========")
from libpysal.weights import KNN
from esda.moran import Moran
if "index" in oof.columns and len(oof) <= len(df):
    coords = df.loc[oof["index"].values, ["Longitude", "Latitude"]].dropna().values
else:
    coords = df[["Longitude", "Latitude"]].dropna().values
    if len(coords) > len(oof): coords = coords[:len(oof)]
resid = (y_true - y_prob)[:len(coords)]
if len(coords) < 30:
    print(f"  too few coords ({len(coords)})"); sys.exit(4)
k = min(8, max(2, len(coords) // 50))
w = KNN.from_array(coords, k=k); w.transform = "r"
mi = Moran(resid, w, permutations=999)
interp = "Independent" if mi.p_sim > 0.05 else "Spatial autocorrelation"
mi_row = {
    "model":          "BiLSTM-PSO (thr70 seed42)",
    "n_residuals":    int(len(resid)),
    "knn_k":          int(k),
    "morans_I":       round(float(mi.I), 4),
    "expected_I":     round(float(mi.EI), 4),
    "z_score":        round(float(mi.z_sim), 3),
    "p_value":        round(float(mi.p_sim), 4),
    "interpretation": interp,
}
mi_df = pd.DataFrame([mi_row])
mi_out = os.path.join(TABLES_DIR, "Table_S6_residual_Moran.csv")
mi_df.to_csv(mi_out, index=False)
print(mi_df.to_string(index=False))
summary = {
    "comment": "",
    "seed_audited": 42, "threshold_audited_ha": 70,
    "model_audited": "BiLSTM-PSO",
    "vif": {"max": max_vif, "n_ge_5": n_over_5, "table": vif_out},
    "calibration": {"brier_score": brier, "figure": cal_png, "bins_table": cal_out},
    "residual_morans_i": {
        "I": float(mi.I), "expected": float(mi.EI),
        "z": float(mi.z_sim), "p": float(mi.p_sim),
        "knn_k": int(k), "n_residuals": int(len(resid)),
        "interpretation": interp,
    },
}
sjson = os.path.join(TABLES_DIR, "Table_S7_assumption_summary.json")
with open(sjson, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print()
print("=" * 60)
print("STEP 5 DONE.")
print("=" * 60)
print(f"  Max VIF      : {max_vif:.2f}")
print(f"  Brier        : {brier:.4f}")
print(f"  Moran I      : {mi.I:.4f}  (p = {mi.p_sim:.4f})")
print()
print("  06_Final_Tables/all_runs_raw.xlsx")
print("  06_Final_Tables/Table_S2_random_seed_sensitivity.xlsx")
print("  06_Final_Tables/Table_S3_threshold_sensitivity.xlsx")
print("  03_Training_Tables/sampling_QC_summary.csv")
print(f"  {TABLES_DIR}/Table_S4_VIF.csv")
print(f"  {TABLES_DIR}/Table_S5_calibration_bins.csv")
print(f"  {TABLES_DIR}/Table_S6_residual_Moran.csv")
print(f"  {TABLES_DIR}/Table_S7_assumption_summary.json")
print(f"  {FIGS_DIR}/Fig_S_calibration.png")
try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo("STEP 5 done",
        f"VIF: {max_vif:.2f}  Brier: {brier:.4f}  Moran I: {mi.I:.4f}")
    a.destroy()
except Exception: pass
