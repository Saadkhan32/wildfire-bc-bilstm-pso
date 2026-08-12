import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.metrics import (roc_auc_score, brier_score_loss, f1_score,
                              precision_score, recall_score, accuracy_score)
root_tk = tk.Tk(); root_tk.withdraw(); root_tk.attributes("-topmost", True)
folder = filedialog.askdirectory(
    title="Select folder with cv_oof_predictions.csv",
    initialdir=r"G:\Deep learning for wildfire susceptibility mapping"
)
root_tk.destroy()
if not folder:
    print("CANCELLED."); sys.exit(1)
folder = Path(folder)
print(f"Folder: {folder}")
oof_csv = folder / "cv_oof_predictions.csv"
if not oof_csv.exists():
    print(f"ERROR: cv_oof_predictions.csv not found in {folder}"); sys.exit(1)
df = pd.read_csv(oof_csv)
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(df.head(3).to_string())
y_true_col = next((c for c in df.columns if c.lower() in
                   ("y_true","label","target","actual","truth","y","fire")), None)
y_prob_col = next((c for c in df.columns if c.lower() in
                   ("y_pred_prob","y_proba","y_prob","probability","pred_prob",
                    "score","pred_score","oof_pred","oof_proba","prob")), None)
print(f"Detected: y_true_col={y_true_col}, y_prob_col={y_prob_col}")
if y_true_col is None or y_prob_col is None:
    print("Could not auto-detect columns. Inspect:")
    for c in df.columns:
        print(f"  {c}: dtype={df[c].dtype}, min={df[c].min()}, max={df[c].max()}, n_unique={df[c].nunique()}")
    sys.exit(2)
y_true = df[y_true_col].astype(int).values
y_prob = df[y_prob_col].astype(float).values
y_pred = (y_prob >= 0.5).astype(int)
print(f"n={len(y_true)}, positives={y_true.sum()}, prob range=[{y_prob.min():.3f},{y_prob.max():.3f}]")
metrics = {
    "AUC":       roc_auc_score(y_true, y_prob),
    "Brier":     brier_score_loss(y_true, y_prob),
    "Accuracy":  accuracy_score(y_true, y_pred),
    "F1":        f1_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred),
    "Recall":    recall_score(y_true, y_pred),
}
paper = {"AUC":0.94,"Brier":None,"Accuracy":0.885,"F1":0.896,"Precision":0.8601,"Recall":0.9352}
print("\nPOINT ESTIMATES vs paper:")
for m, v in metrics.items():
    p = paper.get(m)
    diff = f"(paper: {p:.4f})" if p else ""
    print(f"  {m:10s} = {v:.4f}  {diff}")
N_BOOT = 1000
rng = np.random.default_rng(42)
n = len(y_true)
boot = {m: [] for m in metrics}
for i in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    yt = y_true[idx]; yp = y_pred[idx]; ypp = y_prob[idx]
    if len(np.unique(yt)) < 2: continue
    boot["AUC"].append(roc_auc_score(yt, ypp))
    boot["Brier"].append(brier_score_loss(yt, ypp))
    boot["Accuracy"].append(accuracy_score(yt, yp))
    boot["F1"].append(f1_score(yt, yp))
    boot["Precision"].append(precision_score(yt, yp))
    boot["Recall"].append(recall_score(yt, yp))
TABLES = Path(os.path.expandvars(r"%USERPROFILE%\Documents\wildfire-bc-bilstm-pso\tables"))
TABLES.mkdir(exist_ok=True)
rows = []
print("\nBOOTSTRAP 95% CIs:")
for m, vals in boot.items():
    a = np.array(vals)
    lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
    rows.append({"metric": m, "point_estimate": round(metrics[m], 4),
                 "boot_mean": round(a.mean(), 4),
                 "ci_lo_95": round(lo, 4), "ci_hi_95": round(hi, 4),
                 "ci_width": round(hi-lo, 4), "n_boot": len(vals)})
    print(f"  {m:10s} = {metrics[m]:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")
pd.DataFrame(rows).to_csv(TABLES / "T_metrics_bootstrap.csv", index=False)
print(f"\nWrote: {TABLES}/T_metrics_bootstrap.csv")
root_tk = tk.Tk(); root_tk.withdraw(); root_tk.attributes("-topmost", True)
messagebox.showinfo("Bootstrap complete", f"95% CIs saved to T_metrics_bootstrap.csv")
root_tk.destroy()
