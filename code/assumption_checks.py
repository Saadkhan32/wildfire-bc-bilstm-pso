import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
HERE = Path(os.path.dirname(os.path.abspath(__file__)))
SEED_FOR_CHECKS   = 42
MODEL_FOR_CHECKS  = "BILSTM"
_ap = argparse.ArgumentParser(description="Statistical assumption checks")
_ap.add_argument("--seed-csv",
                 default=str(HERE / ".." / "data" / f"training_points_70ha_seed{SEED_FOR_CHECKS}.csv"),
                 help="training-points CSV for the seed (default: shipped seed table)")
_ap.add_argument("--model-dir",
                 default=str(HERE / ".." / "outputs" / "lstm_bilstm_spatialcv" / f"seed_{SEED_FOR_CHECKS}" / MODEL_FOR_CHECKS),
                 help="folder with holdout_predictions.csv, selected_features_final.csv, feature_meta.json "
                      "(produced by train_models_spatialcv.py, or from the Zenodo model archive)")
_ap.add_argument("--root", default=str(HERE / ".."),
                 help="project root for tables/ and figs/ outputs")
_args = _ap.parse_args()
PROJECT_ROOT   = Path(_args.root)
TABLES_DIR = PROJECT_ROOT / "tables"
FIGS_DIR   = PROJECT_ROOT / "figs"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)
SEED_CSV       = Path(_args.seed_csv)
SEED_MODEL_DIR = Path(_args.model_dir)
HOLDOUT_CSV    = SEED_MODEL_DIR / "holdout_predictions.csv"
SELECTED_FEATS = SEED_MODEL_DIR / "selected_features_final.csv"
META_JSON      = SEED_MODEL_DIR / "feature_meta.json"
print(f"[INFO] PROJECT_ROOT     = {PROJECT_ROOT}")
print(f"[INFO] SEED_CSV         = {SEED_CSV}")
print(f"[INFO] HOLDOUT_CSV      = {HOLDOUT_CSV}")
print(f"[INFO] SELECTED_FEATS   = {SELECTED_FEATS}")
for p in (SEED_CSV, HOLDOUT_CSV, SELECTED_FEATS):
    if not p.exists():
        raise FileNotFoundError(
            f"Required file missing: {p}\n"
            "Run the seed-sensitivity training first, or update PATHS at the top of this script."
        )
print("\n========== Check 1: Variance Inflation Factor (VIF) ==========")
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(SEED_CSV)
selected = pd.read_csv(SELECTED_FEATS)["selected_feature"].tolist()
def strip_prefix(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name
selected_raw = []
for s in selected:
    base = strip_prefix(s)
    base = base.split("_")[0] if base.lower().startswith("lulc") else base
    selected_raw.append(base)
selected_raw = list(dict.fromkeys(selected_raw))
available = [c for c in selected_raw if c in df.columns]
missing   = [c for c in selected_raw if c not in df.columns]
if missing:
    print(f"[WARN] {len(missing)} RFE features not found as raw columns in seed CSV: {missing[:5]}{'...' if len(missing) > 5 else ''}")
X_vif = df[available].select_dtypes(include=[np.number]).copy()
X_vif = X_vif.dropna()
print(f"[INFO] VIF input: {X_vif.shape[0]} rows, {X_vif.shape[1]} numeric predictors")
X_vif_std = pd.DataFrame(StandardScaler().fit_transform(X_vif), columns=X_vif.columns)
vif_rows = []
for i, col in enumerate(X_vif_std.columns):
    try:
        v = variance_inflation_factor(X_vif_std.values, i)
    except Exception as e:
        v = float("nan")
        print(f"[WARN] VIF for {col} failed: {e}")
    vif_rows.append({"predictor": col, "VIF": round(float(v), 3)})
vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
def vif_flag(v):
    if np.isnan(v): return "N/A"
    if v < 5:  return "OK (<5)"
    if v < 10: return "Watch (5-10)"
    return "Problem (>=10)"
vif_df["flag"] = vif_df["VIF"].apply(vif_flag)
vif_out = TABLES_DIR / "T_vif_predictors.csv"
vif_df.to_csv(vif_out, index=False)
print(f"[OK] Wrote {vif_out}")
print(vif_df.to_string(index=False))
max_vif = vif_df["VIF"].max()
n_over_5 = int((vif_df["VIF"] >= 5).sum())
print(f"\n[SUMMARY] max VIF = {max_vif:.2f};  predictors with VIF >= 5: {n_over_5}")
print("\n========== Check 2: Calibration ==========")
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
pred = pd.read_csv(HOLDOUT_CSV)
y_true = pred["y_true"].values.astype(int)
y_prob = pred["y_prob"].values.astype(float)
brier  = brier_score_loss(y_true, y_prob)
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
cal_df = pd.DataFrame({"mean_predicted_prob": mean_pred,
                       "observed_fire_rate":  frac_pos})
cal_out = TABLES_DIR / "T_calibration_bins.csv"
cal_df.to_csv(cal_out, index=False)
fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect calibration")
ax.plot(mean_pred, frac_pos, marker="o", ms=8, lw=1.6,
        color="#c0392b", label=f"BiLSTM-PSO (seed {SEED_FOR_CHECKS})")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed fire rate")
ax.set_title(f"Reliability diagram\nBrier score = {brier:.4f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
cal_png = FIGS_DIR / "Fig_S_calibration.png"
plt.savefig(cal_png, dpi=300, bbox_inches="tight")
plt.savefig(cal_png.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"[OK] Wrote {cal_out}")
print(f"[OK] Wrote {cal_png}")
print(f"[SUMMARY] Brier score = {brier:.4f} (lower is better; perfectly calibrated = 0)")
print("\n========== Check 3: Residual Moran's I ==========")
try:
    from libpysal.weights import KNN
    from esda.moran import Moran
except ImportError:
    raise SystemExit("Need libpysal + esda: pip install libpysal esda")
from sklearn.model_selection import train_test_split
full = pd.read_csv(SEED_CSV)
y_all = pd.to_numeric(full["Status"], errors="coerce").fillna(0).astype(int).values
X_all = full.drop(columns=["Status"], errors="ignore")
_, X_te_idx_proxy, _, y_te_proxy = train_test_split(
    np.arange(len(y_all)), y_all,
    test_size=0.30, random_state=SEED_FOR_CHECKS, stratify=y_all,
)
if len(X_te_idx_proxy) != len(pred):
    print(f"[WARN] test-set size mismatch: {len(X_te_idx_proxy)} vs {len(pred)}; "
          "computing on min-overlap. Verify the same TEST_SIZE / random_state was used "
          "in random_seed_sensitivity_train.py.")
n = min(len(X_te_idx_proxy), len(pred))
test_idx = X_te_idx_proxy[:n]
coords = full.loc[test_idx, ["Longitude", "Latitude"]].dropna().values
residuals = (y_true[:n] - y_prob[:n])[:len(coords)]
if len(coords) < 30:
    raise SystemExit(f"Too few coordinate-pairs ({len(coords)}) to compute Moran's I.")
k = min(8, max(2, len(coords) // 50))
w = KNN.from_array(coords, k=k)
w.transform = "r"
mi = Moran(residuals, w, permutations=999)
mi_row = {
    "model":            f"seed-{SEED_FOR_CHECKS} {MODEL_FOR_CHECKS}",
    "n_test_residuals": len(residuals),
    "knn_k":            k,
    "morans_I":         round(float(mi.I), 4),
    "expected_I":       round(float(mi.EI), 4),
    "z_score":          round(float(mi.z_sim), 3),
    "p_value":          round(float(mi.p_sim), 4),
    "interpretation":   "Independent (low spatial autocorrelation)" if mi.p_sim > 0.05
                        else "Spatially autocorrelated residuals (model may be misspecified)",
}
mi_df  = pd.DataFrame([mi_row])
mi_out = TABLES_DIR / "T_residual_morans_i.csv"
mi_df.to_csv(mi_out, index=False)
print(f"[OK] Wrote {mi_out}")
print(mi_df.to_string(index=False))
print("\n" + "=" * 60)
print("Comment 8 assumption checks complete")
print("=" * 60)
print(f"  T_vif_predictors.csv       max VIF = {max_vif:.2f}, "
      f"predictors VIF>=5: {n_over_5}")
print(f"  T_calibration_bins.csv     Brier   = {brier:.4f}")
print(f"  Fig_S_calibration.png      reliability diagram")
print(f"  T_residual_morans_i.csv    Moran's I = {mi.I:.4f}, p = {mi.p_sim:.4f}")
summary = {
    "comment": "",
    "seed_audited": SEED_FOR_CHECKS,
    "model_audited": MODEL_FOR_CHECKS,
    "vif": {"max": float(max_vif), "n_predictors_ge_5": int(n_over_5),
            "table": str(vif_out)},
    "calibration": {"brier_score": float(brier),
                    "figure": str(cal_png), "bins_table": str(cal_out)},
    "residual_morans_i": {"I": float(mi.I), "expected": float(mi.EI),
                          "z": float(mi.z_sim), "p": float(mi.p_sim),
                          "knn_k": k, "n_residuals": int(len(residuals)),
                          "interpretation": mi_row["interpretation"]},
}
summary_out = TABLES_DIR / "T_c8_assumption_summary.json"
with summary_out.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
