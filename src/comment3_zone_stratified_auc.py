import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import roc_auc_score, average_precision_score
REPO     = r"C:\Users\saadz\Documents\wildfire-bc-bilstm-pso"
COMBO    = "BiLSTM_PSO/thr70/seed42"
OOF_CSV  = os.path.join(REPO, "revision_c8c11", "05_Model_Results",
                        COMBO, "cv_oof_predictions.csv")
TRAIN_CSV = os.path.join(REPO, "revision_c8c11", "03_Training_Tables",
                          "training_thr70_seed42.csv")
BEC      = os.path.join(REPO, "revision_c8c11", "01_Input_Data",
                        "BEC_zones", "BEC_BIOGEOCLIMATIC_POLY", "BEC_POLY_polygon.shp")
OUT_TBL  = os.path.join(REPO, "revision_c8c11", "06_Final_Tables")
os.makedirs(OUT_TBL, exist_ok=True)
print("=" * 70)
print("D3: BEC zone-stratified OOF AUC for BiLSTM-PSO thr70 seed42")
print("=" * 70)
oof = pd.read_csv(OOF_CSV)
print(f"OOF rows: {len(oof)}")
train = pd.read_csv(TRAIN_CSV)
print(f"Training rows: {len(train)}")
if "Latitude" not in train.columns or "Longitude" not in train.columns:
    raise SystemExit("Training CSV missing Latitude/Longitude")
oof = oof.merge(train[["Latitude","Longitude","Status"]].reset_index(),
                left_on="index", right_on="index", how="left",
                suffixes=("","_train"))
oof = oof.dropna(subset=["Latitude","Longitude","y_true","y_pred_oof"])
print(f"After merge with lat/lon: {len(oof)}")
print(f"Loading BEC zones: {BEC}")
bec = gpd.read_file(BEC).to_crs("EPSG:4326")[["MAP_LABEL","ZONE","geometry"]]
oof_gdf = gpd.GeoDataFrame(
    oof,
    geometry=[Point(xy) for xy in zip(oof["Longitude"], oof["Latitude"])],
    crs="EPSG:4326")
joined = gpd.sjoin(oof_gdf, bec, how="left", predicate="within")
joined["zone"] = joined["ZONE"].fillna("Unknown")
print(f"Zones present: {sorted(joined['zone'].unique())}")
HIGH_FIRE = {"IDF", "PP", "ESSF", "SBS"}
joined["zone_group"] = joined["zone"].where(joined["zone"].isin(HIGH_FIRE), "Other")
def boot_auc(y_true, y_score, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            out.append(metric_fn(y_true.iloc[idx], y_score.iloc[idx]))
        except Exception: pass
    if not out: return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))
rows = []
for zname, g in joined.groupby("zone_group"):
    y = g["y_true"].astype(int); s = g["y_pred_oof"].astype(float)
    if y.nunique() < 2 or len(g) < 30: continue
    roc = roc_auc_score(y, s); pr = average_precision_score(y, s)
    roc_ci = boot_auc(y, s, roc_auc_score)
    pr_ci  = boot_auc(y, s, average_precision_score)
    rows.append({"zone": zname,
                 "n": len(g),
                 "n_fires": int((y==1).sum()),
                 "n_pseudo": int((y==0).sum()),
                 "OOF_ROC_AUC": round(roc, 4),
                 "OOF_ROC_AUC_95CI_lo": round(roc_ci[0], 4),
                 "OOF_ROC_AUC_95CI_hi": round(roc_ci[1], 4),
                 "OOF_PR_AUC":  round(pr,  4),
                 "OOF_PR_AUC_95CI_lo": round(pr_ci[0], 4),
                 "OOF_PR_AUC_95CI_hi": round(pr_ci[1], 4)})
df = pd.DataFrame(rows).sort_values("n", ascending=False)
out_csv = os.path.join(OUT_TBL, "T_C3_zone_stratified_AUC.csv")
df.to_csv(out_csv, index=False)
print("\n" + df.to_string(index=False))
print(f"\nSaved: {out_csv}")
