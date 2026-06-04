"""
build_shap_beeswarm.py
=======================
Generate a SHAP beeswarm plot for the winning model (BiLSTM-PSO seed42 thr70).
Uses GradientExplainer (works with Keras BiLSTM+CNN models in TF 2.10).

Output: revision_c8c11/03_Figures/Fig_SHAP_BiLSTM_PSO_beeswarm.pdf + .png

If the wildfire conda env doesn't have shap installed, run:
    pip install shap --no-deps
(--no-deps avoids pulling in incompatible scikit-image, etc.)
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
print(f"TF {tf.__version__}")

try:
    import shap
    print(f"SHAP {shap.__version__}")
except ImportError:
    print("ERROR: shap not installed. Run:  pip install shap --no-deps")
    sys.exit(1)

# ----- Paths -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
TABLES_DIR = os.path.join(REPO_ROOT, "revision_c8c11", "03_Training_Tables")
RESULTS    = os.path.join(REPO_ROOT, "revision_c8c11", "05_Model_Results")
FIG_DIR    = os.path.join(REPO_ROOT, "revision_c8c11", "03_Figures")
os.makedirs(FIG_DIR, exist_ok=True)

M_DIR = os.path.join(RESULTS, "BiLSTM_PSO", "thr70", "seed42")
CSV   = os.path.join(TABLES_DIR, "training_thr70_seed42.csv")

# ----- Human-readable feature labels -----
FEATURE_LABELS = {
    "num__Slope":              "Slope",
    "num__Elevation":          "Elevation",
    "num__TWI":                "TWI",
    "num__Profile_Curvature":  "Profile Curvature",
    "num__NDVI":               "NDVI",
    "num__Max_Temperature":    "Temperature",
    "num__Precipitation":      "Precipitation",
    "num__WS":                 "Windspeed",
    "num__Relative_Humidity":  "RH",
    "num__AET":                "AET",
    "num__DSI":                "Drought Index",
    "num__Soil_Moisture":      "Soil Moisture",
    "num__Distance_roads":     "Distance from roads",
    "num__Distance_rivers":    "Distance from rivers",
    "num__Distance_households":"Distance from households",
    "cat__LULC_11.0":          "LULC (Trees)",
    "cat__LULC_9.0":           "LULC (Rangeland)",
    "cat__LULC_4.0":           "LULC (Forest)",
    "cat__LULC_2.0":           "LULC (Trees, alt)",
}

# ----- Load data, model, preprocessor -----
print(f"\nLoading data: {CSV}")
df = pd.read_csv(CSV)
print(f"  rows: {len(df)}")

print(f"\nLoading model: {M_DIR}")
selected = pd.read_csv(os.path.join(M_DIR, "selected_features_final.csv"))["selected_feature"].tolist()
pre      = joblib.load(os.path.join(M_DIR, "static_preprocessor.joblib"))
model    = tf.keras.models.load_model(os.path.join(M_DIR, "final_model.keras"),
                                       compile=False, safe_mode=False)
print(f"  selected feats: {len(selected)}")
print(f"  model input shape: {model.input_shape}")

# ----- Transform features -----
Xfull = pre.transform(df)
if hasattr(Xfull, "toarray"):
    Xfull = Xfull.toarray()

try:
    all_names = list(pre.get_feature_names_out())
except Exception:
    all_names = []
    for tname, trans, cols in pre.transformers_:
        try:
            names = trans.get_feature_names_out(cols)
        except Exception:
            names = [f"{tname}__{c}" for c in cols]
        all_names.extend(list(names))

name_to_idx = {n: i for i, n in enumerate(all_names)}
sel_idx     = [name_to_idx[s] for s in selected]
X           = Xfull[:, sel_idx].astype(np.float32)
display_names = [FEATURE_LABELS.get(s, s.replace("num__","").replace("cat__","")) for s in selected]
print(f"  transformed X shape: {X.shape}")

# ----- Sample background + foreground for SHAP -----
rng = np.random.default_rng(42)
N_BG = 100   # background sample for GradientExplainer (small = faster, ~enough for stable beeswarm)
N_FG = 500   # foreground sample to plot
bg_idx = rng.choice(len(X), size=N_BG, replace=False)
fg_idx = rng.choice(len(X), size=N_FG, replace=False)
X_bg = X[bg_idx]
X_fg = X[fg_idx]
print(f"  background: {X_bg.shape}, foreground: {X_fg.shape}")

# ----- Compute SHAP values with GradientExplainer (Keras-compatible in TF 2.10) -----
print("\nComputing SHAP values (GradientExplainer)...")
t0 = time.time()
explainer = shap.GradientExplainer(model, X_bg)
shap_values = explainer.shap_values(X_fg)
# shap_values is a list of arrays (one per output); for binary sigmoid it's a single output
if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.array(shap_values).squeeze()
if shap_values.ndim == 3:
    shap_values = shap_values[..., 0]
print(f"  done in {time.time()-t0:.1f}s, shape={shap_values.shape}")

# ----- Plot beeswarm -----
print("\nBuilding beeswarm plot...")
plt.figure(figsize=(9, 7))
shap.summary_plot(
    shap_values, X_fg,
    feature_names=display_names,
    plot_type="dot",
    show=False,
    max_display=len(display_names),
    sort=True,
    color_bar=True,
    cmap=plt.get_cmap("coolwarm"),
)
fig = plt.gcf()
ax  = plt.gca()
ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)

# Customize colorbar label
for child in fig.get_children():
    if hasattr(child, "get_label") and child.get_label() == "<colorbar>":
        child.set_ylabel("Feature value", fontsize=10)

pdf_out = os.path.join(FIG_DIR, "Fig_SHAP_BiLSTM_PSO_beeswarm.pdf")
png_out = os.path.join(FIG_DIR, "Fig_SHAP_BiLSTM_PSO_beeswarm.png")
plt.tight_layout()
plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
plt.savefig(png_out, dpi=200, bbox_inches="tight")
print(f"Wrote {pdf_out}")
print(f"Wrote {png_out}")

# Also save raw SHAP values for the manuscript SI
import pickle
with open(os.path.join(REPO_ROOT, "revision_c8c11", "06_Final_Tables", "SHAP_BiLSTM_PSO_values.pkl"), "wb") as f:
    pickle.dump({
        "shap_values":   shap_values,
        "X_fg":          X_fg,
        "feature_names": display_names,
        "model_key":     "BiLSTM_PSO_thr70_seed42",
    }, f)
print("\nDone.")
