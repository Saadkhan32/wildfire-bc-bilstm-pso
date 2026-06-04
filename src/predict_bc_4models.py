"""
predict_bc_4models.py
=====================
Apply the 4 winning seed42/thr70 models (LSTM, BiLSTM, LSTM-PSO, BiLSTM-PSO) to
the BC 500 m predictor raster stack to produce 4 wildfire susceptibility GeoTIFFs.

Uses the EXACT same prepare_features() pipeline as the trainer (Aspect->sin/cos,
log1p on distance columns, LULC->category, drop ID/lat/lon).

Inputs:
  revision_c8c11/01_Input_Data/rasters/{19 GeoTIFFs}
  revision_c8c11/05_Model_Results/{MODEL}/thr70/seed42/{final_model.keras,
                                                         static_preprocessor.joblib,
                                                         selected_features_final.csv}
Outputs:
  revision_c8c11/02_GIS_Output/BC_susceptibility_{MODEL}.tif    (float32, 0-1)
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.windows import Window

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(f"TF {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

# ----- Custom layer for non-PSO models -----
try:
    register_ks = keras.saving.register_keras_serializable
except Exception:
    register_ks = keras.utils.register_keras_serializable

@register_ks(package="Wildfire")
class AttentionPool1D(layers.Layer):
    def __init__(self, attn_units=128, **kwargs):
        super().__init__(**kwargs)
        self.attn_units = int(attn_units)
        self.d1 = layers.Dense(self.attn_units, activation="tanh")
        self.d2 = layers.Dense(1)
        self.softmax = layers.Softmax(axis=1)
    def call(self, x):
        a = self.softmax(self.d2(self.d1(x)))
        return tf.reduce_sum(x * a, axis=1)
    def get_config(self):
        return {"attn_units": self.attn_units, **super().get_config()}

# ----- Paths -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
RAS_DIR    = os.path.join(REPO_ROOT, "revision_c8c11", "01_Input_Data", "rasters")
RESULTS    = os.path.join(REPO_ROOT, "revision_c8c11", "05_Model_Results")
OUT_DIR    = os.path.join(REPO_ROOT, "revision_c8c11", "02_GIS_Output")
os.makedirs(OUT_DIR, exist_ok=True)

# Raster -> column-name mapping. Must match the column names the trainer expects.
RASTER_COLS = {
    "Slope":               "Slope.tif",
    "Elevation":           "Elevation.tif",
    "TWI":                 "TWI.tif",
    "Profile_Curvature":   "Profile_Curvature.tif",
    "Plan_Curvature":      "Plan_Curvature.tif",
    "NDVI":                "NDVI.tif",
    "LULC":                "LULC.tif",
    "Max_Temperature":     "Max_Temperature.tif",
    "Precipitation":       "Precipitation.tif",
    "WS":                  "WS.tif",
    "Relative_Humidity":   "Relative_Humidity.tif",
    "AET":                 "AET.tif",
    "DSI":                 "DSI.tif",
    "Soil_Moisture":       "Soil_Moisture.tif",
    "Distance_roads":      "Distance_roads.tif",
    "Distance_rivers":     "Distance_rivers.tif",
    "Distance_households": "Distance_households.tif",
    "Aspect":              "Aspect.tif",
}

MODELS = ["LSTM", "BiLSTM", "LSTM_PSO", "BiLSTM_PSO"]

# ----- prepare_features() replica of the trainer -----
def is_distance_like(c):
    lc = c.lower()
    return ("distance" in lc) or lc.startswith("dist") or ("_dist" in lc) or ("_distance" in lc)

def safe_log1p(x):
    x = pd.to_numeric(x, errors="coerce").copy()
    x[x < 0] = np.nan
    return np.log1p(x)

CAT_LIKE = {"lulc","landuse","land_cover","landcover"}

def prepare_features(df_in):
    work = df_in.copy()
    if "Aspect" in work.columns:
        rad = np.deg2rad(pd.to_numeric(work["Aspect"], errors="coerce"))
        work["Aspect_sin"] = np.sin(rad)
        work["Aspect_cos"] = np.cos(rad)
        work.drop(columns=["Aspect"], inplace=True)
    for c in list(work.columns):
        if is_distance_like(c):
            work[c] = safe_log1p(work[c])
    cat_cols = []
    for c in work.columns:
        if (c.lower() in CAT_LIKE) or (work[c].dtype == "object") or str(work[c].dtype).startswith("category"):
            cat_cols.append(c)
    X_df = work.copy()
    for c in work.columns:
        if c not in cat_cols:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
        else:
            X_df[c] = X_df[c].astype("category")
    return X_df

# ----- Load rasters as a stack on the Slope.tif grid (assumed reference) -----
print("\n--- Loading raster stack ---")
ref_path = os.path.join(RAS_DIR, "Slope.tif")
with rasterio.open(ref_path) as ref:
    ref_profile = ref.profile.copy()
    ref_shape   = ref.shape       # (H, W)
    ref_nodata  = ref.nodata

H, W = ref_shape
print(f"  Reference grid (Slope.tif): {H} rows x {W} cols = {H*W:,} pixels")
print(f"  CRS: {ref_profile.get('crs')}, transform: {ref_profile.get('transform')}")

stack = {}
for col, fn in RASTER_COLS.items():
    path = os.path.join(RAS_DIR, fn)
    with rasterio.open(path) as src:
        if src.shape != ref_shape:
            sys.exit(f"FATAL: raster grid mismatch on {fn}: {src.shape} vs ref {ref_shape}")
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
    stack[col] = arr
    print(f"  loaded {fn:30s} dtype=float32 min={np.nanmin(arr):.2f} max={np.nanmax(arr):.2f}")

# ----- Build a single 'valid' mask: every pixel with all features present -----
valid_mask = np.ones((H, W), dtype=bool)
for col, arr in stack.items():
    valid_mask &= np.isfinite(arr)
n_valid = int(valid_mask.sum())
print(f"\n  Valid pixels (all features present): {n_valid:,} of {H*W:,} ({100*n_valid/(H*W):.1f}%)")

# Flatten valid pixels -> dataframe
print("\n--- Building feature DataFrame ---")
flat = {col: arr[valid_mask] for col, arr in stack.items()}
df_pixels = pd.DataFrame(flat)
print(f"  df shape: {df_pixels.shape}")

# Apply trainer-equivalent feature prep
df_prepared = prepare_features(df_pixels)
print(f"  prepared df shape: {df_prepared.shape}")

# ----- Loop over models -----
for model_key in MODELS:
    print(f"\n=== {model_key} ===")
    m_dir = os.path.join(RESULTS, model_key, "thr70", "seed42")
    keras_path = os.path.join(m_dir, "final_model.keras")
    prep_path  = os.path.join(m_dir, "static_preprocessor.joblib")
    feat_csv   = os.path.join(m_dir, "selected_features_final.csv")
    if not (os.path.exists(keras_path) and os.path.exists(prep_path)):
        print(f"  SKIP: missing files in {m_dir}")
        continue

    selected = pd.read_csv(feat_csv)["selected_feature"].tolist()
    print(f"  selected features: {len(selected)}")

    pre = joblib.load(prep_path)
    Xfull = pre.transform(df_prepared)
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
    missing = [s for s in selected if s not in name_to_idx]
    if missing:
        print(f"  ERROR: missing transformed feats (first 5): {missing[:5]}")
        continue
    sel_idx = [name_to_idx[s] for s in selected]
    X = Xfull[:, sel_idx].astype(np.float32)
    print(f"  X shape: {X.shape}")

    model = tf.keras.models.load_model(keras_path, compile=False)
    print(f"  model input shape: {model.input_shape}")

    t0 = time.time()
    BATCH = 16384
    preds = np.zeros(X.shape[0], dtype=np.float32)
    n_batches = (X.shape[0] + BATCH - 1) // BATCH
    for bi in range(n_batches):
        lo, hi = bi*BATCH, min((bi+1)*BATCH, X.shape[0])
        preds[lo:hi] = model.predict(X[lo:hi], verbose=0, batch_size=BATCH).flatten()
        if bi % 20 == 0:
            print(f"    batch {bi+1}/{n_batches}", end="\r")
    print(f"\n  prediction done in {time.time()-t0:.1f}s")
    print(f"  prob range: min={preds.min():.4f} max={preds.max():.4f} mean={preds.mean():.4f}")

    # Write back to a raster matching the reference grid
    out_arr = np.full((H, W), np.nan, dtype=np.float32)
    out_arr[valid_mask] = preds
    out_path = os.path.join(OUT_DIR, f"BC_susceptibility_{model_key}.tif")
    out_profile = ref_profile.copy()
    out_profile.update({
        "dtype": "float32",
        "nodata": -9999.0,
        "count": 1,
        "compress": "lzw",
    })
    out_arr_w = np.where(np.isnan(out_arr), -9999.0, out_arr).astype(np.float32)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(out_arr_w, 1)
    print(f"  wrote {out_path}")
    # Free model memory before next loop
    del model
    tf.keras.backend.clear_session()

print("\nDone.  4 susceptibility rasters saved to revision_c8c11/02_GIS_Output/")
