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

# tqdm with auto-install fallback (same pattern as the headless trainer)
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found; installing...", flush=True)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm",
                            "--quiet", "--disable-pip-version-check"])
    from tqdm import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Force CPU-only mode if FORCE_CPU=1 is set (e.g. on AMD/DirectML desktops where
# CudnnRNN has no GPU kernel for LSTM/BiLSTM). Inference is still fast on CPU
# for the ~500K-1M valid BC pixels.
if os.environ.get("FORCE_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(f"TF {tf.__version__}")

# If any GPU is visible AND it's a DirectML pluggable device, hide it.
# This prevents the CudnnRNN-not-supported crash during LSTM prediction.
gpus = tf.config.list_physical_devices('GPU')
if gpus and os.environ.get("FORCE_CPU") != "1":
    # Heuristic: DirectML reports device_type='GPU' but the device name has no
    # NVIDIA marker. Safest is to just hide all GPUs unless the user explicitly
    # wants GPU; we're doing CPU inference here.
    print(f"  Detected GPU devices: {gpus}")
    print(f"  Hiding GPUs from TF (CPU-only inference for LSTM/BiLSTM compatibility)")
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception as e:
        print(f"  Warning: could not hide GPUs ({e}); proceeding anyway")
print(f"Effective devices: {tf.config.list_physical_devices()}")

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
for col, fn in tqdm(list(RASTER_COLS.items()), desc="Loading rasters", unit="raster"):
    path = os.path.join(RAS_DIR, fn)
    with rasterio.open(path) as src:
        if src.shape != ref_shape:
            sys.exit(f"FATAL: raster grid mismatch on {fn}: {src.shape} vs ref {ref_shape}")
        arr = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
    stack[col] = arr
    tqdm.write(f"  {fn:30s}  min={np.nanmin(arr):.2f}  max={np.nanmax(arr):.2f}")

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

# Capture (row, col) and projected (x, y) coordinates of each valid pixel — used
# later for the combined CSV/XLSX export.
rows_idx, cols_idx = np.where(valid_mask)
ref_transform = ref_profile["transform"]
# Cell centers: affine * (col + 0.5, row + 0.5)
x_proj = ref_transform.a * (cols_idx + 0.5) + ref_transform.b * (rows_idx + 0.5) + ref_transform.c
y_proj = ref_transform.d * (cols_idx + 0.5) + ref_transform.e * (rows_idx + 0.5) + ref_transform.f
ref_crs = ref_profile.get("crs")
print(f"  pixel coords ready: row/col + projected x/y in CRS {ref_crs}")

# Apply trainer-equivalent feature prep
df_prepared = prepare_features(df_pixels)
print(f"  prepared df shape: {df_prepared.shape}")

# Accumulator for combined CSV/XLSX export
all_preds_dict = {}     # model_key -> 1D prob array (length = n_valid pixels)

# ----- Loop over models -----
model_pbar = tqdm(MODELS, desc="Models", unit="model", position=0, leave=True)
for model_key in model_pbar:
    model_pbar.set_postfix_str(model_key)
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
    batch_pbar = tqdm(range(n_batches), desc=f"  {model_key} predict",
                      unit="batch", position=1, leave=False)
    for bi in batch_pbar:
        lo, hi = bi*BATCH, min((bi+1)*BATCH, X.shape[0])
        preds[lo:hi] = model.predict(X[lo:hi], verbose=0, batch_size=BATCH).flatten()
    batch_pbar.close()
    print(f"  prediction done in {time.time()-t0:.1f}s")
    print(f"  prob range: min={preds.min():.4f} max={preds.max():.4f} mean={preds.mean():.4f}")

    # Store predictions for combined CSV/XLSX export later
    all_preds_dict[model_key] = preds.copy()

    # --- Continuous probability raster (float32, 0-1) ---
    out_arr = np.full((H, W), np.nan, dtype=np.float32)
    out_arr[valid_mask] = preds
    cont_path = os.path.join(OUT_DIR, f"BC_susceptibility_{model_key}.tif")
    cont_profile = ref_profile.copy()
    cont_profile.update({
        "dtype": "float32", "nodata": -9999.0, "count": 1, "compress": "lzw",
    })
    cont_arr_w = np.where(np.isnan(out_arr), -9999.0, out_arr).astype(np.float32)
    with rasterio.open(cont_path, "w", **cont_profile) as dst:
        dst.write(cont_arr_w, 1)
    print(f"  wrote {cont_path}")

    # --- Classified raster (uint8, 1-5) using fixed manual breaks ---
    # 1 = Very Low (<= 0.2)
    # 2 = Low      (0.2 < x <= 0.4)
    # 3 = Moderate (0.4 < x <= 0.6)
    # 4 = High     (0.6 < x <= 0.8)
    # 5 = Very High(0.8 < x <= 1.0)
    cls_arr = np.zeros((H, W), dtype=np.uint8)   # 0 = nodata
    p = out_arr  # nan-aware
    cls_arr[(p <= 0.2)]                     = 1
    cls_arr[(p >  0.2) & (p <= 0.4)]        = 2
    cls_arr[(p >  0.4) & (p <= 0.6)]        = 3
    cls_arr[(p >  0.6) & (p <= 0.8)]        = 4
    cls_arr[(p >  0.8)]                     = 5
    # Force nodata pixels back to 0 (the NaN comparisons above produce False, so they stayed 0)

    cls_path = os.path.join(OUT_DIR, f"BC_susceptibility_class_{model_key}.tif")
    cls_profile = ref_profile.copy()
    cls_profile.update({
        "dtype": "uint8", "nodata": 0, "count": 1, "compress": "lzw",
    })
    with rasterio.open(cls_path, "w", **cls_profile) as dst:
        dst.write(cls_arr, 1)
        # Embed the colormap so ArcGIS Pro / QGIS auto-display it:
        # (R, G, B, A) per class, matching the user's reference symbology
        dst.write_colormap(1, {
            0: (0,   0,   0,   0),     # nodata, transparent
            1: (26,  150, 65,  255),   # Very Low  -> dark green
            2: (166, 217, 106, 255),   # Low       -> light green
            3: (255, 255, 191, 255),   # Moderate  -> pale yellow
            4: (253, 174, 97,  255),   # High      -> orange
            5: (215, 25,  28,  255),   # Very High -> red
        })
    # Class counts for diagnostic
    bc = np.bincount(cls_arr.flatten(), minlength=6)
    print(f"  wrote {cls_path}")
    print(f"  class counts: VeryLow={bc[1]:,}  Low={bc[2]:,}  Mod={bc[3]:,}  "
          f"High={bc[4]:,}  VeryHigh={bc[5]:,}  (nodata={bc[0]:,})")

    # Free model memory before next loop
    del model
    tf.keras.backend.clear_session()

model_pbar.close()
print("\nDone.  8 rasters saved to revision_c8c11/02_GIS_Output/:")
print("  4 continuous (float32, 0-1):    BC_susceptibility_{MODEL}.tif")
print("  4 classified (uint8, 1-5):      BC_susceptibility_class_{MODEL}.tif")
print("\nThe classified rasters carry an embedded colormap (Very Low -> Very High,")
print("green to red) so ArcGIS Pro displays them with the correct symbology on open.")

# ----- Combined per-pixel CSV / XLSX (one row per valid BC pixel, all 4 models) -----
print("\n--- Building combined per-pixel probability table (all 4 models) ---")

def classify(p):
    c = np.zeros_like(p, dtype=np.uint8)
    c[(p <= 0.2)]                    = 1
    c[(p >  0.2) & (p <= 0.4)]       = 2
    c[(p >  0.4) & (p <= 0.6)]       = 3
    c[(p >  0.6) & (p <= 0.8)]       = 4
    c[(p >  0.8)]                    = 5
    return c

n_valid = int(valid_mask.sum())
combined = {
    "pixel_idx": np.arange(n_valid, dtype=np.int64),
    "row":       rows_idx.astype(np.int32),
    "col":       cols_idx.astype(np.int32),
    "x_proj":    x_proj.astype(np.float64),
    "y_proj":    y_proj.astype(np.float64),
}
for m in MODELS:
    if m in all_preds_dict:
        combined[f"prob_{m}"]  = all_preds_dict[m].astype(np.float32)
        combined[f"class_{m}"] = classify(all_preds_dict[m])

df_out = pd.DataFrame(combined)
print(f"  combined table shape: {df_out.shape}")
print(f"  columns: {list(df_out.columns)}")

# CSV (always) - chunked write with progress bar so user knows it's progressing
csv_path = os.path.join(OUT_DIR, "BC_susceptibility_pixels_all4models.csv")
t0 = time.time()
CHUNK = 50_000
n_chunks = (len(df_out) + CHUNK - 1) // CHUNK
csv_pbar = tqdm(total=n_chunks, desc="Writing CSV", unit="chunk")
with open(csv_path, "w", newline="") as f:
    df_out.head(0).to_csv(f, index=False)   # write header
    for ci in range(n_chunks):
        lo, hi = ci*CHUNK, min((ci+1)*CHUNK, len(df_out))
        df_out.iloc[lo:hi].to_csv(f, index=False, header=False, float_format="%.6f")
        csv_pbar.update(1)
csv_pbar.close()
csv_mb = os.path.getsize(csv_path) / 1024 / 1024
print(f"  wrote {csv_path}  ({csv_mb:.1f} MB, {time.time()-t0:.1f}s)")

# Excel (only if rows fit Excel's 1,048,576-row limit, and openpyxl is present)
XLSX_ROW_LIMIT = 1_048_575  # header + data <= 1,048,576
if len(df_out) > XLSX_ROW_LIMIT:
    print(f"  XLSX skipped: {len(df_out):,} rows exceeds Excel limit of {XLSX_ROW_LIMIT:,}")
else:
    try:
        xlsx_path = os.path.join(OUT_DIR, "BC_susceptibility_pixels_all4models.xlsx")
        t0 = time.time()
        print(f"  Writing XLSX (may take 30s-2min for {len(df_out):,} rows)...")
        # openpyxl is the default xlsx writer for pandas
        df_out.to_excel(xlsx_path, index=False, engine="openpyxl")
        xlsx_mb = os.path.getsize(xlsx_path) / 1024 / 1024
        print(f"  wrote {xlsx_path}  ({xlsx_mb:.1f} MB, {time.time()-t0:.1f}s)")
    except ModuleNotFoundError:
        print(f"  XLSX skipped: openpyxl not installed.  Install with:")
        print(f"    pip install openpyxl")
    except Exception as e:
        print(f"  XLSX skipped: {e}")

# Per-model summary lines (mean prob, class distribution)
print("\nPer-model probability summary:")
for m in MODELS:
    if m not in all_preds_dict:
        continue
    p = all_preds_dict[m]
    cl = classify(p)
    pct = lambda k: 100 * (cl == k).sum() / len(cl)
    print(f"  {m:<14} mean prob = {p.mean():.4f}   "
          f"VeryLow={pct(1):5.1f}%  Low={pct(2):5.1f}%  Mod={pct(3):5.1f}%  "
          f"High={pct(4):5.1f}%  VeryHigh={pct(5):5.1f}%")
