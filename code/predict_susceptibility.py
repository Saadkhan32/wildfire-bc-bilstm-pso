import os, argparse, joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
from tensorflow import keras
from tensorflow.keras import layers
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODELS = ["LSTM", "BiLSTM", "LSTM_PSO", "BiLSTM_PSO"]
NODATA = -9999.0
DISTANCE_COLS = ["Distance_roads", "Distance_rivers", "Distance_households"]
EXPECTED_RAW = ["Slope", "Elevation", "TWI", "Profile_Curvature", "Plan_Curvature",
                "NDVI", "LULC", "Max_Temperature", "Precipitation", "WS",
                "Specific_Humidity", "AET", "DSI", "Soil_Moisture",
                "Distance_roads", "Distance_rivers", "Distance_households", "Aspect"]

def build_grid(rasters_dir, target_crs, target_res):
    ref_path = os.path.join(rasters_dir, "Elevation.tif")
    with rasterio.open(ref_path) as ref:
        transform, width, height = calculate_default_transform(
            ref.crs, target_crs, ref.width, ref.height, *ref.bounds, resolution=target_res)
    stacked = np.full((height, width, len(EXPECTED_RAW)), np.nan, dtype=np.float32)
    for i, col in enumerate(EXPECTED_RAW):
        p = os.path.join(rasters_dir, col + ".tif")
        if not os.path.exists(p):
            print("  WARN: missing " + col + ".tif")
            continue
        method = Resampling.nearest if col == "LULC" else Resampling.bilinear
        with rasterio.open(p) as src:
            dst = np.full((height, width), np.nan, dtype=np.float32)
            reproject(rasterio.band(src, 1), dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=target_crs,
                      src_nodata=src.nodata, dst_nodata=np.nan,
                      resampling=method)
            stacked[:, :, i] = dst
    return stacked, transform

def build_frame(stacked):
    H, W, F = stacked.shape
    valid = np.isfinite(stacked).all(axis=2).ravel()
    df = pd.DataFrame(stacked.reshape(-1, F), columns=EXPECTED_RAW)
    for c in DISTANCE_COLS:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            x = x.where(x >= 0, np.nan)
            df[c] = np.log1p(x)
    if "Aspect" in df.columns:
        rad = np.deg2rad(pd.to_numeric(df["Aspect"], errors="coerce"))
        df["Aspect_sin"] = np.sin(rad)
        df["Aspect_cos"] = np.cos(rad)
        df = df.drop(columns=["Aspect"])
    return df, H, W, valid

def predict_one(df, H, W, valid, transform, target_crs, model_path, preproc_path, features_path, out_path):
    print("Model: " + model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    preproc = joblib.load(preproc_path)
    sel = pd.read_csv(features_path)["selected_feature"].tolist()
    Xt = preproc.transform(df)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    try:
        names_all = preproc.get_feature_names_out(df.columns)
    except Exception:
        names_all = np.array(["f" + str(i) for i in range(Xt.shape[1])])
    Xs = Xt[:, np.isin(names_all, sel)]
    preds = np.full(H * W, NODATA, dtype=np.float32)
    v_idx = np.where(valid)[0]
    BATCH = 100000
    for s in range(0, len(v_idx), BATCH):
        chunk = v_idx[s:s + BATCH]
        preds[chunk] = model.predict(Xs[chunk], batch_size=2048, verbose=0).ravel()
    rows = v_idx // W
    cols = v_idx % W
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    out = pd.DataFrame({"x_epsg3005": np.round(xs, 3), "y_epsg3005": np.round(ys, 3),
                        "probability": np.round(preds[v_idx], 6)})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print("  wrote " + out_path + " (" + str(len(out)) + " pixels; EPSG:3005 cell centres; "
          "rasterize in ArcGIS Pro via XY Table To Point then Point to Raster at 1500 m)")

def main():
    ap = argparse.ArgumentParser(
        description="Province-wide wildfire susceptibility prediction. With no --model, "
                    "predicts all four models using the standard repository paths.")
    ap.add_argument("--model")
    ap.add_argument("--preprocessor")
    ap.add_argument("--features")
    ap.add_argument("--out")
    ap.add_argument("--rasters", default=os.path.join(REPO, "data", "rasters"))
    ap.add_argument("--target_crs", default="EPSG:3005")
    ap.add_argument("--target_res", type=float, default=1500.0)
    ap.add_argument("--outdir", default=os.path.join(REPO, "outputs", "susceptibility"))
    a = ap.parse_args()
    stacked, transform = build_grid(a.rasters, a.target_crs, a.target_res)
    df, H, W, valid = build_frame(stacked)
    if a.model:
        predict_one(df, H, W, valid, transform, a.target_crs, a.model, a.preprocessor, a.features, a.out)
        return
    for name in MODELS:
        mdir = os.path.join(REPO, "models", name, "thr70", "seed42")
        mpath = os.path.join(mdir, "final_model.keras")
        if not os.path.exists(mpath):
            print("  SKIP " + name + ": " + mpath + " not found (download weights from Zenodo)")
            continue
        predict_one(df, H, W, valid, transform, a.target_crs, mpath,
                    os.path.join(mdir, "static_preprocessor.joblib"),
                    os.path.join(mdir, "selected_features_final.csv"),
                    os.path.join(a.outdir, "BC_susceptibility_" + name + ".csv"))
    print("Probability tables written under " + a.outdir)
    print("Shipped rasters and tables in data/susceptibility/ are unchanged; they stay the canonical inputs for ROC, class-share, and cross-border (Fig. 19 / Table 8).")

if __name__ == "__main__":
    main()
