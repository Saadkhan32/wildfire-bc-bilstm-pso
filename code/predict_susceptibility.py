import os, sys, argparse, joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")  # run inference on CPU
except Exception:
    pass
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--preprocessor", required=True)
ap.add_argument("--rasters", required=True, help="folder containing 17 predictor TIFs")
ap.add_argument("--features", required=True, help="selected_features_final.csv")
ap.add_argument("--out", required=True)
ap.add_argument("--target_crs", default="EPSG:3005")
ap.add_argument("--target_res", type=float, default=500.0)
args = ap.parse_args()
print(f"Loading model: {args.model}")
model = tf.keras.models.load_model(args.model, compile=False, custom_objects={})
print(f"Loading preprocessor: {args.preprocessor}")
preproc = joblib.load(args.preprocessor)
print(f"Loading selected features: {args.features}")
sel = pd.read_csv(args.features)["selected_feature"].tolist()
EXPECTED_RAW = ["Slope","Elevation","TWI","Profile_Curvature","Plan_Curvature",
                "NDVI","LULC","Max_Temperature","Precipitation","WS",
                "Specific_Humidity","AET","DSI","Soil_Moisture",
                "Distance_roads","Distance_rivers","Distance_households",
                "Aspect"]
ref_path = os.path.join(args.rasters, "Elevation.tif")
print(f"Using reference grid from: {ref_path}")
with rasterio.open(ref_path) as ref:
    transform, width, height = calculate_default_transform(
        ref.crs, args.target_crs, ref.width, ref.height, *ref.bounds,
        resolution=args.target_res)
    ref_bounds = ref.bounds
print(f"Output grid: {width} x {height} @ {args.target_res} m, {args.target_crs}")
stacked = np.full((height, width, len(EXPECTED_RAW)), np.nan, dtype=np.float32)
for i, col in enumerate(EXPECTED_RAW):
    p = os.path.join(args.rasters, f"{col}.tif")
    if not os.path.exists(p):
        print(f"  WARN: missing {col}.tif -- using NaNs"); continue
    print(f"  Reprojecting {col} ...")
    with rasterio.open(p) as src:
        dst = np.full((height, width), np.nan, dtype=np.float32)
        reproject(rasterio.band(src,1), dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=args.target_crs,
                  resampling=Resampling.bilinear)
        stacked[:,:,i] = dst
print("Building DataFrame for preprocessor ...")
H, W, F = stacked.shape
flat = stacked.reshape(-1, F)
df = pd.DataFrame(flat, columns=EXPECTED_RAW)
if "Aspect" in df.columns:
    rad = np.deg2rad(df["Aspect"].fillna(0).values % 360)
    df["Aspect_sin"] = np.sin(rad); df["Aspect_cos"] = np.cos(rad)
    df = df.drop(columns=["Aspect"])
print(f"Total pixels: {len(df)}; valid (any non-NaN): {df.notna().any(axis=1).sum()}")
print("Applying preprocessor ...")
Xt = preproc.transform(df)
if hasattr(Xt, "toarray"): Xt = Xt.toarray()
try:
    names_all = preproc.get_feature_names_out(df.columns)
except Exception:
    names_all = np.array([f"f{i}" for i in range(Xt.shape[1])])
mask = np.isin(names_all, sel)
Xs = Xt[:, mask]
print(f"After RFE mask: {Xs.shape[1]} features (expected {len(sel)})")
valid = ~np.isnan(Xs).any(axis=1)
print(f"Valid rows for prediction: {valid.sum()}")
preds = np.full(H*W, np.nan, dtype=np.float32)
BATCH = 100_000
v_idx = np.where(valid)[0]
print("Predicting ...")
for s in range(0, len(v_idx), BATCH):
    chunk = v_idx[s:s+BATCH]
    p = model.predict(Xs[chunk], batch_size=2048, verbose=0).ravel()
    preds[chunk] = p
    if s % (BATCH*5) == 0: print(f"  {s}/{len(v_idx)}")
print("Done predicting.")
out_arr = preds.reshape(H, W)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
print(f"Writing: {args.out}")
with rasterio.open(args.out, "w",
        driver="GTiff", height=H, width=W, count=1,
        dtype="float32", crs=args.target_crs, transform=transform,
        nodata=np.nan, compress="lzw") as dst:
    dst.write(out_arr, 1)
print(f"OK -- {valid.sum()} valid pixels written.")
