# -*- coding: utf-8 -*-
"""
LSTM and BiLSTM code with FE Changed LR 10 fold Spatial CV.py
=============================================================
Non-PSO LSTM and BiLSTM wildfire-susceptibility classifiers with
leakage-safe RF-RFE feature selection, evaluated under 10-fold spatial
GroupKFold (50 km blocks). This is the interactive (Tkinter file-picker)
companion to src/c8c11_non_pso_cli.py and produces identical artefacts
so they can be aggregated together by step4 collectors.

Companion repository for the manuscript:
  "An Interpretable Deep Learning Framework for Wildfire Susceptibility
   and Exposure Assessment in Western Canada", Ecological Informatics, 2026.

Artefacts written under <out_root>/<LSTM|BILSTM>/ :
  - final_model.keras
  - static_preprocessor.joblib
  - selected_features_final.csv
  - feature_meta.json
  - cv_metrics_10fold.csv
  - cv_oof_predictions.csv
  - metrics_summary.json
  - best_params.json
"""

import os, math, json, time, joblib, random, logging, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tkinter import Tk, filedialog

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
tf.get_logger().setLevel(logging.ERROR)

# ------------------------------ CONFIG ---------------------------------------
TARGET_NAME   = "Status"   # loader expects 'Status'
ASPECT_KEY    = "Aspect"
RANDOM_STATE  = 42
GRID_KM       = 50.0       # 50 km spatial CV block size

# Training (fixed 30 epochs)
EPOCHS             = 30
EARLYSTOP_PATIENCE = 8
PLATEAU_PATIENCE   = 3
MIN_LR             = 1e-6
N_FOLDS            = 10

# RFE (~3k rows)
RFE_KEEP_FRACTION = 0.60
RFE_MIN_FEATURES  = 12
RFE_N_ESTIMATORS  = 400
RFE_MAX_DEPTH     = None

# Class weights
USE_CLASS_WEIGHTS = "auto"   # auto / True / False

# Simple fixed hparams (no CNN)
# Forced: lr=0.0003, batch_size=64 for BOTH models
BASE_HPS = {
    "lstm": dict(
        units=256, layers=1,
        dropout=0.20, recurrent_drop=0.10,
        l2=1e-6, lr=3e-4,
        batch_size=64, label_smooth=0.01, spatial_drop=0.08
    ),
    "bilstm": dict(
        units=320, layers=2,
        dropout=0.22, recurrent_drop=0.12,
        l2=2e-6, lr=3e-4,
        batch_size=64, label_smooth=0.01, spatial_drop=0.12
    ),
}
RUNS = ["lstm", "bilstm"]

# ------------------------------ IO & UTILS -----------------------------------
def set_all_seeds(seed=RANDOM_STATE):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        # TF 2.10 lacks deterministic GPU impl of UnsortedSegmentSum (used by
        # Keras AUC metric); enable bit-level op-determinism only on CPU.
        if not tf.config.list_physical_devices("GPU"):
            tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

def ask_paths() -> Tuple[str, str]:
    Tk().withdraw()
    fpath = filedialog.askopenfilename(title="Select your Excel/CSV file",
                                       filetypes=[("Tables", "*.xlsx *.xls *.csv")])
    if not fpath: raise SystemExit("No data file chosen.")
    outdir = filedialog.askdirectory(title="Select an output folder (Cancel = use data folder)")
    if not outdir:
        outdir = os.path.join(os.path.dirname(fpath), "lstm_outputs_for_loader")
    os.makedirs(outdir, exist_ok=True)
    return fpath, outdir

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    return pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)

def find_col_contains(df: pd.DataFrame, keyword: str) -> Optional[str]:
    kw = keyword.lower()
    for c in df.columns:
        if kw in c.lower(): return c
    return None

def add_aspect_sincos(df: pd.DataFrame, aspect_col: str) -> pd.DataFrame:
    a = pd.to_numeric(df[aspect_col], errors="coerce")
    rad = np.deg2rad(a % 360.0)
    df["Aspect_sin"] = np.sin(rad)
    df["Aspect_cos"] = np.cos(rad)
    return df.drop(columns=[aspect_col])

def is_distance_like(col: str) -> bool:
    lc = col.lower()
    return ("distance" in lc) or lc.startswith("dist") or ("_dist" in lc) or ("_distance" in lc)

def safe_log1p_inplace(df: pd.DataFrame):
    for c in list(df.columns):
        if c == TARGET_NAME: continue
        if is_distance_like(c):
            s = pd.to_numeric(df[c], errors="coerce").copy()
            s[s < 0] = np.nan
            df[c] = np.log1p(s)
    return df

def compute_block_ids(lat, lon, grid_km=50.0):
    """Same as PSO scripts: WGS84 degree-binning into ~grid_km cells."""
    import math
    lat = np.asarray(lat, dtype=float); lon = np.asarray(lon, dtype=float)
    lat_mean = np.nanmean(lat) if np.isfinite(np.nanmean(lat)) else 45.0
    deg_lat = grid_km / 111.0
    deg_lon = grid_km / (111.0 * max(0.2, math.cos(math.radians(lat_mean))))
    gx = np.floor(lon / deg_lon).astype(np.int64)
    gy = np.floor(lat / deg_lat).astype(np.int64)
    gx_off = gx - gx.min()
    return (gy * (gx_off.max() + 1) + gx_off).astype(np.int64)

def compute_class_weights_dict(y: np.ndarray) -> dict:
    classes = np.array([0,1]); w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {0: float(w[0]), 1: float(w[1])}

def maybe_class_weights(y: np.ndarray):
    if USE_CLASS_WEIGHTS is True:  return compute_class_weights_dict(y)
    if USE_CLASS_WEIGHTS is False: return None
    p = float(np.mean(y == 1))
    return None if 0.45 <= p <= 0.55 else compute_class_weights_dict(y)

# --------------------------- PREPROCESSOR & RFE ------------------------------
def make_ohe():
    try:    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except: return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(X_df: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X_df.columns if str(X_df[c].dtype) in ("object","category")]
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", make_ohe())])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

def _get_feature_names(pre: ColumnTransformer, X_cols: List[str]) -> np.ndarray:
    try:
        return pre.get_feature_names_out(X_cols)
    except Exception:
        Xt = pre.transform(pd.DataFrame([0]*len(X_cols), index=X_cols).T)
        if hasattr(Xt, "toarray"): Xt = Xt.toarray()
        return np.array([f"f{i}" for i in range(Xt.shape[1])])

def run_rfe(pre: ColumnTransformer, X_tr_df: pd.DataFrame, y_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xt = pre.transform(X_tr_df)
    if hasattr(Xt, "toarray"): Xt = Xt.toarray()
    n_total = Xt.shape[1]
    n_keep = max(RFE_MIN_FEATURES, int(round(n_total * RFE_KEEP_FRACTION)))
    n_keep = min(n_keep, n_total)

    names_all = _get_feature_names(pre, list(X_tr_df.columns))
    if n_keep >= n_total:
        return np.ones(n_total, dtype=bool), names_all

    rf = RandomForestClassifier(
        n_estimators=RFE_N_ESTIMATORS,
        max_depth=RFE_MAX_DEPTH,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    rfe = RFE(estimator=rf, n_features_to_select=n_keep, step=0.1)
    rfe.fit(Xt, y_tr)
    mask = rfe.support_.astype(bool)
    names_sel = names_all[mask]
    return mask, names_sel

def transform_with_mask(pre: ColumnTransformer, X_df: pd.DataFrame, mask: Optional[np.ndarray]) -> np.ndarray:
    X = pre.transform(X_df)
    if hasattr(X, "toarray"): X = X.toarray()
    return X[:, mask] if mask is not None else X

# --------------------------- KERAS SERIAL LAYERS -----------------------------
try:
    register_ks = keras.saving.register_keras_serializable  # Keras 3+
except Exception:
    register_ks = keras.utils.register_keras_serializable   # tf.keras fallback

@register_ks(package="Wildfire")
class AttentionPool1D(layers.Layer):
    def __init__(self, attn_units=128, **kwargs):
        super().__init__(**kwargs)
        self.attn_units = int(attn_units)
        self.d1 = layers.Dense(self.attn_units, activation="tanh")
        self.d2 = layers.Dense(1, activation=None)
        self.softmax = layers.Softmax(axis=1)
    def call(self, x):
        h = self.d1(x); e = self.d2(h); a = self.softmax(e)
        return tf.reduce_sum(x * a, axis=1)
    def get_config(self):
        return {"attn_units": self.attn_units, **super().get_config()}

def make_optimizer(lr, clipnorm=1.0):
    try:
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=clipnorm)
    except Exception:
        return keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)

def se_block_timewise(x, ratio=8):
    d = int(x.shape[-1])
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(1, d // ratio), activation="relu")(s)
    s = layers.Dense(d, activation="sigmoid")(s)
    s = layers.Reshape((1, d))(s)
    return layers.Multiply()([x, s])

def build_lstm_or_bilstm_flatinput(
    n_features: int,
    units=240, layers_n=1,
    dropout=0.20, recurrent_drop=0.10, spatial_drop=0.10,
    l2_reg=1e-6, lr=5e-4, layer_norm=True, label_smooth=0.0,
    bidirectional=False
):
    """Accepts FLAT (n_features,) input; internally reshapes to (T=n_features, 1)."""
    reg = regularizers.l2(l2_reg) if (l2_reg and l2_reg > 0) else None
    inp = keras.Input(shape=(n_features,), name="x")
    x = layers.Reshape((n_features, 1), name="reshape_to_seq")(inp)

    if spatial_drop and spatial_drop > 0:
        x = layers.SpatialDropout1D(spatial_drop)(x)

    def lstm_block():
        cell = layers.LSTM(
            units, return_sequences=True,
            dropout=dropout, recurrent_dropout=recurrent_drop,
            kernel_regularizer=reg
        )
        return layers.Bidirectional(cell) if bidirectional else cell

    x = lstm_block()(x)
    if layers_n == 2:
        x = lstm_block()(x)

    x = se_block_timewise(x, ratio=8)

    avg = layers.GlobalAveragePooling1D()(x)
    mx  = layers.GlobalMaxPooling1D()(x)
    att = AttentionPool1D(attn_units=max(64, units//2), name="attention_pool")(x)
    x   = layers.Concatenate()([avg, mx, att])

    if layer_norm:
        x = layers.LayerNormalization()(x)

    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)(x)
    model = keras.Model(inputs=inp, outputs=out, name=("BiLSTM" if bidirectional else "LSTM"))

    opt  = make_optimizer(lr, clipnorm=1.0)
    loss = keras.losses.BinaryCrossentropy(label_smoothing=float(label_smooth))
    model.compile(
        optimizer=opt, loss=loss,
        metrics=[keras.metrics.AUC(name="pr_auc", curve="PR"),
                 keras.metrics.AUC(name="roc_auc", curve="ROC")]
    )
    return model

# --------------------------- 10-FOLD CV TRAINING -----------------------------
def train_one_model_10fold(X_df: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                           model_type: str, used_cols: List[str],
                           lat_col: Optional[str], lon_col: Optional[str],
                           out_dir: str, epochs: int = EPOCHS):
    """Train ONE model family (lstm / bilstm) with proper 10-fold spatial GroupKFold CV.
    Saves all artefacts in PSO-schema (matches src/c8c11_non_pso_cli.py)."""
    set_all_seeds(RANDOM_STATE)
    p = BASE_HPS[model_type]
    bidir = (model_type == "bilstm")
    os.makedirs(out_dir, exist_ok=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.full(len(y), np.nan, dtype=float)
    fold_metrics = []
    pre_final = None
    mask_final = None
    feat_names_final = None
    final_model = None

    cw = maybe_class_weights(y)

    for k, (tr_idx, va_idx) in enumerate(
            tqdm(gkf.split(X_df, y, groups), total=N_FOLDS,
                 desc=f"[CV {model_type.upper()}] folds", leave=True), start=1):
        Xtr_df, Xva_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        # Fit preprocessor + RFE on training rows only (leakage-safe)
        pre = build_preprocessor(Xtr_df)
        pre.fit(Xtr_df)
        mask, feat_names = run_rfe(pre, Xtr_df, ytr)

        Xtr = transform_with_mask(pre, Xtr_df, mask)
        Xva = transform_with_mask(pre, Xva_df, mask)

        # Internal val split (stratified) for ES on the training fold
        i_tr, i_va = train_test_split(
            np.arange(len(ytr)), test_size=0.15,
            random_state=RANDOM_STATE, stratify=ytr
        )

        n_features = int(Xtr.shape[1])
        model = build_lstm_or_bilstm_flatinput(
            n_features=n_features,
            units=int(p["units"]), layers_n=int(p["layers"]),
            dropout=float(p["dropout"]), recurrent_drop=float(p["recurrent_drop"]),
            spatial_drop=float(p["spatial_drop"]), l2_reg=float(p["l2"]),
            lr=float(p["lr"]), layer_norm=True, label_smooth=float(p["label_smooth"]),
            bidirectional=bidir
        )

        es = keras.callbacks.EarlyStopping(monitor="val_roc_auc", mode="max",
                                           patience=EARLYSTOP_PATIENCE,
                                           restore_best_weights=True, verbose=0)
        rl = keras.callbacks.ReduceLROnPlateau(monitor="val_roc_auc", mode="max",
                                               factor=0.5, patience=PLATEAU_PATIENCE,
                                               min_lr=MIN_LR, verbose=0)

        model.fit(
            Xtr[i_tr], ytr[i_tr],
            epochs=epochs, batch_size=int(p["batch_size"]),
            validation_data=(Xtr[i_va], ytr[i_va]),
            callbacks=[es, rl], verbose=0,
            class_weight=(cw if cw is not None else None)
        )

        yhat = model.predict(Xva, batch_size=int(p["batch_size"]), verbose=0).ravel()
        oof_pred[va_idx] = yhat
        pr = float(average_precision_score(yva, yhat))
        try:
            roc = float(roc_auc_score(yva, yhat))
        except Exception:
            roc = float("nan")
        fold_metrics.append({"fold": int(k), "pr_auc": pr, "roc_auc": roc})
        print(f"    fold {k:>2}/{N_FOLDS}  pr_auc={pr:.4f}  roc_auc={roc:.4f}")

        # Keep last fold's artefacts as the saved final model (matches CLI behaviour)
        pre_final = pre
        mask_final = mask
        feat_names_final = feat_names
        final_model = model

    # ---- Save outputs in PSO-schema ----
    pd.DataFrame(fold_metrics).to_csv(
        os.path.join(out_dir, "cv_metrics_10fold.csv"), index=False)
    pd.DataFrame({"index": np.arange(len(y)), "y_true": y,
                  "y_pred_oof": oof_pred}).to_csv(
        os.path.join(out_dir, "cv_oof_predictions.csv"), index=False)
    cv_mean = {
        "pr_auc":  float(np.nanmean([m["pr_auc"] for m in fold_metrics])),
        "roc_auc": float(np.nanmean([m["roc_auc"] for m in fold_metrics])),
    }
    valid = ~np.isnan(oof_pred)
    cv_oof = {
        "pr_auc":  float(average_precision_score(y[valid], oof_pred[valid])),
        "roc_auc": float(roc_auc_score(y[valid], oof_pred[valid])),
    }
    summary = {
        "cv_mean": cv_mean,
        "cv_oof":  cv_oof,
        "final_holdout": {"pr_auc": cv_oof["pr_auc"], "roc_auc": cv_oof["roc_auc"]},
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump({"hps": p, "note": "non-PSO baseline, fixed hyperparameters, 10-fold spatial GroupKFold"},
                  f, indent=2)
    if final_model is not None:
        final_model.save(os.path.join(out_dir, "final_model.keras"))
    if pre_final is not None:
        joblib.dump(pre_final, os.path.join(out_dir, "static_preprocessor.joblib"))
        pd.DataFrame({"selected_feature": feat_names_final}).to_csv(
            os.path.join(out_dir, "selected_features_final.csv"), index=False)
    n_features_after_RFE = int(mask_final.sum()) if mask_final is not None else None
    with open(os.path.join(out_dir, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"raw_columns": used_cols, "target": TARGET_NAME,
                   "lat_col": lat_col, "lon_col": lon_col,
                   "grid_km": GRID_KM,
                   "n_features_after_RFE": n_features_after_RFE}, f, indent=2)
    print(f"[CV-OOF] {model_type.upper()}  ROC-AUC={cv_oof['roc_auc']:.4f}  "
          f"PR-AUC={cv_oof['pr_auc']:.4f}")
    return summary

# --------------------------- MAIN TRAIN & SAVE -------------------------------
def main():
    set_all_seeds()

    # paths & data
    data_path, out_root = ask_paths()
    df = read_table(data_path)
    df = df.replace([-9999, -99999, 9999, 99999], np.nan)

    # aspect + distance transforms (match loader preprocessing)
    if ASPECT_KEY in df.columns:
        df = add_aspect_sincos(df, ASPECT_KEY)
        print(f"[INFO] Converted '{ASPECT_KEY}' -> Aspect_sin / Aspect_cos")
    df = safe_log1p_inplace(df)

    # spatial columns (excluded from training; used for GroupKFold blocks)
    lon_name = find_col_contains(df, "long") or find_col_contains(df, "lon") or find_col_contains(df, "longitude")
    lat_name = find_col_contains(df, "lati") or find_col_contains(df, "lat") or find_col_contains(df, "latitude")
    drop_cols = [c for c in [lon_name, lat_name] if c]

    if TARGET_NAME not in df.columns:
        raise SystemExit(f"Label column '{TARGET_NAME}' not found.")

    y = pd.to_numeric(df[TARGET_NAME], errors="coerce").fillna(0).astype(int).values
    X = df.drop(columns=[TARGET_NAME] + drop_cols, errors="ignore").copy()
    used_cols = list(X.columns)
    print(f"[INFO] Feature columns used (raw): {len(used_cols)}")

    # Spatial GroupKFold groups
    if lon_name and lat_name:
        lat_all = pd.to_numeric(df[lat_name], errors="coerce").values
        lon_all = pd.to_numeric(df[lon_name], errors="coerce").values
        groups = compute_block_ids(lat_all, lon_all, GRID_KM)
        print(f"[INFO] Spatial groups: {len(np.unique(groups))} unique blocks (grid_km={GRID_KM})")
    else:
        raise SystemExit("Latitude/Longitude columns required for 10-fold spatial GroupKFold CV.")

    # Train each model family with proper 10-fold spatial GroupKFold CV
    for model_type in RUNS:
        model_dir = os.path.join(out_root, model_type.upper())
        os.makedirs(model_dir, exist_ok=True)
        print(f"\n[Main] Training {model_type.upper()} ({EPOCHS} epochs x {N_FOLDS} folds)")
        train_one_model_10fold(
            X_df=X, y=y, groups=groups,
            model_type=model_type, used_cols=used_cols,
            lat_col=lat_name, lon_col=lon_name,
            out_dir=model_dir, epochs=EPOCHS,
        )

    print("\n[DONE] Per-model folders ready (10-fold spatial CV artefacts):\n"
          " - LSTM/\n - BILSTM/\n"
          "Each contains: final_model.keras, static_preprocessor.joblib, "
          "selected_features_final.csv, feature_meta.json, "
          "cv_metrics_10fold.csv, cv_oof_predictions.csv, "
          "metrics_summary.json, best_params.json.")

if __name__ == "__main__":
    main()
