# -*- coding: utf-8 -*-
"""
c8c11_non_pso_cli.py
====================
CLI wrapper around your non-PSO LSTM/BiLSTM training, evaluated under the
same 10-fold spatial-block GroupKFold protocol as the PSO scripts so all
four models can be compared on equal footing.

Trains BOTH LSTM and BiLSTM (no PSO) from one CSV. Saves outputs in the
SAME schema as your PSO scripts so c8c11_step4_collect_and_summarize.py
picks them up automatically:

    <base_out_dir>/LSTM/thr<H>/seed<S>/
        final_model.keras
        static_preprocessor.joblib
        selected_features_final.csv
        cv_metrics_10fold.csv
        cv_oof_predictions.csv
        metrics_summary.json
        best_params.json     <-- empty {}, kept for collector compatibility
        feature_meta.json

    <base_out_dir>/BiLSTM/thr<H>/seed<S>/
        (same files)

Usage (called by c8c11_step3_run_all_models_sensitivity.py):

    python c8c11_non_pso_cli.py --data X.csv --base_out_dir Y \
        --thr 70 --seed 42 --epochs 30 --grid_km 50
"""
import os
import sys
import math
import json
import random
import argparse
import logging
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from typing import Tuple, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, brier_score_loss)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
tf.get_logger().setLevel(logging.ERROR)

TARGET = "Status"
ASPECT = "Aspect"
N_FOLDS = 10
EARLYSTOP_PATIENCE = 8
PLATEAU_PATIENCE = 3
MIN_LR = 1e-6
RFE_KEEP_FRACTION = 0.60
RFE_MIN_FEATURES = 12
RFE_N_ESTIMATORS = 400

# Same hyperparameters as your non-PSO BASE_HPS
HPS = {
    "lstm":   dict(units=256, layers_n=1, dropout=0.20, recurrent_drop=0.10,
                   spatial_drop=0.08, l2=1e-6, lr=3e-4, batch_size=64,
                   label_smooth=0.01, bidirectional=False),
    "bilstm": dict(units=320, layers_n=2, dropout=0.22, recurrent_drop=0.12,
                   spatial_drop=0.12, l2=2e-6, lr=3e-4, batch_size=64,
                   label_smooth=0.01, bidirectional=True),
}

# -------- repro --------
def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

# -------- data --------
def add_aspect_sincos(df, col):
    a = pd.to_numeric(df[col], errors="coerce")
    rad = np.deg2rad(a % 360.0)
    df["Aspect_sin"] = np.sin(rad); df["Aspect_cos"] = np.cos(rad)
    return df.drop(columns=[col])

def is_distance_like(c):
    lc = c.lower()
    return "distance" in lc or lc.startswith("dist") or "_dist" in lc

def safe_log1p(df):
    for c in list(df.columns):
        if c == TARGET: continue
        if is_distance_like(c):
            s = pd.to_numeric(df[c], errors="coerce").copy()
            s[s < 0] = np.nan
            df[c] = np.log1p(s)
    return df

def compute_block_ids(lat, lon, grid_km=50.0):
    """Same as PSO scripts: WGS84 degree-binning into ~grid_km cells."""
    lat = np.asarray(lat, dtype=float); lon = np.asarray(lon, dtype=float)
    lat_mean = np.nanmean(lat) if np.isfinite(np.nanmean(lat)) else 45.0
    deg_lat = grid_km / 111.0
    deg_lon = grid_km / (111.0 * max(0.2, math.cos(math.radians(lat_mean))))
    gx = np.floor(lon / deg_lon).astype(np.int64)
    gy = np.floor(lat / deg_lat).astype(np.int64)
    gx_off = gx - gx.min()
    return (gy * (gx_off.max() + 1) + gx_off).astype(np.int64)

# -------- preprocessor + RFE --------
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(X_df):
    cat = [c for c in X_df.columns if str(X_df[c].dtype) in ("object", "category")]
    num = [c for c in X_df.columns if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("ohe", make_ohe())]), cat),
    ], remainder="drop")

def get_feat_names(pre, X_df):
    try:
        return pre.get_feature_names_out(X_df.columns)
    except Exception:
        Xt = pre.transform(X_df.iloc[:1])
        if hasattr(Xt, "toarray"): Xt = Xt.toarray()
        return np.array([f"f{i}" for i in range(Xt.shape[1])])

def run_rfe(pre, X_tr, y_tr, seed):
    Xt = pre.transform(X_tr)
    if hasattr(Xt, "toarray"): Xt = Xt.toarray()
    n_total = Xt.shape[1]
    n_keep = max(RFE_MIN_FEATURES, int(round(n_total * RFE_KEEP_FRACTION)))
    n_keep = min(n_keep, n_total)
    names_all = get_feat_names(pre, X_tr)
    if n_keep >= n_total:
        return np.ones(n_total, dtype=bool), names_all
    rf = RandomForestClassifier(n_estimators=RFE_N_ESTIMATORS, n_jobs=-1,
                                random_state=seed, class_weight="balanced")
    rfe = RFE(estimator=rf, n_features_to_select=n_keep, step=0.1)
    rfe.fit(Xt, y_tr)
    mask = rfe.support_.astype(bool)
    return mask, names_all[mask]

def transform_mask(pre, X_df, mask):
    X = pre.transform(X_df)
    if hasattr(X, "toarray"): X = X.toarray()
    return X[:, mask] if mask is not None else X

# -------- model (same architecture as your non-PSO script) --------
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

def se_block(x, ratio=8):
    d = int(x.shape[-1])
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(1, d // ratio), activation="relu")(s)
    s = layers.Dense(d, activation="sigmoid")(s)
    s = layers.Reshape((1, d))(s)
    return layers.Multiply()([x, s])

def make_optimizer(lr):
    try:
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=1.0)
    except Exception:
        return keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

def build_model(n_features, p):
    reg = regularizers.l2(p["l2"]) if p["l2"] > 0 else None
    inp = keras.Input(shape=(n_features,))
    x = layers.Reshape((n_features, 1))(inp)
    if p["spatial_drop"] > 0:
        x = layers.SpatialDropout1D(p["spatial_drop"])(x)
    def block():
        cell = layers.LSTM(p["units"], return_sequences=True,
                            dropout=p["dropout"], recurrent_dropout=p["recurrent_drop"],
                            kernel_regularizer=reg)
        return layers.Bidirectional(cell) if p["bidirectional"] else cell
    x = block()(x)
    if p["layers_n"] == 2:
        x = block()(x)
    x = se_block(x)
    avg = layers.GlobalAveragePooling1D()(x)
    mx  = layers.GlobalMaxPooling1D()(x)
    att = AttentionPool1D(attn_units=max(64, p["units"] // 2))(x)
    x = layers.Concatenate()([avg, mx, att])
    x = layers.LayerNormalization()(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)(x)
    m = keras.Model(inp, out, name=("BiLSTM" if p["bidirectional"] else "LSTM"))
    m.compile(optimizer=make_optimizer(p["lr"]),
              loss=keras.losses.BinaryCrossentropy(label_smoothing=p["label_smooth"]),
              metrics=[keras.metrics.AUC(name="pr_auc", curve="PR"),
                       keras.metrics.AUC(name="roc_auc", curve="ROC")])
    return m

# -------- one-model 10-fold CV --------
def train_one_model(X_df, y, groups, model_type, seed, out_dir, epochs):
    set_all_seeds(seed)
    p = HPS[model_type]
    os.makedirs(out_dir, exist_ok=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.full(len(y), np.nan, dtype=float)
    fold_metrics = []
    pre_final = None; mask_final = None; feat_names_final = None
    final_model = None

    cw = None
    if len(np.unique(y)) == 2:
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        cw = {0: float(w[0]), 1: float(w[1])}

    for k, (tr_idx, va_idx) in enumerate(gkf.split(X_df, y, groups), start=1):
        Xtr_df, Xva_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        pre = build_preprocessor(Xtr_df); pre.fit(Xtr_df)
        mask, feat_names = run_rfe(pre, Xtr_df, ytr, seed)
        Xtr = transform_mask(pre, Xtr_df, mask)
        Xva = transform_mask(pre, Xva_df, mask)
        i_tr, i_va = train_test_split(np.arange(len(ytr)), test_size=0.15,
                                       random_state=seed, stratify=ytr)
        model = build_model(int(Xtr.shape[1]), p)
        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_roc_auc", mode="max",
                                          patience=EARLYSTOP_PATIENCE,
                                          restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_roc_auc", mode="max",
                                               factor=0.5, patience=PLATEAU_PATIENCE,
                                               min_lr=MIN_LR, verbose=0),
        ]
        model.fit(Xtr[i_tr], ytr[i_tr], epochs=epochs, batch_size=p["batch_size"],
                  validation_data=(Xtr[i_va], ytr[i_va]),
                  callbacks=cbs, verbose=0, class_weight=cw)
        yhat = model.predict(Xva, batch_size=p["batch_size"], verbose=0).ravel()
        oof_pred[va_idx] = yhat
        pr = float(average_precision_score(yva, yhat))
        try:    roc = float(roc_auc_score(yva, yhat))
        except: roc = float("nan")
        fold_metrics.append({"fold": int(k), "pr_auc": pr, "roc_auc": roc})
        print(f"    fold {k:>2}/{N_FOLDS}  pr_auc={pr:.4f}  roc_auc={roc:.4f}")
        pre_final = pre; mask_final = mask; feat_names_final = feat_names
        final_model = model

    # ---- Save outputs in PSO-schema ----
    pd.DataFrame(fold_metrics).to_csv(os.path.join(out_dir, "cv_metrics_10fold.csv"),
                                       index=False)
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
        json.dump({"hps": p, "note": "non-PSO baseline, fixed hyperparameters"}, f, indent=2)
    if final_model is not None:
        final_model.save(os.path.join(out_dir, "final_model.keras"))
    if pre_final is not None:
        joblib.dump(pre_final, os.path.join(out_dir, "static_preprocessor.joblib"))
        pd.DataFrame({"selected_feature": feat_names_final}).to_csv(
            os.path.join(out_dir, "selected_features_final.csv"), index=False)
    n_features_after_RFE = int(mask_final.sum()) if mask_final is not None else None
    with open(os.path.join(out_dir, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"raw_columns": list(X_df.columns), "target": TARGET,
                   "n_features_after_RFE": n_features_after_RFE}, f, indent=2)
    return summary

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Non-PSO LSTM/BiLSTM CLI for C8+C11 sensitivity.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--base_out_dir", required=True,
                    help="root output dir; creates LSTM/thr<H>/seed<S>/ and BiLSTM/thr<H>/seed<S>/ inside")
    ap.add_argument("--thr", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--grid_km", type=float, default=50.0)
    ap.add_argument("--models", default="lstm,bilstm",
                    help="comma-separated; subset of lstm,bilstm")
    args = ap.parse_args()

    print(f"[non-PSO CLI] data: {args.data}")
    print(f"[non-PSO CLI] thr={args.thr}  seed={args.seed}  epochs={args.epochs}")
    df = pd.read_csv(args.data) if args.data.lower().endswith(".csv") \
         else pd.read_excel(args.data)
    df = df.dropna(subset=[TARGET]).replace([-9999, -99999, 9999, 99999], np.nan)
    if ASPECT in df.columns:
        df = add_aspect_sincos(df, ASPECT)
    df = safe_log1p(df)
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int).values

    DROP = {TARGET, "Longitude", "Latitude", "UniqueID",
            "OBJECTID", "FID", "OID", "pointid", "CID", "CID_"}
    lat = df["Latitude"].values if "Latitude" in df.columns else None
    lon = df["Longitude"].values if "Longitude" in df.columns else None
    X = df.drop(columns=[c for c in df.columns if c in DROP], errors="ignore").copy()
    for c in X.columns:
        try:
            X[c] = pd.to_numeric(X[c])
        except (ValueError, TypeError):
            pass  # leave as-is if not convertible

    if lat is None or lon is None:
        print("ERROR: no Longitude/Latitude in CSV; cannot build spatial groups.")
        sys.exit(2)
    groups = compute_block_ids(lat, lon, grid_km=args.grid_km)
    print(f"[non-PSO CLI] rows={len(df)}  fire={(y==1).sum()}  nonfire={(y==0).sum()}  "
          f"unique groups={len(np.unique(groups))}")

    wanted = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    NAME_MAP = {"lstm": "LSTM", "bilstm": "BiLSTM"}  # cross-platform-safe casing
    for mt in wanted:
        name = NAME_MAP.get(mt, mt.upper())
        out_dir = os.path.join(args.base_out_dir, name,
                                f"thr{args.thr}", f"seed{args.seed}")
        summary_json = os.path.join(out_dir, "metrics_summary.json")
        if os.path.exists(summary_json):
            print(f"  SKIP {name}: already done -> {summary_json}")
            continue
        print(f"\n  Training {name} ({args.epochs} epochs x {N_FOLDS} folds) ...")
        s = train_one_model(X, y, groups, mt, args.seed, out_dir, args.epochs)
        print(f"  {name} cv_oof_roc={s['cv_oof']['roc_auc']:.4f}  "
              f"cv_oof_pr={s['cv_oof']['pr_auc']:.4f}")
    print("\n[non-PSO CLI] done.")

if __name__ == "__main__":
    main()
