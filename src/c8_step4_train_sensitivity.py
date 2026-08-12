import os
import sys
import json
import random
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Tuple, List, Optional, Dict
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                              f1_score, precision_score, recall_score,
                              confusion_matrix, brier_score_loss)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
tf.get_logger().setLevel(logging.ERROR)
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
TARGET_NAME = "Status"
ASPECT_KEY  = "Aspect"
TEST_SIZE   = 0.30
EPOCHS = 30
EARLYSTOP_PATIENCE = 8
PLATEAU_PATIENCE   = 3
MIN_LR = 1e-6
RFE_KEEP_FRACTION = 0.60
RFE_MIN_FEATURES  = 12
RFE_N_ESTIMATORS  = 350
USE_CLASS_WEIGHTS = "auto"
BASE_HPS = {
    "lstm":   dict(units=256, layers=1, dropout=0.20, recurrent_drop=0.10, l2=1e-6,
                   lr=3e-4, batch_size=64, label_smooth=0.01, spatial_drop=0.08),
    "bilstm": dict(units=320, layers=2, dropout=0.22, recurrent_drop=0.12, l2=2e-6,
                   lr=3e-4, batch_size=64, label_smooth=0.01, spatial_drop=0.12),
}
RUNS = ["lstm", "bilstm"]
BOOTSTRAP_N    = 1000
BOOTSTRAP_SEED = 42
def pick_folder(title):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=title)
    r.destroy(); return p
def confirm(title, msg):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    a = messagebox.askyesno(title, msg)
    r.destroy(); return a
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
def read_table(path):
    return pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
def add_aspect_sincos(df, col):
    a = pd.to_numeric(df[col], errors="coerce")
    rad = np.deg2rad(a % 360.0)
    df["Aspect_sin"] = np.sin(rad)
    df["Aspect_cos"] = np.cos(rad)
    return df.drop(columns=[col])
def is_distance_like(col):
    lc = col.lower()
    return "distance" in lc or lc.startswith("dist") or "_dist" in lc
def safe_log1p_inplace(df):
    for c in list(df.columns):
        if c == TARGET_NAME: continue
        if is_distance_like(c):
            s = pd.to_numeric(df[c], errors="coerce").copy()
            s[s < 0] = np.nan
            df[c] = np.log1p(s)
    return df
def clean_nodata(df):
    df = df.dropna(subset=[TARGET_NAME]).copy()
    return df.replace([-9999, -99999, 9999, 99999], np.nan)
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
def build_preprocessor(X_df):
    cat = [c for c in X_df.columns if str(X_df[c].dtype) in ("object", "category")]
    num = [c for c in X_df.columns if c not in cat]
    np_ = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cp_ = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())])
    return ColumnTransformer([("num", np_, num), ("cat", cp_, cat)], remainder="drop")
def get_feature_names(pre, X_df):
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
    names_all = get_feature_names(pre, X_tr)
    if n_keep >= n_total:
        return np.ones(n_total, dtype=bool), names_all
    rf = RandomForestClassifier(n_estimators=RFE_N_ESTIMATORS, n_jobs=-1,
                                random_state=seed, class_weight="balanced")
    rfe = RFE(estimator=rf, n_features_to_select=n_keep, step=0.1)
    rfe.fit(Xt, y_tr)
    mask = rfe.support_.astype(bool)
    return mask, names_all[mask]
def transform_with_mask(pre, X_df, mask):
    X = pre.transform(X_df)
    if hasattr(X, "toarray"): X = X.toarray()
    return X[:, mask] if mask is not None else X
def maybe_class_weights(y):
    p = float(np.mean(y == 1))
    if 0.45 <= p <= 0.55 and USE_CLASS_WEIGHTS == "auto":
        return None
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(cw[0]), 1: float(cw[1])}
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
def make_optimizer(lr):
    try:
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=1.0)
    except Exception:
        return keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
def se_block(x, ratio=8):
    d = int(x.shape[-1])
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(1, d // ratio), activation="relu")(s)
    s = layers.Dense(d, activation="sigmoid")(s)
    s = layers.Reshape((1, d))(s)
    return layers.Multiply()([x, s])
def build_model(n_features, p, bidir):
    reg = regularizers.l2(p["l2"]) if p["l2"] > 0 else None
    inp = keras.Input(shape=(n_features,))
    x = layers.Reshape((n_features, 1))(inp)
    if p["spatial_drop"] > 0:
        x = layers.SpatialDropout1D(p["spatial_drop"])(x)
    cell = layers.LSTM(p["units"], return_sequences=True,
                        dropout=p["dropout"], recurrent_dropout=p["recurrent_drop"],
                        kernel_regularizer=reg)
    x = (layers.Bidirectional(cell) if bidir else cell)(x)
    if p["layers"] == 2:
        cell2 = layers.LSTM(p["units"], return_sequences=True,
                             dropout=p["dropout"], recurrent_dropout=p["recurrent_drop"],
                             kernel_regularizer=reg)
        x = (layers.Bidirectional(cell2) if bidir else cell2)(x)
    x = se_block(x)
    avg = layers.GlobalAveragePooling1D()(x)
    mx  = layers.GlobalMaxPooling1D()(x)
    att = AttentionPool1D(attn_units=max(64, p["units"] // 2))(x)
    x = layers.Concatenate()([avg, mx, att])
    x = layers.LayerNormalization()(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)(x)
    m = keras.Model(inp, out, name=("BiLSTM" if bidir else "LSTM"))
    m.compile(optimizer=make_optimizer(p["lr"]),
              loss=keras.losses.BinaryCrossentropy(label_smoothing=p["label_smooth"]),
              metrics=[keras.metrics.AUC(name="auc", curve="ROC")])
    return m
def calc_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "AUC":         float(roc_auc_score(y_true, y_prob)),
        "AP":          float(average_precision_score(y_true, y_prob)),
        "Brier":       float(brier_score_loss(y_true, y_prob)),
        "Accuracy":    float(accuracy_score(y_true, y_pred)),
        "F1":          float(f1_score(y_true, y_pred)),
        "Precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "NPV":         float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
def bootstrap_auc_ci(y_true, y_prob):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y_true); aucs = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2: continue
        aucs.append(roc_auc_score(yt, yp))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
def train_one_seed(seed, data_path, out_root):
    set_all_seeds(seed)
    print(f"\n========== seed {seed}  |  {os.path.basename(data_path)} ==========")
    df = read_table(data_path)
    df = clean_nodata(df)
    if ASPECT_KEY in df.columns:
        df = add_aspect_sincos(df, ASPECT_KEY)
    df = safe_log1p_inplace(df)
    if TARGET_NAME not in df.columns:
        raise ValueError(f"'{TARGET_NAME}' not in {data_path}")
    y = pd.to_numeric(df[TARGET_NAME], errors="coerce").fillna(0).astype(int).values
    DROP_EXACT = {TARGET_NAME, "Longitude", "Latitude", "UniqueID",
                  "OBJECTID", "FID", "OID", "pointid", "CID", "CID_"}
    X = df.drop(columns=[c for c in df.columns if c in DROP_EXACT],
                errors="ignore").copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="ignore")
    print(f"  rows={len(df)}  raw_features={len(X.columns)}  "
          f"fire={(y==1).sum()}  nonfire={(y==0).sum()}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              random_state=seed, stratify=y)
    out = []
    for mt in RUNS:
        p = BASE_HPS[mt]; bidir = (mt == "bilstm")
        mdir = os.path.join(out_root, f"seed_{seed}", mt.upper())
        os.makedirs(mdir, exist_ok=True)
        pre = build_preprocessor(X_tr); pre.fit(X_tr)
        mask, feats = run_rfe(pre, X_tr, y_tr, seed)
        Xtr = transform_with_mask(pre, X_tr, mask)
        Xte = transform_with_mask(pre, X_te, mask)
        cw = maybe_class_weights(y_tr)
        i_tr, i_va = train_test_split(np.arange(len(y_tr)), test_size=0.15,
                                       random_state=seed, stratify=y_tr)
        model = build_model(int(Xtr.shape[1]), p, bidir)
        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                          patience=EARLYSTOP_PATIENCE,
                                          restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                               factor=0.5, patience=PLATEAU_PATIENCE,
                                               min_lr=MIN_LR, verbose=0),
        ]
        model.fit(Xtr[i_tr], y_tr[i_tr], epochs=EPOCHS, batch_size=p["batch_size"],
                  validation_data=(Xtr[i_va], y_tr[i_va]),
                  callbacks=cbs, verbose=0, class_weight=cw)
        y_prob = model.predict(Xte, batch_size=p["batch_size"], verbose=0).ravel()
        m = calc_metrics(y_te, y_prob)
        lo, hi = bootstrap_auc_ci(y_te, y_prob)
        m["AUC_CI_lo"], m["AUC_CI_hi"] = lo, hi
        out.append({"seed": seed, "model": mt.upper(), "n_total": len(df),
                    "n_train": len(y_tr), "n_test": len(y_te),
                    "n_features_after_RFE": int(Xtr.shape[1]), **m})
        print(f"  [{mt.upper():6s}]  AUC={m['AUC']:.4f}  [{lo:.4f},{hi:.4f}]  "
              f"Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  Brier={m['Brier']:.4f}")
        model.save(os.path.join(mdir, "final_model.keras"))
        joblib.dump(pre, os.path.join(mdir, "static_preprocessor.joblib"))
        pd.DataFrame({"selected_feature": feats}).to_csv(
            os.path.join(mdir, "selected_features_final.csv"), index=False)
        with open(os.path.join(mdir, "feature_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "model": mt.upper(),
                       "raw_columns": list(X.columns), "target": TARGET_NAME,
                       "test_size": TEST_SIZE,
                       "n_features_after_RFE": int(Xtr.shape[1])}, f, indent=2)
        pd.DataFrame({"y_true": y_te, "y_prob": y_prob,
                      "y_pred": (y_prob >= 0.5).astype(int)}).to_csv(
            os.path.join(mdir, "holdout_predictions.csv"), index=False)
    return out
def main():
    print("=" * 60)
    print("STEP 4 / 5: Random-seed sensitivity training")
    print("=" * 60)
    print()
    print("Dialog 1 of 2: Pick Wildfire_C8\\02_Seed_Tables folder")
    print("(the one containing training_seed_42.csv ... training_seed_909.csv)")
    seed_dir = pick_folder("STEP 4 dialog 1: pick Wildfire_C8\\02_Seed_Tables")
    if not seed_dir:
        print("CANCELLED."); sys.exit(1)
    print(f"  seed tables: {seed_dir}")
    found = []
    for s in SEEDS:
        f = os.path.join(seed_dir, f"training_seed_{s}.csv")
        if os.path.exists(f): found.append((s, f))
    if not found:
        print(f"ERROR: no training_seed_*.csv files in {seed_dir}")
        print("Did you finish STEP 3?")
        sys.exit(2)
    print(f"  found {len(found)} / {len(SEEDS)} seed CSVs.")
    print("\nDialog 2 of 2: Pick Wildfire_C8\\03_Model_Results output folder")
    out_root = pick_folder("STEP 4 dialog 2: pick Wildfire_C8\\03_Model_Results")
    if not out_root:
        print("CANCELLED."); sys.exit(1)
    print(f"  output: {out_root}")
    if not confirm(
        "Ready to run STEP 4?",
        f"Will train LSTM + BiLSTM on {len(found)} seed datasets.\n\n"
        f"Total wall-time: ~{len(found) * 2 * 4} min on a single GPU.\n\nProceed?"
    ):
        sys.exit(0)
    os.makedirs(out_root, exist_ok=True)
    all_rows = []
    for s, p in found:
        all_rows.extend(train_one_seed(s, p, out_root))
    if not all_rows:
        print("No results produced."); return
    full = pd.DataFrame(all_rows)
    full.to_excel(os.path.join(out_root, "random_seed_sensitivity_all_results.xlsx"),
                  index=False)
    full.to_csv(os.path.join(out_root, "random_seed_sensitivity_all_results.csv"),
                index=False)
    METRICS = ["AUC", "AP", "Brier", "Accuracy", "F1",
               "Precision", "Sensitivity", "Specificity", "NPV"]
    rows = []
    for mdl, g in full.groupby("model"):
        r = {"model": mdl, "n_seeds": len(g)}
        for m in METRICS:
            r[f"{m}_mean"] = float(g[m].mean())
            r[f"{m}_SD"]   = float(g[m].std())
            r[f"{m}_min"]  = float(g[m].min())
            r[f"{m}_max"]  = float(g[m].max())
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_excel(os.path.join(out_root, "random_seed_sensitivity_clean_summary.xlsx"),
                     index=False)
    summary.to_csv(os.path.join(out_root, "random_seed_sensitivity_clean_summary.csv"),
                   index=False)
    print("\n" + "=" * 60)
    print("STEP 4 DONE.")
    print("=" * 60)
    print(f"\nAll results:    {out_root}\\random_seed_sensitivity_all_results.xlsx")
    print(f"Clean summary:  {out_root}\\random_seed_sensitivity_clean_summary.xlsx")
    print("\nHeadline:")
    cols = ["model"] + [f"{m}_mean" for m in ["AUC", "AP", "Accuracy", "F1"]] \
                    + [f"{m}_SD"   for m in ["AUC", "AP", "Accuracy", "F1"]]
    print(summary[cols].to_string(index=False))
    print("\nNext: run STEP 5 to compute the three assumption checks:")
    print("  python src/c8_step5_assumption_checks.py")
    try:
        a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
        messagebox.showinfo(
            "STEP 4 complete",
            "Random-seed sensitivity training done.\n\nNext: STEP 5.",
        )
        a.destroy()
    except Exception:
        pass
if __name__ == "__main__":
    main()
