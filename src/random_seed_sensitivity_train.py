import os
import json
import random
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                              f1_score, precision_score, recall_score, confusion_matrix,
                              brier_score_loss)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
tf.get_logger().setLevel(logging.ERROR)
SEED_FILES_FOLDER = r"G:\Wildfire_RandomSeed_Sensitivity\02_Seed_Tables"
OUT_ROOT          = r"G:\Wildfire_RandomSeed_Sensitivity\03_Model_Results"
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
TARGET_NAME = "Status"
ASPECT_KEY  = "Aspect"
TEST_SIZE   = 0.30
EPOCHS             = 30
EARLYSTOP_PATIENCE = 8
PLATEAU_PATIENCE   = 3
MIN_LR             = 1e-6
RFE_KEEP_FRACTION = 0.60
RFE_MIN_FEATURES  = 12
RFE_N_ESTIMATORS  = 350
RFE_MAX_DEPTH     = None
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
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)
def add_aspect_sincos(df: pd.DataFrame, col: str) -> pd.DataFrame:
    nuniq = df[col].nunique(dropna=True)
    if nuniq < 10:
        print(f"[WARN] '{col}' has only {nuniq} unique values; check if categorical.")
    a   = pd.to_numeric(df[col], errors="coerce")
    rad = np.deg2rad(a % 360.0)
    df["Aspect_sin"] = np.sin(rad)
    df["Aspect_cos"] = np.cos(rad)
    return df.drop(columns=[col])
def is_distance_like(col: str) -> bool:
    lc = col.lower()
    return ("distance" in lc) or lc.startswith("dist") or "_dist" in lc
def safe_log1p_inplace(df: pd.DataFrame):
    for c in list(df.columns):
        if c == TARGET_NAME:
            continue
        if is_distance_like(c):
            s = pd.to_numeric(df[c], errors="coerce").copy()
            s[s < 0] = np.nan
            df[c] = np.log1p(s)
    return df
def clean_nodata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=[TARGET_NAME]).copy()
    df = df.replace([-9999, -99999, 9999, 99999], np.nan)
    return df
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
def build_preprocessor(X_df: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X_df.columns if str(X_df[c].dtype) in ("object", "category")]
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler",  StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe",     make_ohe())])
    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)], remainder="drop")
def get_feature_names(pre: ColumnTransformer, X_df: pd.DataFrame) -> np.ndarray:
    try:
        return pre.get_feature_names_out(X_df.columns)
    except Exception:
        sample = X_df.iloc[:1]
        Xt = pre.transform(sample)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        return np.array([f"f{i}" for i in range(Xt.shape[1])])
def run_rfe(pre: ColumnTransformer, X_tr_df: pd.DataFrame, y_tr: np.ndarray,
            seed: int) -> Tuple[np.ndarray, np.ndarray]:
    Xt = pre.transform(X_tr_df)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    n_total = Xt.shape[1]
    n_keep  = max(RFE_MIN_FEATURES, int(round(n_total * RFE_KEEP_FRACTION)))
    n_keep  = min(n_keep, n_total)
    names_all = get_feature_names(pre, X_tr_df)
    if n_keep >= n_total:
        return np.ones(n_total, dtype=bool), names_all
    rf = RandomForestClassifier(n_estimators=RFE_N_ESTIMATORS, max_depth=RFE_MAX_DEPTH,
                                n_jobs=-1, random_state=seed, class_weight="balanced")
    rfe = RFE(estimator=rf, n_features_to_select=n_keep, step=0.1)
    rfe.fit(Xt, y_tr)
    mask = rfe.support_.astype(bool)
    return mask, names_all[mask]
def transform_with_mask(pre: ColumnTransformer, X_df: pd.DataFrame,
                        mask: Optional[np.ndarray]) -> np.ndarray:
    X = pre.transform(X_df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X[:, mask] if mask is not None else X
def maybe_class_weights(y: np.ndarray):
    if USE_CLASS_WEIGHTS is True:
        cw = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
        return {0: float(cw[0]), 1: float(cw[1])}
    if USE_CLASS_WEIGHTS is False:
        return None
    p = float(np.mean(y == 1))
    if 0.45 <= p <= 0.55:
        return None
    cw = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
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
def build_lstm_or_bilstm(n_features, units, layers_n, dropout, recurrent_drop, spatial_drop,
                         l2_reg, lr, label_smooth, bidirectional):
    reg = regularizers.l2(l2_reg) if (l2_reg and l2_reg > 0) else None
    inp = keras.Input(shape=(n_features,), name="x")
    x = layers.Reshape((n_features, 1))(inp)
    if spatial_drop > 0:
        x = layers.SpatialDropout1D(spatial_drop)(x)
    def block():
        cell = layers.LSTM(units, return_sequences=True,
                           dropout=dropout, recurrent_dropout=recurrent_drop,
                           kernel_regularizer=reg)
        return layers.Bidirectional(cell) if bidirectional else cell
    x = block()(x)
    if layers_n == 2:
        x = block()(x)
    x = se_block_timewise(x, ratio=8)
    avg = layers.GlobalAveragePooling1D()(x)
    mx  = layers.GlobalMaxPooling1D()(x)
    att = AttentionPool1D(attn_units=max(64, units // 2))(x)
    x   = layers.Concatenate()([avg, mx, att])
    x   = layers.LayerNormalization()(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)(x)
    model = keras.Model(inp, out, name=("BiLSTM" if bidirectional else "LSTM"))
    model.compile(optimizer=make_optimizer(lr, clipnorm=1.0),
                  loss=keras.losses.BinaryCrossentropy(label_smoothing=label_smooth),
                  metrics=[keras.metrics.AUC(name="auc", curve="ROC")])
    return model
def calc_metrics(y_true, y_prob, thr=0.5) -> Dict[str, float]:
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
def bootstrap_auc_ci(y_true, y_prob, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    lo, hi = float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
    return lo, hi
def train_one_seed(seed, data_path, out_root) -> list:
    set_all_seeds(seed)
    print(f"\n{'='*60}\nSeed {seed}  |  {data_path}\n{'='*60}")
    df = read_table(data_path)
    df = clean_nodata(df)
    if ASPECT_KEY in df.columns:
        df = add_aspect_sincos(df, ASPECT_KEY)
    df = safe_log1p_inplace(df)
    if TARGET_NAME not in df.columns:
        raise ValueError(f"Missing '{TARGET_NAME}' in {data_path}")
    y = pd.to_numeric(df[TARGET_NAME], errors="coerce").fillna(0).astype(int).values
    DROP_EXACT = {TARGET_NAME, "Longitude", "Latitude", "UniqueID", "OBJECTID",
                  "FID", "OID", "pointid", "CID", "CID_"}
    drop_cols = [c for c in df.columns if c in DROP_EXACT]
    X = df.drop(columns=drop_cols, errors="ignore").copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="ignore")
    print(f"  rows={len(df)}  features_raw={len(X.columns)}  "
          f"fire={int((y==1).sum())}  nonfire={int((y==0).sum())}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              random_state=seed, stratify=y)
    out_rows = []
    for model_type in RUNS:
        p     = BASE_HPS[model_type]
        bidir = (model_type == "bilstm")
        mdir  = os.path.join(out_root, f"seed_{seed}", model_type.upper())
        os.makedirs(mdir, exist_ok=True)
        pre = build_preprocessor(X_tr); pre.fit(X_tr)
        mask, feat_names = run_rfe(pre, X_tr, y_tr, seed)
        Xtr = transform_with_mask(pre, X_tr, mask)
        Xte = transform_with_mask(pre, X_te, mask)
        cw = maybe_class_weights(y_tr)
        idx_tr_sub, idx_va_sub = train_test_split(np.arange(len(y_tr)),
                                                   test_size=0.15, random_state=seed,
                                                   stratify=y_tr)
        n_features = int(Xtr.shape[1])
        model = build_lstm_or_bilstm(n_features=n_features,
                                     units=p["units"], layers_n=p["layers"],
                                     dropout=p["dropout"], recurrent_drop=p["recurrent_drop"],
                                     spatial_drop=p["spatial_drop"], l2_reg=p["l2"],
                                     lr=p["lr"], label_smooth=p["label_smooth"],
                                     bidirectional=bidir)
        es = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                            patience=EARLYSTOP_PATIENCE,
                                            restore_best_weights=True, verbose=0)
        rl = keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                                factor=0.5, patience=PLATEAU_PATIENCE,
                                                min_lr=MIN_LR, verbose=0)
        model.fit(Xtr[idx_tr_sub], y_tr[idx_tr_sub], epochs=EPOCHS,
                  batch_size=p["batch_size"],
                  validation_data=(Xtr[idx_va_sub], y_tr[idx_va_sub]),
                  callbacks=[es, rl], verbose=0,
                  class_weight=cw if cw is not None else None)
        y_prob = model.predict(Xte, batch_size=p["batch_size"], verbose=0).ravel()
        m = calc_metrics(y_te, y_prob, thr=0.5)
        auc_lo, auc_hi = bootstrap_auc_ci(y_te, y_prob)
        m["AUC_CI_lo"] = auc_lo
        m["AUC_CI_hi"] = auc_hi
        row = {"seed": seed, "model": model_type.upper(),
               "n_total": len(df), "n_train": len(y_tr), "n_test": len(y_te),
               "n_features_after_RFE": n_features, **m}
        out_rows.append(row)
        print(f"  [{model_type.upper():6s}]  AUC={m['AUC']:.4f}  [{auc_lo:.4f},{auc_hi:.4f}]  "
              f"AP={m['AP']:.4f}  Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  "
              f"Brier={m['Brier']:.4f}")
        model.save(os.path.join(mdir, "final_model.keras"))
        joblib.dump(pre, os.path.join(mdir, "static_preprocessor.joblib"))
        pd.DataFrame({"selected_feature": feat_names}).to_csv(
            os.path.join(mdir, "selected_features_final.csv"), index=False)
        with open(os.path.join(mdir, "feature_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "model": model_type.upper(),
                       "raw_columns": list(X.columns), "target": TARGET_NAME,
                       "test_size": TEST_SIZE,
                       "n_features_after_RFE": n_features}, f, indent=2)
        pd.DataFrame({"y_true": y_te, "y_prob": y_prob,
                      "y_pred": (y_prob >= 0.5).astype(int)}).to_csv(
            os.path.join(mdir, "holdout_predictions.csv"), index=False)
    return out_rows
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    all_rows = []
    for seed in SEEDS:
        candidates = [os.path.join(SEED_FILES_FOLDER, f"training_seed_{seed}.csv"),
                      os.path.join(SEED_FILES_FOLDER, f"training_seed_{seed}.xlsx")]
        data_path = next((p for p in candidates if os.path.exists(p)), None)
        if data_path is None:
            print(f"[WARN] no file for seed {seed}; skipped"); continue
        all_rows.extend(train_one_seed(seed, data_path, OUT_ROOT))
    if not all_rows:
        print("No results. Check SEED_FILES_FOLDER."); return
    full = pd.DataFrame(all_rows)
    full.to_excel(os.path.join(OUT_ROOT, "random_seed_sensitivity_all_results.xlsx"),
                  index=False)
    full.to_csv(os.path.join(OUT_ROOT, "random_seed_sensitivity_all_results.csv"),
                index=False)
    METRIC_COLS = ["AUC", "AP", "Brier", "Accuracy", "F1",
                   "Precision", "Sensitivity", "Specificity", "NPV"]
    rows = []
    for mdl, grp in full.groupby("model"):
        r = {"model": mdl, "n_seeds": len(grp)}
        for m in METRIC_COLS:
            r[f"{m}_mean"] = float(grp[m].mean())
            r[f"{m}_SD"]   = float(grp[m].std())
            r[f"{m}_min"]  = float(grp[m].min())
            r[f"{m}_max"]  = float(grp[m].max())
            r[f"{m}_range"] = r[f"{m}_max"] - r[f"{m}_min"]
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_excel(os.path.join(OUT_ROOT, "random_seed_sensitivity_clean_summary.xlsx"),
                     index=False)
    summary.to_csv(os.path.join(OUT_ROOT, "random_seed_sensitivity_clean_summary.csv"),
                   index=False)
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  all results: {OUT_ROOT}\\random_seed_sensitivity_all_results.xlsx")
    print(f"  summary:     {OUT_ROOT}\\random_seed_sensitivity_clean_summary.xlsx")
    print("=" * 60)
    cols = ["model"] + [f"{m}_mean" for m in ["AUC", "AP", "Accuracy", "F1"]] \
                    + [f"{m}_SD"   for m in ["AUC", "AP", "Accuracy", "F1"]]
    print("\nHeadline (mean and SD across seeds):")
    print(summary[cols].to_string(index=False))
if __name__ == "__main__":
    main()
