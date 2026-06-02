# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:49:30 2025

@author: saadz
"""

# -*- coding: utf-8 -*-
"""
Wildfire Susceptibility — BiLSTM + PSO + Leakage-safe RFE
(Compact PSO for ~3k rows, 30 epochs, Spyder-friendly progress)

Adds:
  • Out-of-fold (OOF) PR_AUC and ROC_AUC after the 10-fold CV
  • Final holdout PR_AUC and ROC_AUC for the trained model
  • Saves:
      - cv_metrics_10fold.csv              (per-fold metrics)
      - cv_oof_predictions.csv             (index, y_true, y_pred_oof)
      - metrics_summary.json               (cv_mean, cv_oof, final_holdout)
      - selected_features_final.csv
      - static_preprocessor.joblib
      - best_params.json
      - final_model.keras
      - predictions_allrows.csv

Run:
  python bilstm_pso_rfe_small_tqdm.py --data "C:/path/to/data.csv" --out_dir "C:/path/to/out"
"""

import os, math, json, argparse, warnings, time
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

import joblib

import tensorflow as tf
from tensorflow import keras as K

from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ================================ Progress helpers ============================

def make_progress_callbacks(desc: str, mode: str, monitor: str, patience: int):
    """Return callbacks according to desired progress mode."""
    cbs = [K.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=patience, restore_best_weights=True)]
    if mode == "tqdm":
        try:
            from tqdm.keras import TqdmCallback
            cbs.insert(0, TqdmCallback(verbose=0, desc=desc, leave=False))
        except Exception:
            pass
    return cbs

# ================================ Utilities ==================================

def is_distance_like(col: str) -> bool:
    lc = col.lower()
    return ("distance" in lc) or lc.startswith("dist") or ("_dist" in lc) or ("_distance" in lc)

def aspect_to_sin_cos(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    rad = np.deg2rad(pd.to_numeric(s, errors="coerce"))
    return np.sin(rad), np.cos(rad)

def safe_log1p(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").copy()
    x[x < 0] = np.nan
    return np.log1p(x)

def compute_block_ids(lat: np.ndarray, lon: np.ndarray, grid_km: float = 50.0) -> np.ndarray:
    """
    Return integer group IDs for spatial CV by binning lat/lon into ~grid_km cells.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_mean = np.nanmean(lat) if np.isfinite(np.nanmean(lat)) else 45.0

    deg_lat = grid_km / 111.0
    deg_lon = grid_km / (111.0 * max(0.2, np.cos(np.deg2rad(lat_mean))))

    gx = np.floor(lon / deg_lon).astype(np.int64)
    gy = np.floor(lat / deg_lat).astype(np.int64)

    gx_off = gx - gx.min()
    groups = gy * (gx_off.max() + 1) + gx_off
    return groups.astype(np.int64)

def build_preprocessor(X_df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = [c for c in X_df.columns if str(X_df[c].dtype) in ["category", "object"]]
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except Exception:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", ohe)])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    return pre, num_cols, cat_cols

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Optional[str], Optional[str]]:
    if "Status" not in df.columns:
        raise SystemExit("Missing 'Status' column (0/1).")
    work = df.copy()

    # Aspect -> sin/cos
    if "Aspect" in work.columns:
        s, c = aspect_to_sin_cos(work["Aspect"])
        work["Aspect_sin"] = s
        work["Aspect_cos"] = c
        work.drop(columns=["Aspect"], inplace=True)

    # Log1p for distance-like columns
    for c in list(work.columns):
        if c != "Status" and is_distance_like(c):
            work[c] = safe_log1p(work[c])

    # Detect lat/lon (not used as features)
    lat_col = next((c for c in work.columns if c.lower() in ["latitude", "lat"]), None)
    lon_col = next((c for c in work.columns if c.lower() in ["longitude", "lon", "lng"]), None)

    # Choose features (exclude Status/lat/lon)
    candidates = [c for c in work.columns if c not in ["Status", lat_col, lon_col]]
    cat_like = {"lulc", "landuse", "land_cover", "landcover"}
    cat_cols = []
    for c in candidates:
        if (c.lower() in cat_like) or (work[c].dtype == "object") or str(work[c].dtype).startswith("category"):
            cat_cols.append(c)
    X_df = work[candidates].copy()
    for c in candidates:
        if c not in cat_cols:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
        else:
            X_df[c] = X_df[c].astype("category")

    used_cols = list(X_df.columns)
    return X_df, used_cols, lat_col, lon_col

# ============================ RFE (Feature Selection) =========================

def _get_feature_names(pre: ColumnTransformer, X_df: pd.DataFrame) -> np.ndarray:
    try:
        return pre.get_feature_names_out(X_df.columns)
    except Exception:
        Xt = pre.transform(X_df.iloc[:1])
        if hasattr(Xt, "toarray"): Xt = Xt.toarray()
        return np.array([f"f{i}" for i in range(Xt.shape[1])])

def run_rfe(pre: ColumnTransformer, X_tr_df: pd.DataFrame, y_tr: np.ndarray,
            keep_fraction: float = 0.6, keep_min: int = 12,
            n_estimators: int = 400, max_depth: Optional[int] = None, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit RF-RFE on preprocessed training data only; return (mask, selected_feature_names) in transformed space.
    """
    Xt = pre.transform(X_tr_df)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    n_total = Xt.shape[1]
    n_keep = max(keep_min, int(round(n_total * float(keep_fraction))))
    n_keep = min(n_keep, n_total)

    names_all = _get_feature_names(pre, X_tr_df)

    if n_keep >= n_total:
        mask = np.ones(n_total, dtype=bool)
        return mask, names_all

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced"
    )
    rfe = RFE(estimator=rf, n_features_to_select=n_keep, step=0.1)
    rfe.fit(Xt, y_tr)
    mask = rfe.support_.astype(bool)
    names_sel = names_all[mask]
    return mask, names_sel

def transform_with_mask(pre: ColumnTransformer, X_df: pd.DataFrame, mask: Optional[np.ndarray]) -> np.ndarray:
    X = pre.transform(X_df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X[:, mask] if (mask is not None) else X

# ============================ Model & Optimizer ===============================

# --- Register custom schedule so Keras can serialize it ---
try:
    register_ks = K.saving.register_keras_serializable   # Keras 3+
except Exception:
    from tensorflow.keras.utils import register_keras_serializable as register_ks

@register_ks(package="LRSchedules")
class WarmupCosine(K.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(1, warmup_steps))
        self.total_steps = int(max(self.warmup_steps + 1, total_steps))

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * step / tf.cast(self.warmup_steps, tf.float32)
        prog = (step - tf.cast(self.warmup_steps, tf.float32)) / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        prog = tf.clip_by_value(prog, 0.0, 1.0)
        cosine = 0.5 * self.base_lr * (1.0 + tf.cos(math.pi * prog))
        return tf.where(step < self.warmup_steps, warm, cosine)

    def get_config(self):
        return {"base_lr": self.base_lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def _coerce_lr(lr, default=1e-3):
    if isinstance(lr, (K.optimizers.schedules.LearningRateSchedule,)) or callable(lr):
        return lr
    try:
        return float(lr)
    except Exception:
        return float(default)

def make_optimizer(lr, clipnorm=None, weight_decay=2e-4):
    lr = _coerce_lr(lr, default=1e-3)
    try:
        return K.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, clipnorm=clipnorm)
    except Exception:
        return K.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)

def build_bilstm_cnn(n_features: int,
                     conv_filters: int,
                     kernel_size: int,
                     units1: int,
                     units2: int,
                     dropout: float,
                     sdrop: float,
                     base_lr: float,
                     weight_decay: float,
                     epochs: int,
                     steps_per_epoch: int,
                     label_smoothing: float) -> K.Model:

    warmup = max(1, int(0.05 * steps_per_epoch * epochs))
    total = max(warmup + 1, steps_per_epoch * epochs)
    lr = WarmupCosine(base_lr, warmup, total)
    opt = make_optimizer(lr, clipnorm=1.0, weight_decay=weight_decay)

    inp = K.Input(shape=(n_features,), name="x")
    x = K.layers.Reshape((n_features, 1))(inp)
    x = K.layers.Conv1D(int(conv_filters), int(kernel_size), padding="same", activation="relu")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.SpatialDropout1D(float(sdrop))(x)

    x = K.layers.Bidirectional(K.layers.LSTM(int(units1), return_sequences=(units2>0)))(x)
    if int(units2) > 0:
        x = K.layers.Bidirectional(K.layers.LSTM(int(units2)))(x)

    x = K.layers.Dropout(float(dropout))(x)
    x = K.layers.Dense(64, activation="relu")(x)
    out = K.layers.Dense(1, activation="sigmoid")(x)

    model = K.Model(inp, out, name="bilstm_cnn")
    model.compile(
        optimizer=opt,
        loss=K.losses.BinaryCrossentropy(label_smoothing=float(label_smoothing)),
        metrics=[K.metrics.AUC(curve="PR", name="pr_auc"),
                 K.metrics.AUC(curve="ROC", name="roc_auc")]
    )
    return model

# ========================= CV evaluation with RFE =============================

def fit_with_progress(model, Xtr, ytr, Xva, yva, epochs, batch, monitor, patience, class_weight, desc, progress_mode):
    callbacks = make_progress_callbacks(desc, progress_mode, monitor, patience)
    verbose = 0 if progress_mode == "tqdm" else (1 if progress_mode == "keras" else 0)
    model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=epochs,
        batch_size=batch,
        verbose=verbose,  # Spyder-friendly if "keras"
        callbacks=callbacks,
        class_weight=class_weight
    )
    return model

def evaluate_cv_rfe(X_df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, P: Dict,
                    folds: int, epochs: int, objective: str, rfe_keep_frac: float,
                    rfe_keep_min: int, rfe_estimators: int, rfe_max_depth: Optional[int],
                    progress_mode: str) -> Tuple[float, float]:
    gkf = GroupKFold(n_splits=folds)
    pr_scores, roc_scores = [], []

    # class weights
    classes = np.unique(y)
    cw = None
    if len(classes) == 2:
        w = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        cw = {int(c): float(v) for c, v in zip(classes, w)}

    monitor = "val_roc_auc" if objective == "roc" else "val_pr_auc"

    for fold_id, (tr, va) in enumerate(tqdm(gkf.split(X_df, y, groups),
                                            total=folds,
                                            desc="PSO CV folds",
                                            leave=False), start=1):
        Xtr_df, Xva_df = X_df.iloc[tr], X_df.iloc[va]
        ytr, yva = y[tr], y[va]

        pre, _, _ = build_preprocessor(Xtr_df)
        pre.fit(Xtr_df)

        mask, _ = run_rfe(pre, Xtr_df, ytr,
                          keep_fraction=rfe_keep_frac, keep_min=rfe_keep_min,
                          n_estimators=rfe_estimators, max_depth=rfe_max_depth, seed=SEED)

        Xtr = transform_with_mask(pre, Xtr_df, mask)
        Xva = transform_with_mask(pre, Xva_df, mask)

        steps = max(1, len(tr) // P["batch"])
        model = build_bilstm_cnn(
            n_features=Xtr.shape[1],
            conv_filters=P["conv_filters"],
            kernel_size=P["kernel_size"],
            units1=P["units1"],
            units2=P["units2"],
            dropout=P["dropout"],
            sdrop=P["sdrop"],
            base_lr=P["lr"],
            weight_decay=P["weight_decay"],
            epochs=epochs,
            steps_per_epoch=steps,
            label_smoothing=P["label_smoothing"]
        )
        model = fit_with_progress(
            model, Xtr, ytr, Xva, yva,
            epochs=epochs, batch=P["batch"],
            monitor=monitor, patience=6, class_weight=cw,
            desc=f"Fold {fold_id}/{folds}", progress_mode=progress_mode
        )

        y_hat = model.predict(Xva, batch_size=P["batch"], verbose=0).ravel()
        pr = float(average_precision_score(yva, y_hat))
        try:
            roc = float(roc_auc_score(yva, y_hat))
        except Exception:
            roc = float("nan")

        pr_scores.append(pr); roc_scores.append(roc)
        K.backend.clear_session()

    return float(np.nanmean(pr_scores)), float(np.nanmean(roc_scores))

# =========================== PSO objective (compact) ==========================

def make_pso_objective(X_df: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                       folds: int, epochs: int, objective: str,
                       rfe_keep_frac: float, rfe_keep_min: int,
                       rfe_estimators: int, rfe_max_depth: Optional[int],
                       progress_mode: str):
    """
    Particle vector (small-data compact bounds):
      [log10_lr, weight_decay, dropout, sdrop, units1, units2, log2_batch, conv_filters, kernel_size, label_smoothing]
    """
    def decode(vec):
        # Small-data bounds + snapping
        lr = 10 ** float(np.clip(vec[0], -3.9, -3.3))
        wd = float(np.clip(vec[1], 1e-5, 4e-4))
        dr = float(np.clip(vec[2], 0.18, 0.36))
        sdrop = float(np.clip(vec[3], 0.08, 0.25))
        u1 = int(32 * round(np.clip(vec[4], 160, 352) / 32))
        u2 = int(32 * round(np.clip(vec[5],   0, 128) / 32))
        bs = int(2 ** int(np.clip(round(vec[6]), 5, 7)))   # {32,64,128}
        cf = int(8 * round(np.clip(vec[7], 24, 64) / 8))
        ks = int(np.clip(round(vec[8]), 3, 5)); ks = 5 if ks >= 5 else 3  # {3,5}
        ls = float(np.clip(vec[9], 0.005, 0.02))
        return {"lr": lr, "weight_decay": wd, "dropout": dr, "sdrop": sdrop,
                "units1": u1, "units2": u2, "batch": bs,
                "conv_filters": cf, "kernel_size": ks, "label_smoothing": ls}

    def objective_fn(particles):
        losses = []
        # tqdm for particle batch
        for p in tqdm(particles, desc="PSO particle evals", leave=False):
            P = decode(p)
            pr_auc, roc_auc = evaluate_cv_rfe(
                X_df=X_df, y=y, groups=groups, P=P,
                folds=folds, epochs=epochs, objective=objective,
                rfe_keep_frac=rfe_keep_frac, rfe_keep_min=rfe_keep_min,
                rfe_estimators=rfe_estimators, rfe_max_depth=rfe_max_depth,
                progress_mode=progress_mode
            )
            score = roc_auc if objective == "roc" else pr_auc
            losses.append(1.0 - float(score))  # minimize
        return np.array(losses, dtype=np.float64)
    return objective_fn

# ================================== Main =====================================

def pick_file_dialog(title="Select data file (.csv/.xlsx)") -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(title=title, filetypes=[("Data files","*.csv *.xlsx *.xls"), ("All files","*.*")])
        return path or None
    except Exception:
        return None

def pick_dir_dialog(title="Select output folder") -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askdirectory(title=title)
        return path or None
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="BiLSTM + PSO + RFE (spatial CV, compact search, Spyder-friendly)")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--objective", type=str, default="roc", choices=["roc", "pr"], help="Optimize ROC AUC or PR AUC")
    parser.add_argument("--grid_km", type=float, default=50.0, help="Grid size for spatial CV blocks")

    # Compact PSO budget for ~3k rows (cuts runtime)
    parser.add_argument("--pso_particles", type=int, default=6, help="Swarm size")
    parser.add_argument("--pso_iters", type=int, default=6, help="PSO iterations")
    parser.add_argument("--pso_folds", type=int, default=2, help="Folds for PSO evaluation")

    # Epochs: 30 for both search & retrain
    parser.add_argument("--search_epochs", type=int, default=30, help="Epochs during PSO evaluation")
    parser.add_argument("--retrain_epochs", type=int, default=30, help="Epochs during final CV retraining")

    # Progress mode (Spyder users: 'keras' is most reliable)
    parser.add_argument("--progress", type=str, default="keras", choices=["tqdm","keras","none"],
                        help="Per-epoch progress style")

    # RFE defaults tuned for ~3k rows
    parser.add_argument("--rfe_fraction", type=float, default=0.60, help="RFE keep fraction (of transformed features)")
    parser.add_argument("--rfe_min_features", type=int, default=12, help="RFE minimum kept features")
    parser.add_argument("--rfe_n_estimators", type=int, default=400, help="RF trees for RFE ranking")
    parser.add_argument("--rfe_max_depth", type=int, default=None, help="RF max depth for RFE (None = unlimited)")
    args = parser.parse_args()

    # GPU mem growth
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    # Pickers if paths not provided
    if not args.data:
        path = pick_file_dialog("Select data file (.csv/.xlsx)")
        if not path: raise SystemExit("No data file selected.")
        args.data = path; print(f"[Picker] Data: {args.data}")
    if not args.out_dir:
        out = pick_dir_dialog("Select output folder")
        if not out: raise SystemExit("No output folder selected.")
        args.out_dir = out; print(f"[Picker] Output: {args.out_dir}")

    # Load
    if args.data.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.data)
    else:
        df = pd.read_csv(args.data)
    if "Status" not in df.columns:
        raise SystemExit("Expected a label column named 'Status' (0/1).")

    # Features (leakage-safe steps happen inside CV)
    X_df, used_cols, lat_col, lon_col = prepare_features(df)
    y = pd.to_numeric(df["Status"], errors="coerce").fillna(0).astype(int).values

    # Groups for spatial CV
    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce").values
        lon = pd.to_numeric(df[lon_col], errors="coerce").values
        groups = compute_block_ids(lat, lon, grid_km=args.grid_km)
    else:
        rng = np.random.default_rng(SEED)
        groups = rng.integers(0, max(5, args.pso_folds), size=len(y)).astype(np.int64)
        print("[WARN] Latitude/Longitude not found — using random groups for CV (spatial leakage not protected).")

    # Save meta
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"used_cols": used_cols, "lat_col": lat_col, "lon_col": lon_col, "grid_km": args.grid_km}, f, indent=2)

    # -------------------------- PSO (small-data bounds) -----------------------
    try:
        import pyswarms as ps
    except Exception:
        raise SystemExit("pyswarms not found. Install with: pip install pyswarms")

    # Lower/upper bounds (small-data) for each particle dim
    lb = np.array([-3.9, 1e-5, 0.18, 0.08, 160,   0, 5, 24, 3, 0.005], dtype=float)
    ub = np.array([-3.3, 4e-4, 0.36, 0.25, 352, 128, 7, 64, 5, 0.020], dtype=float)

    objective_fn = make_pso_objective(
        X_df=X_df, y=y, groups=groups,
        folds=int(args.pso_folds), epochs=int(args.search_epochs),
        objective=args.objective,
        rfe_keep_frac=float(args.rfe_fraction), rfe_keep_min=int(args.rfe_min_features),
        rfe_estimators=int(args.rfe_n_estimators), rfe_max_depth=(None if args.rfe_max_depth in [None, 0] else int(args.rfe_max_depth)),
        progress_mode=args.progress
    )

    optimizer = ps.single.GlobalBestPSO(
        n_particles=int(args.pso_particles),
        dimensions=10,
        options={"c1": 1.5, "c2": 1.5, "w": 0.7},
        bounds=(lb, ub),
        bh_strategy="periodic"
    )
    print(f"[PSO] objective={args.objective.upper()}  particles={args.pso_particles}  iters={args.pso_iters}  folds={args.pso_folds}  epochs={args.search_epochs}  rfe_fraction={args.rfe_fraction}")
    start = time.time()
    cost, pos = optimizer.optimize(objective_fn, iters=int(args.pso_iters), verbose=True)
    dur = time.time() - start
    print(f"[PSO] best loss={cost:.6f}  best {args.objective.upper()}={1.0 - cost:.6f}  time={dur/60:.1f} min")

    # Decode best particle to params
    def decode(vec):
        lr = 10 ** float(np.clip(vec[0], -3.9, -3.3))
        wd = float(np.clip(vec[1], 1e-5, 4e-4))
        dr = float(np.clip(vec[2], 0.18, 0.36))
        sdrop = float(np.clip(vec[3], 0.08, 0.25))
        u1 = int(32 * round(np.clip(vec[4], 160, 352) / 32))
        u2 = int(32 * round(np.clip(vec[5],   0, 128) / 32))
        bs = int(2 ** int(np.clip(round(vec[6]), 5, 7)))   # {32,64,128}
        cf = int(8 * round(np.clip(vec[7], 24, 64) / 8))
        ks = int(np.clip(round(vec[8]), 3, 5)); ks = 5 if ks >= 5 else 3
        ls = float(np.clip(vec[9], 0.005, 0.02))
        return {"lr": lr, "weight_decay": wd, "dropout": dr, "sdrop": sdrop,
                "units1": u1, "units2": u2, "batch": bs,
                "conv_filters": cf, "kernel_size": ks, "label_smoothing": ls}

    best_params = decode(pos)
    with open(os.path.join(args.out_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print("[BEST] params:", best_params)

    # ------------------- 10-fold spatial CV (with RFE) -----------------------
    gkf = GroupKFold(n_splits=10)
    fold_metrics = []
    # NEW: OOF container
    oof_pred = np.full(len(y), np.nan, dtype=float)

    classes = np.unique(y)
    cw = None
    if len(classes) == 2:
        w = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        cw = {int(c): float(v) for c, v in zip(classes, w)}

    monitor = "val_roc_auc" if args.objective == "roc" else "val_pr_auc"

    for k, (tr, va) in enumerate(tqdm(gkf.split(X_df, y, groups),
                                      total=10, desc="Final CV (10 folds)", leave=True), start=1):
        Xtr_df, Xva_df = X_df.iloc[tr], X_df.iloc[va]
        ytr, yva = y[tr], y[va]

        pre, _, _ = build_preprocessor(Xtr_df)
        pre.fit(Xtr_df)
        mask, feat_names = run_rfe(pre, Xtr_df, ytr,
                                   keep_fraction=float(args.rfe_fraction),
                                   keep_min=int(args.rfe_min_features),
                                   n_estimators=int(args.rfe_n_estimators),
                                   max_depth=(None if args.rfe_max_depth in [None, 0] else int(args.rfe_max_depth)),
                                   seed=SEED)

        Xtr = transform_with_mask(pre, Xtr_df, mask)
        Xva = transform_with_mask(pre, Xva_df, mask)

        steps = max(1, len(tr)//best_params["batch"])
        model = build_bilstm_cnn(
            n_features=Xtr.shape[1],
            conv_filters=best_params["conv_filters"],
            kernel_size=best_params["kernel_size"],
            units1=best_params["units1"],
            units2=best_params["units2"],
            dropout=best_params["dropout"],
            sdrop=best_params["sdrop"],
            base_lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            epochs=int(args.retrain_epochs),
            steps_per_epoch=steps,
            label_smoothing=best_params["label_smoothing"]
        )
        model = fit_with_progress(
            model, Xtr, ytr, Xva, yva,
            epochs=int(args.retrain_epochs), batch=best_params["batch"],
            monitor=monitor, patience=8, class_weight=cw,
            desc=f"CV fold {k}/10", progress_mode=args.progress
        )

        y_hat = model.predict(Xva, batch_size=best_params["batch"], verbose=0).ravel()
        pr = float(average_precision_score(yva, y_hat))
        try:
            roc = float(roc_auc_score(yva, y_hat))
        except Exception:
            roc = float("nan")
        fold_metrics.append({"fold": int(k), "pr_auc": pr, "roc_auc": roc})

        # NEW: store OOF preds
        oof_pred[va] = y_hat

        K.backend.clear_session()

    # Save per-fold metrics
    dfm = pd.DataFrame(fold_metrics)
    dfm.to_csv(os.path.join(args.out_dir, "cv_metrics_10fold.csv"), index=False)
    cv_pr_mean = float(dfm["pr_auc"].mean())
    cv_roc_mean = float(dfm["roc_auc"].mean())
    print(f"[CV10] mean PR-AUC={cv_pr_mean:.4f}  ROC-AUC={cv_roc_mean:.4f}")

    # NEW: Combined (OOF) metrics across all folds
    oof_mask = ~np.isnan(oof_pred)
    oof_pr = float(average_precision_score(y[oof_mask], oof_pred[oof_mask]))
    try:
        oof_roc = float(roc_auc_score(y[oof_mask], oof_pred[oof_mask]))
    except Exception:
        oof_roc = float("nan")
    # Save OOF predictions
    pd.DataFrame({"index": np.arange(len(y)), "y_true": y, "y_pred_oof": oof_pred}).to_csv(
        os.path.join(args.out_dir, "cv_oof_predictions.csv"), index=False
    )
    print(f"[CV10] COMBINED (OOF) PR-AUC={oof_pr:.4f}  ROC-AUC={oof_roc:.4f}")

    # ---------------- Final model (shared preproc + RFE on training rows) ----
    tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=SEED)
    Xtr_df, Xva_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    pre_final, _, _ = build_preprocessor(Xtr_df)
    pre_final.fit(Xtr_df)

    mask_final, feat_names_sel = run_rfe(
        pre_final, Xtr_df, ytr,
        keep_fraction=float(args.rfe_fraction),
        keep_min=int(args.rfe_min_features),
        n_estimators=int(args.rfe_n_estimators),
        max_depth=(None if args.rfe_max_depth in [None, 0] else int(args.rfe_max_depth)),
        seed=SEED
    )
    # Save preprocessor & selected features
    joblib.dump(pre_final, os.path.join(args.out_dir, "static_preprocessor.joblib"))
    pd.DataFrame({"selected_feature": feat_names_sel}).to_csv(
        os.path.join(args.out_dir, "selected_features_final.csv"), index=False
    )

    Xtr = transform_with_mask(pre_final, Xtr_df, mask_final)
    Xva = transform_with_mask(pre_final, Xva_df, mask_final)
    steps = max(1, len(tr_idx)//best_params["batch"])

    final_model = build_bilstm_cnn(
        n_features=Xtr.shape[1],
        conv_filters=best_params["conv_filters"],
        kernel_size=best_params["kernel_size"],
        units1=best_params["units1"],
        units2=best_params["units2"],
        dropout=best_params["dropout"],
        sdrop=best_params["sdrop"],
        base_lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        epochs=int(args.retrain_epochs),
        steps_per_epoch=steps,
        label_smoothing=best_params["label_smoothing"]
    )
    final_model = fit_with_progress(
        final_model, Xtr, ytr, Xva, yva,
        epochs=int(args.retrain_epochs), batch=best_params["batch"],
        monitor=monitor, patience=8, class_weight=cw,
        desc="Final model", progress_mode=args.progress
    )

    # Predict on all rows (transform with same preprocessor + mask)
    X_all = transform_with_mask(pre_final, X_df, mask_final)
    y_prob = final_model.predict(X_all, batch_size=best_params["batch"], verbose=0).ravel()

    # Save model & predictions
    final_model.save(os.path.join(args.out_dir, "final_model.keras"))
    out_pred = df.copy()
    out_pred["prob_wildfire"] = y_prob
    out_pred.to_csv(os.path.join(args.out_dir, "predictions_allrows.csv"), index=False)

    # NEW: final holdout metrics (on the 15% validation split)
    yva_hat = final_model.predict(Xva, batch_size=best_params["batch"], verbose=0).ravel()
    final_pr = float(average_precision_score(yva, yva_hat))
    try:
        final_roc = float(roc_auc_score(yva, yva_hat))
    except Exception:
        final_roc = float("nan")
    print(f"[FINAL] holdout PR-AUC={final_pr:.4f}  ROC-AUC={final_roc:.4f}")

    # NEW: write a single summary JSON for convenience
    summary = {
        "cv_mean": {"pr_auc": cv_pr_mean, "roc_auc": cv_roc_mean},
        "cv_oof": {"pr_auc": oof_pr, "roc_auc": oof_roc},
        "final_holdout": {"pr_auc": final_pr, "roc_auc": final_roc}
    }
    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Saved: best_params.json, cv_metrics_10fold.csv, cv_oof_predictions.csv, metrics_summary.json, final_model.keras, predictions_allrows.csv, static_preprocessor.joblib, selected_features_final.csv, feature_meta.json")

if __name__ == "__main__":
    main()
