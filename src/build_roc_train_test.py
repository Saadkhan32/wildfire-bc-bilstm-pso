import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(f"TF {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
TABLES_DIR = os.path.join(REPO_ROOT, "revision_c8c11", "03_Training_Tables")
RESULTS    = os.path.join(REPO_ROOT, "revision_c8c11", "05_Model_Results")
FIG_DIR    = os.path.join(REPO_ROOT, "revision_c8c11", "03_Figures")
os.makedirs(FIG_DIR, exist_ok=True)
CSV_70_42  = os.path.join(TABLES_DIR, "training_thr70_seed42.csv")
SEED       = 42
TEST_SIZE  = 0.15
MODELS = {
    "LSTM":       {"dir": "LSTM",       "color": "#d62728", "label": "LSTM"},
    "BiLSTM":     {"dir": "BiLSTM",     "color": "#2ca02c", "label": "BiLSTM"},
    "LSTM_PSO":   {"dir": "LSTM_PSO",   "color": "#ff7f0e", "label": "LSTM-PSO"},
    "BiLSTM_PSO": {"dir": "BiLSTM_PSO", "color": "#1f77b4", "label": "BiLSTM-PSO"},
}
def is_distance_like(c: str) -> bool:
    lc = c.lower()
    return ("distance" in lc) or lc.startswith("dist") or ("_dist" in lc) or ("_distance" in lc)
def safe_log1p(x):
    x = pd.to_numeric(x, errors="coerce").copy()
    x[x < 0] = np.nan
    return np.log1p(x)
ID_COLS = {"UniqueID","Unique_ID","unique_id",
           "OBJECTID","ObjectID","objectid","object_id",
           "FID","Fid","fid","OID","Oid","oid",
           "pointid","PointID","point_id","POINT_ID",
           "CID","CID_","Cid","cid","ID","Id","id",
           "index","Index","INDEX","row_id","RowID"}
CAT_LIKE = {"lulc","landuse","land_cover","landcover"}
def prepare_features(df_in):
    work = df_in.copy()
    if "Aspect" in work.columns:
        rad = np.deg2rad(pd.to_numeric(work["Aspect"], errors="coerce"))
        work["Aspect_sin"] = np.sin(rad)
        work["Aspect_cos"] = np.cos(rad)
        work.drop(columns=["Aspect"], inplace=True)
    for c in list(work.columns):
        if c != "Status" and is_distance_like(c):
            work[c] = safe_log1p(work[c])
    lat_col = next((c for c in work.columns if c.lower() in ["latitude","lat"]), None)
    lon_col = next((c for c in work.columns if c.lower() in ["longitude","lon","lng"]), None)
    excluded = {"Status", lat_col, lon_col} | ID_COLS
    candidates = [c for c in work.columns if c not in excluded]
    cat_cols = []
    for c in candidates:
        if (c.lower() in CAT_LIKE) or (work[c].dtype == "object") or str(work[c].dtype).startswith("category"):
            cat_cols.append(c)
    X_df = work[candidates].copy()
    for c in candidates:
        if c not in cat_cols:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
        else:
            X_df[c] = X_df[c].astype("category")
    return X_df
print(f"\nLoading training data: {CSV_70_42}")
df = pd.read_csv(CSV_70_42)
print(f"  rows: {len(df)}, cols: {len(df.columns)}")
y_all = df["Status"].to_numpy().astype(int)
X_df_prepared = prepare_features(df)
print(f"  prepared X_df: {len(X_df_prepared.columns)} feature columns ready for preprocessor")
print(f"  Status: {dict(zip(*np.unique(y_all, return_counts=True)))}")
idx = np.arange(len(y_all))
tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_all, random_state=SEED)
print(f"  train rows: {len(tr_idx)} | test rows: {len(te_idx)}")
def model_predict(model_key):
    info  = MODELS[model_key]
    m_dir = os.path.join(RESULTS, info["dir"], "thr70", "seed42")
    keras_path = os.path.join(m_dir, "final_model.keras")
    prep_path  = os.path.join(m_dir, "static_preprocessor.joblib")
    feat_csv   = os.path.join(m_dir, "selected_features_final.csv")
    if not (os.path.exists(keras_path) and os.path.exists(prep_path) and os.path.exists(feat_csv)):
        print(f"  [{model_key}] SKIP: missing files in {m_dir}")
        return None
    selected = pd.read_csv(feat_csv)["selected_feature"].tolist()
    print(f"  [{model_key}] selected features: {len(selected)}")
    pre = joblib.load(prep_path)
    Xfull = pre.transform(X_df_prepared)
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
        print(f"  [{model_key}] ERROR: missing transformed feats: {missing[:5]} ...")
        return None
    sel_idx = [name_to_idx[s] for s in selected]
    if hasattr(Xfull, "toarray"):
        Xfull = Xfull.toarray()
    X = Xfull[:, sel_idx].astype(np.float32)
    model = tf.keras.models.load_model(keras_path, compile=False)
    t0 = time.time()
    y_pred = model.predict(X, verbose=0, batch_size=256).flatten()
    print(f"  [{model_key}] prediction done in {time.time()-t0:.1f}s, shape={y_pred.shape}")
    return {
        "y_true_tr":  y_all[tr_idx], "y_pred_tr":  y_pred[tr_idx],
        "y_true_te":  y_all[te_idx], "y_pred_te":  y_pred[te_idx],
    }
results = {}
for k in MODELS.keys():
    print(f"\n=== {k} ===")
    out = model_predict(k)
    if out is not None:
        results[k] = out
        full_y    = np.concatenate([out["y_true_tr"], out["y_true_te"]])
        full_pred = np.concatenate([out["y_pred_tr"], out["y_pred_te"]])
        full_auc  = roc_auc_score(full_y, full_pred)
        ms_path = os.path.join(RESULTS, MODELS[k]["dir"], "thr70", "seed42", "metrics_summary.json")
        with open(ms_path) as f:
            ms = json.load(f)
        expected = ms["cv_oof"]["roc_auc"]
        delta = full_auc - expected
        flag = "OK" if abs(delta) < 0.05 else "MISMATCH"
        print(f"  [{k}] sanity: full-data AUC={full_auc:.4f}  vs cv_oof_roc={expected:.4f}  delta={delta:+.4f}  [{flag}]")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for panel, (ax, kind, title) in enumerate([
    (axes[0], "tr", "Train"),
    (axes[1], "te", "Test"),
]):
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9467bd", linewidth=1.2)
    for k, info in MODELS.items():
        if k not in results:
            continue
        y_true = results[k][f"y_true_{kind}"]
        y_pred = results[k][f"y_pred_{kind}"]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, color=info["color"], linewidth=2.0,
                label=f"{info['label']} (AUC = {auc:.4f})")
    ax.set_xlim(0, 1.001); ax.set_ylim(0, 1.001)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    if panel == 0:
        ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.grid(True, alpha=0.25, linestyle="-")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
plt.tight_layout()
pdf_out = os.path.join(FIG_DIR, "Fig_ROC_4models_train_test.pdf")
png_out = os.path.join(FIG_DIR, "Fig_ROC_4models_train_test.png")
plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
plt.savefig(png_out, dpi=200, bbox_inches="tight")
print(f"\nWrote {pdf_out}")
print(f"Wrote {png_out}")
print("\nDone.")
