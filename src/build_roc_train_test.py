"""
build_roc_train_test.py
========================
Generate a two-panel ROC plot (Train + Test) for the four models at thr70/seed42:
  LSTM, BiLSTM, LSTM-PSO, BiLSTM-PSO

Reproduces the exact 85/15 stratified split used by both trainers
(train_test_split(test_size=0.15, stratify=y, random_state=42)).

Output: revision_c8c11/03_Figures/Fig_ROC_4models_train_test.pdf + .png
"""

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

# Suppress TF info logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(f"TF {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

# ----- Custom layer used by non-PSO LSTM/BiLSTM models -----
# Must mirror c8c11_non_pso_cli.py L180-192 so Keras can deserialize the
# Wildfire>AttentionPool1D layer in saved final_model.keras files.
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

# ----- Load data + reproduce the split -----
print(f"\nLoading training data: {CSV_70_42}")
df = pd.read_csv(CSV_70_42)
print(f"  rows: {len(df)}, cols: {len(df.columns)}")

# Derive Aspect_sin / Aspect_cos exactly as both trainers do (BiLSTM PSO FE.py L124-125,
# c8c11_non_pso_cli.py L101). PSO preprocessor's column transformer expects these columns.
if "Aspect" in df.columns:
    rad = np.deg2rad(df["Aspect"].astype(float))
    df["Aspect_sin"] = np.sin(rad)
    df["Aspect_cos"] = np.cos(rad)
    print(f"  derived Aspect_sin / Aspect_cos from Aspect")

# Target column is 'Status' (1 = fire, 0 = pseudo-absence)
y_all = df["Status"].to_numpy().astype(int)
print(f"  Status: {dict(zip(*np.unique(y_all, return_counts=True)))}")

# Reproduce the 85/15 split exactly as both trainers did
idx = np.arange(len(y_all))
tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_all, random_state=SEED)
print(f"  train rows: {len(tr_idx)} | test rows: {len(te_idx)}")

# ----- Helper: load each model, predict on train+test -----
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

    # Apply the ColumnTransformer to ALL rows of df, then select the columns
    Xfull = pre.transform(df)
    # ColumnTransformer outputs column names via get_feature_names_out()
    try:
        all_names = list(pre.get_feature_names_out())
    except Exception:
        # Older sklearn fallback: try named_transformers_
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
    # Convert sparse to dense if needed
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

# ----- Run all 4 models -----
results = {}
for k in MODELS.keys():
    print(f"\n=== {k} ===")
    out = model_predict(k)
    if out is not None:
        results[k] = out

# ----- Plot -----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for panel, (ax, kind, title) in enumerate([
    (axes[0], "tr", "Train"),
    (axes[1], "te", "Test"),
]):
    # Diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9467bd", linewidth=1.2)
    # Plot curves
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
