import os
import sys
import warnings
import pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import shap
    print("SHAP " + shap.__version__)
except ImportError:
    print("ERROR: shap not installed. Run:  pip install shap --no-deps")
    sys.exit(1)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
TABLES_DIR = os.path.join(REPO_ROOT, "data")
RESULTS    = os.path.join(REPO_ROOT, "models")
FIG_DIR    = os.path.join(REPO_ROOT, "figs")
SHAP_DIR   = os.path.join(REPO_ROOT, "data", "shap")
os.makedirs(FIG_DIR, exist_ok=True)
M_DIR    = os.path.join(RESULTS, "BiLSTM_PSO", "thr70", "seed42")
CSV      = os.path.join(TABLES_DIR, "training_points_70ha_seed42.csv")
SHAP_PKL = os.path.join(SHAP_DIR, "SHAP_BiLSTM_PSO_values.pkl")
FEATURE_LABELS = {
    "num__Slope":               "Slope",
    "num__Elevation":           "Elevation",
    "num__TWI":                 "TWI",
    "num__Profile_Curvature":   "Profile Curvature",
    "num__NDVI":                "NDVI",
    "num__Max_Temperature":     "Temperature",
    "num__Precipitation":       "Precipitation",
    "num__WS":                  "Windspeed",
    "num__Specific_Humidity":   "Specific Humidity",
    "num__AET":                 "AET",
    "num__DSI":                 "Drought Index",
    "num__Soil_Moisture":       "Soil Moisture",
    "num__Distance_roads":      "Distance from roads",
    "num__Distance_rivers":     "Distance from rivers",
    "num__Distance_households": "Distance from households",
    "cat__LULC_11.0":           "LULC (Trees)",
    "cat__LULC_9.0":            "LULC (Rangeland)",
    "cat__LULC_4.0":            "LULC (Forest)",
    "cat__LULC_2.0":            "LULC (Trees, alt)",
}
ID_COLS = {"UniqueID","Unique_ID","unique_id","OBJECTID","ObjectID","objectid","object_id",
           "FID","Fid","fid","OID","Oid","oid","pointid","PointID","point_id","POINT_ID",
           "CID","CID_","Cid","cid","ID","Id","id","index","Index","INDEX","row_id","RowID"}
CAT_LIKE = {"lulc","landuse","land_cover","landcover"}
def is_distance_like(c):
    lc = c.lower()
    return ("distance" in lc) or lc.startswith("dist") or ("_dist" in lc) or ("_distance" in lc)
def safe_log1p(x):
    x = pd.to_numeric(x, errors="coerce").copy()
    x[x < 0] = np.nan
    return np.log1p(x)
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
def compute_shap_values():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print("TF " + tf.__version__)
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
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
    df = pd.read_csv(CSV)
    df_prepared = prepare_features(df)
    selected = pd.read_csv(os.path.join(M_DIR, "selected_features_final.csv"))["selected_feature"].tolist()
    pre = joblib.load(os.path.join(M_DIR, "static_preprocessor.joblib"))
    model = tf.keras.models.load_model(os.path.join(M_DIR, "final_model.keras"), compile=False)
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
                names = [tname + "__" + c for c in cols]
            all_names.extend(list(names))
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    sel_idx = [name_to_idx[s] for s in selected]
    X = Xfull[:, sel_idx].astype(np.float32)
    labels = [FEATURE_LABELS.get(s, s.replace("num__", "").replace("cat__", "")) for s in selected]
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X), size=100, replace=False)
    fg_idx = rng.choice(len(X), size=500, replace=False)
    X_bg = X[bg_idx]
    X_fg = X[fg_idx]
    explainer = shap.GradientExplainer(model, X_bg)
    sv = explainer.shap_values(X_fg)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.array(sv).squeeze()
    if sv.ndim == 3:
        sv = sv[..., 0]
    os.makedirs(SHAP_DIR, exist_ok=True)
    with open(SHAP_PKL, "wb") as f:
        pickle.dump({"shap_values": sv, "X_fg": X_fg, "feature_names": labels,
                     "model_key": "BiLSTM_PSO_thr70_seed42"}, f)
    return sv, X_fg, labels
if os.path.exists(SHAP_PKL):
    print("Loading archived SHAP values: " + SHAP_PKL)
    with open(SHAP_PKL, "rb") as f:
        d = pickle.load(f)
    shap_values = np.array(d["shap_values"])
    X_fg = np.array(d["X_fg"])
    display_names = list(d["feature_names"])
else:
    print("No archived SHAP values found; recomputing with GradientExplainer.")
    shap_values, X_fg, display_names = compute_shap_values()
print("Building beeswarm plot...")
plt.figure(figsize=(9, 7))
shap.summary_plot(
    shap_values, X_fg,
    feature_names=display_names,
    plot_type="dot",
    show=False,
    max_display=len(display_names),
    sort=True,
    color_bar=True,
    cmap=plt.get_cmap("coolwarm"),
)
fig = plt.gcf()
ax = plt.gca()
ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
for child in fig.get_children():
    if hasattr(child, "get_label") and child.get_label() == "<colorbar>":
        child.set_ylabel("Feature value", fontsize=10)
pdf_out = os.path.join(FIG_DIR, "Fig_SHAP_BiLSTM_PSO_beeswarm.pdf")
png_out = os.path.join(FIG_DIR, "Fig_SHAP_BiLSTM_PSO_beeswarm.png")
plt.tight_layout()
plt.savefig(pdf_out, dpi=300, bbox_inches="tight")
plt.savefig(png_out, dpi=200, bbox_inches="tight")
print("Wrote " + pdf_out)
print("Wrote " + png_out)
print("Done.")
