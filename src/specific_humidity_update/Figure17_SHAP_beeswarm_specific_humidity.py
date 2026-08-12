"""
Regenerate Fig. 17 (SHAP beeswarm) with the humidity predictor relabelled
'RH' -> 'Specific Humidity'.

The model was trained on ERA5-Land specific humidity (kg/kg); the original
SHAP figure mislabelled this feature as 'RH'. No model re-run is needed - the
stored SHAP values are reused verbatim and only the feature label changes.

Input : revision_c8c11/06_Final_Tables/SHAP_BiLSTM_PSO_values.pkl
        dict with keys: shap_values (500,16), X_fg (500,16), feature_names (16)
Output: Fig17_SHAP_specifichumidity.{png,pdf}

Run from repo root:  python src/specific_humidity_update/regenerate_fig17_shap.py
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKL = os.path.join(REPO, "revision_c8c11", "06_Final_Tables", "SHAP_BiLSTM_PSO_values.pkl")
OUT = os.path.join(REPO, "Fig17_SHAP_specifichumidity")

def main():
    d = pickle.load(open(PKL, "rb"))
    shap_values = np.asarray(d["shap_values"])
    X = np.asarray(d["X_fg"])
    # relabel the humidity feature; everything else is unchanged
    names = ["Specific Humidity" if n == "RH" else n for n in d["feature_names"]]

    # plot_size forces a landscape figure matching the original embedded figure
    shap.summary_plot(
        shap_values, X, feature_names=names, plot_type="dot", show=False,
        max_display=len(names), sort=True, color_bar=True,
        cmap=plt.get_cmap("coolwarm"), plot_size=(10.6, 8.1),
    )
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (impact on susceptibility output)", fontsize=11)
    fig.axes[-1].set_ylabel("Feature value (low → high)", fontsize=10)  # colorbar
    fig.savefig(f"{OUT}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    order = np.argsort(-np.abs(shap_values).mean(0))
    print("Saved:", OUT + ".{png,pdf}")
    print("Feature order (top -> bottom):", [names[i] for i in order])

if __name__ == "__main__":
    main()
