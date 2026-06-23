import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, "..", "data", "shap", "SHAP_BiLSTM_PSO_values.pkl")
OUT = os.path.join(HERE, "..", "figs", "Fig17_SHAP")

def main():
    d = pickle.load(open(PKL, "rb"))
    sv = np.asarray(d["shap_values"]); X = np.asarray(d["X_fg"])
    names = list(d["feature_names"])
    shap.summary_plot(sv, X, feature_names=names, plot_type="dot", show=False,
                      max_display=len(names), sort=True, color_bar=True,
                      cmap=plt.get_cmap("coolwarm"), plot_size=(10.6, 8.1))
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (impact on susceptibility output)", fontsize=11)
    fig.axes[-1].set_ylabel("Feature value (low → high)", fontsize=10)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", dpi=300, bbox_inches="tight")
    order = np.argsort(-np.abs(sv).mean(0))
    print("Saved:", OUT + ".{png,pdf}")
    print("Top factors:", ", ".join(names[i] for i in order[:5]))

if __name__ == "__main__":
    main()
