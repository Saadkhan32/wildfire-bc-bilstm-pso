import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUSC = os.path.join(HERE, "..", "data", "susceptibility")
OUT = os.path.join(HERE, "..", "figs", "Fig16_class_area.png")
EDGES = [0.2, 0.4, 0.6, 0.8]
CLASSES = ["Very Low", "Low", "Moderate", "High", "Very High"]
COLORS = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]

MODELS = [("BiLSTM_PSO", "BiLSTM-PSO"), ("LSTM_PSO", "LSTM-PSO"),
          ("BiLSTM", "BiLSTM"), ("LSTM", "LSTM")]

def class_pct(path):
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(float); nod = ds.nodata
    a = a.ravel(); a = a[np.isfinite(a)]
    if nod is not None:
        a = a[a != nod]
    a = a[(a >= 0) & (a <= 1)]
    idx = np.digitize(a, EDGES)
    return [100.0 * np.count_nonzero(idx == k) / a.size for k in range(5)]

def main():
    rows = []
    for key, label in MODELS:
        p = os.path.join(SUSC, f"BC_susceptibility_{key}.tif")
        if not os.path.exists(p):
            print(f"[skip] missing {p}"); continue
        rows.append((label, class_pct(p)))

    print(f"{'Model':12s}" + "".join(f"{c:>11s}" for c in CLASSES))
    for label, pct in rows:
        print(f"{label:12s}" + "".join(f"{v:>10.1f}%" for v in pct))

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [r[0] for r in rows][::-1]
    data = np.array([r[1] for r in rows])[::-1]
    y = np.arange(len(labels)); left = np.zeros(len(labels))
    for k in range(5):
        ax.barh(y, data[:, k], left=left, color=COLORS[k], edgecolor="white",
                label=CLASSES[k], height=0.6)
        for i in range(len(labels)):
            if data[i, k] > 3:
                ax.text(left[i] + data[i, k] / 2, y[i], f"{data[i,k]:.1f}%",
                        ha="center", va="center", fontsize=9)
        left += data[:, k]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, 100); ax.set_xlabel("% of BC land area")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("\nSaved:", OUT)

if __name__ == "__main__":
    main()
