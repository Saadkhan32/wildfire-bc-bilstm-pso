import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from figure_style import apply_style, save_figure

apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "analysis_dataset.csv")
OUTDIR = os.path.join(HERE, "..", "figs")
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "Fig6_wildfire_trend")
SKY, BLUE, VERM, GREY = "#56B4E9", "#0072B2", "#D55E00", "#666666"

D = pd.read_csv(DATA).set_index("year")
YR = D.index.values.astype(float)
T = YR - YR[0]

def mann_kendall(x):
    x = np.asarray(x, float)
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    vals, cnt = np.unique(x, return_counts=True)
    tie = np.sum(cnt * (cnt - 1) * (2 * cnt + 5))
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    z = (s - 1) / np.sqrt(var) if s > 0 else ((s + 1) / np.sqrt(var) if s < 0 else 0.0)
    tau = s / (0.5 * n * (n - 1))
    return tau, 2 * (1 - stats.norm.cdf(abs(z)))

def theil_sen(x):
    x = np.asarray(x, float)
    return np.median([(x[j] - x[i]) / (T[j] - T[i])
                      for i in range(len(x)) for j in range(i + 1, len(x))])

def sen_line(x):
    sl = theil_sen(x)
    return (np.median(x) - sl * np.median(T)) + sl * T, sl

def main():
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    axr = ax.twinx()
    axr.spines["top"].set_visible(False)
    fig.subplots_adjust(left=0.135, right=0.85, top=0.95, bottom=0.30)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.88", lw=0.5, zorder=0)
    c = D.n70.values
    cl_, cs = sen_line(c)
    ctau, cp = mann_kendall(c)
    a = D.area70.values / 1e6
    al_, asl = sen_line(a)
    atau, ap = mann_kendall(D.area70.values)
    b1 = ax.bar(YR, c, color=SKY, width=0.72, zorder=2)
    l1, = ax.plot(YR, cl_, color=BLUE, lw=2.2, zorder=4)
    l2, = axr.plot(YR, a, color="0.6", lw=0.8, marker="o", ms=3.4,
                   mfc=VERM, mec="white", mew=0.6, zorder=3)
    l3, = axr.plot(YR, al_, color=VERM, lw=2.2, zorder=4)
    ax.set_ylabel("Fire count (>=70 ha)")
    axr.set_ylabel("Area burned (x10^6 ha)", color=VERM)
    axr.tick_params(axis="y", colors=VERM)
    axr.spines["right"].set_color(VERM)
    ax.set_xlabel("Year")
    ax.set_xticks(range(2000, 2025, 4))
    axr.set_ylim(-0.12, 3.05)
    ax.set_ylim(0, 300)
    ax.text(0.035, 0.97,
            "Count: tau = %+.2f, p = %.3f; Sen = %+.1f fires/yr\n"
            "Area: tau = %+.2f, p = %.3f; Sen = +%s ha/yr" % (
                ctau, cp, cs, atau, ap, format(int(round(asl * 1e6)), ",")),
            transform=ax.transAxes, va="top", ha="left", fontsize=6.6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.5))
    ax.legend([b1, l1, l2, l3],
              ["Fire count (>=70 ha)", "count trend (Sen)", "Area burned", "area trend (Sen)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False,
              handlelength=1.6, columnspacing=1.4, fontsize=7)
    save_figure(fig, OUT)

if __name__ == "__main__":
    main()
