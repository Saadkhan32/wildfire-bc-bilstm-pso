import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data", "analysis_dataset.csv")
SH_MONTHLY = os.path.join(REPO, "data", "climate", "BC_ERA5Land_monthly_specific_humidity_2000_2024.csv")
OUTDIR = os.path.join(REPO, "figs")
OUT = os.path.join(OUTDIR, "Fig18_ENSO_teleconnection")

OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#9A9A9A")
DIV = LinearSegmentedColormap.from_list("BuWhOr", [OI["blue"], "#f7f7f7", OI["verm"]])

def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
    })

def save_figure(fig, path_noext, png_dpi=180):
    fig.savefig(path_noext + ".pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path_noext + ".tif", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path_noext + ".png", dpi=png_dpi, bbox_inches="tight", pad_inches=0.05)
    im = Image.open(path_noext + ".tif").convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    Image.alpha_composite(bg, im).convert("RGB").save(
        path_noext + ".tif", dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)

def load_data():
    D = pd.read_csv(DATA).set_index("year")
    if "SH" not in D.columns:
        sh = pd.read_csv(SH_MONTHLY).rename(columns={"specific_humidity_mean": "sh"})
        fs = sh[sh.month.isin([5, 6, 7, 8])].groupby("year")["sh"].mean()
        D["SH"] = D.index.map(fs)
    return D

def clean(D, a, b):
    m = ~(D[a].isna() | D[b].isna())
    return D[a][m].values, D[b][m].values

def spearman_ci(D, a, b):
    x, y = clean(D, a, b)
    r, p = stats.spearmanr(x, y)
    n = len(x)
    se = np.sqrt((1 + r ** 2 / 2.0) / (n - 3))
    z = np.arctanh(r)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), p

def detrend(s):
    s = s.astype(float)
    m = ~s.isna()
    tt = np.arange(len(s))
    out = s.values.astype(float).copy()
    a = tt[m.values]
    y = s[m].values
    b, c = np.polyfit(a, y, 1)
    out[m.values] = y - (c + b * a)
    return pd.Series(out, index=s.index)

def stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

DCOL = ["Tmax", "Precip", "SH", "Soil", "PDSI", "AET"]
DLAB = ["Temperature", "Precipitation", "Sp. humidity", "Soil moisture",
        "Drought", "Evapotranspiration"]
TELE = ["oni_djf", "oni_mam", "oni_jja", "pdo_djf", "pdo_mam", "pdo_jja"]
TLAB = ["ENSO winter", "ENSO spring", "ENSO summer", "PDO winter", "PDO spring", "PDO summer"]

def panel_a(ax, D):
    GREEN, BLUE, GREY = OI["green"], OI["blue"], "#666666"
    ax.grid(axis="x", color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    variables = [("Sp. humidity", "SH"), ("Precipitation", "Precip"),
                 ("Soil moisture", "Soil"), ("Drought", "PDSI"),
                 ("Evapotranspiration", "AET"), ("Temperature", "Tmax")]
    yv = [2.0 + (5 - i) * 1.35 for i in range(6)]
    off = 0.24

    def draw(r, lo, hi, y, color, p, above):
        sig = p < 0.05
        ax.plot([lo, hi], [y, y], color=color, lw=2.1, solid_capstyle="round", zorder=3)
        ax.plot(r, y, "o", ms=6.0, zorder=4, mfc=color if sig else "white", mec=color, mew=1.4)
        dy = 0.135 if above else -0.135
        ax.text(r, y + dy, f"{r:+.2f}{stars(p)}", ha="center",
                va="bottom" if above else "top", fontsize=7,
                color=color, fontweight="bold" if sig else "normal", zorder=5)

    for (lab, col), y in zip(variables, yv):
        ra, loa, hia, pa = spearman_ci(D, "oni_jja", col)
        draw(ra, loa, hia, y + off, GREEN, pa, True)
        rb, lob, hib, pb = spearman_ci(D, col, "area70")
        draw(rb, lob, hib, y - off, BLUE, pb, False)
    rd, lod, hid, pd_ = spearman_ci(D, "oni_jja", "area70")
    draw(rd, lod, hid, 0.5, GREY, pd_, True)
    ax.axhline(1.25, color="0.8", lw=0.7, ls=(0, (4, 3)))
    ax.axvline(0, color="0.4", lw=0.9, zorder=2)
    ax.set_yticks(yv + [0.5])
    ax.set_yticklabels([v[0] for v in variables] + ["ENSO (direct)"])
    ax.set_ylim(-0.25, 9.6)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Spearman $\\rho$ (95 % CI)")
    ax.set_title("ENSO $\\rightarrow$ fire-season climate $\\rightarrow$ area burned")

def panel_b(ax, D):
    Rm = np.array([[stats.spearmanr(*clean(D, t, d))[0] for d in DCOL] for t in TELE])
    Pm = np.array([[stats.spearmanr(*clean(D, t, d))[1] for d in DCOL] for t in TELE])
    im = ax.imshow(Rm, cmap=DIV, vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(DCOL)))
    ax.set_xticklabels(DLAB, rotation=32, ha="right", fontsize=7)
    ax.set_yticks(range(len(TELE)))
    ax.set_yticklabels(TLAB)
    ax.set_xticks(np.arange(-.5, len(DCOL), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(TELE), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.3)
    ax.tick_params(which="minor", length=0)
    ax.axhline(2.5, color="0.15", lw=1.9)
    for i in range(len(TELE)):
        for j in range(len(DCOL)):
            ax.text(j, i, f"{Rm[i, j]:+.2f}{stars(Pm[i, j])}", ha="center", va="center",
                    fontsize=6.0, fontweight="bold" if Pm[i, j] < 0.05 else "normal",
                    color="white" if abs(Rm[i, j]) > 0.45 else "0.12")
    ax.set_title("ENSO and PDO vs fire-season climate")
    return im

def panel_c(ax, D):
    pairs = [("SH", "Sp. humidity"), ("Precip", "Precipitation"), ("Tmax", "Temperature")]
    raw = [abs(stats.spearmanr(*clean(D, a, "area70"))[0]) for a, _ in pairs]
    det = []
    for a, _ in pairs:
        da, db = detrend(D[a]), detrend(D["area70"])
        m = ~(da.isna() | db.isna())
        det.append(abs(stats.spearmanr(da[m], db[m])[0]))
    x = np.arange(len(pairs))
    w = 0.36
    ax.bar(x - w / 2, raw, w, color=OI["verm"], label="Raw (with trend)")
    ax.bar(x + w / 2, det, w, color=OI["sky"], label="Detrended (interannual)")
    for i, (r, d) in enumerate(zip(raw, det)):
        ax.text(i - w / 2, r + .02, f"{r:.2f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, d + .02, f"{d:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in pairs])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("$|\\rho|$ with area burned")
    ax.legend(frameon=False, loc="upper center", ncol=1,
              bbox_to_anchor=(0.5, 1.0), handlelength=1.2, fontsize=7.6)
    ax.set_title("Interannual vs trend control")

def legend_paired(fig):
    h = [Line2D([0], [0], marker="o", color=OI["green"], lw=2, mfc=OI["green"], mec=OI["green"], ms=6, label="ENSO $\\rightarrow$ climate"),
         Line2D([0], [0], marker="o", color=OI["blue"], lw=2, mfc=OI["blue"], mec=OI["blue"], ms=6, label="climate $\\rightarrow$ area burned"),
         Line2D([0], [0], marker="o", color="0.4", lw=0, mfc="white", mec="0.4", ms=6, label="open = not significant")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=3,
               frameon=False, fontsize=7.2, handletextpad=0.3, columnspacing=1.1)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    apply_style()
    D = load_data()

    fig = plt.figure(figsize=(7.48, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1], height_ratios=[1, 1],
                          hspace=0.6, wspace=0.5, left=0.155, right=0.93, top=0.93, bottom=0.16)
    axa = fig.add_subplot(gs[:, 0]); panel_a(axa, D)
    axb = fig.add_subplot(gs[0, 1]); im = panel_b(axb, D)
    cb = fig.colorbar(im, ax=axb, fraction=0.046, pad=0.03)
    cb.set_label("Spearman $\\rho$", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5, width=0.5)
    cb.outline.set_linewidth(0.5)
    axc = fig.add_subplot(gs[1, 1]); panel_c(axc, D)
    for ax, L, dx in [(axa, "a", 0.13), (axb, "b", 0.085), (axc, "c", 0.085)]:
        p = ax.get_position()
        fig.text(p.x0 - dx, p.y1 + 0.012, L, fontsize=11, fontweight="bold")
    legend_paired(fig)
    fig.text(0.5, 0.012, "* p < 0.05      ** p < 0.01      *** p < 0.001",
             ha="center", fontsize=7.2, color="0.35")
    save_figure(fig, OUT)
    print("Saved:", OUT + ".{pdf,tif,png}")

    for a, b, lab in [("oni_jja", "SH", "ENSO(summer)->Sp.humidity"),
                      ("SH", "area70", "Sp.humidity->area burned")]:
        r, lo, hi, p = spearman_ci(D, a, b)
        print(f"  {lab}: rho={r:+.2f} (95% CI {lo:+.2f},{hi:+.2f}; p={p:.3f})")

if __name__ == "__main__":
    main()
