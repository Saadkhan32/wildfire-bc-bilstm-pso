"""Single-page Fig. 12 (portrait): seasonal climate trends and
wildfire/non-wildfire composites - nine variables in a 2 x 5 grid, with the
legend and notes in the tenth cell. Designed to fill one Word page at text
width (aspect ~1.38) with journal-readable font sizes.

Every number is computed from the packaged data (Theil-Sen slope and 95% CI
of the May-August annual means, drawn with its confidence band; Welch tests
with Benjamini-Hochberg correction across the nine variables). A value table
is printed at the end for aligning the manuscript text.

Inputs (data/, git-tracked):
    BC_2000_2024_monthly_climate_wide.csv
    BC_ERA5Land_monthly_specific_humidity_2000_2024.csv
    BC_TerraClimate_monthly_{AET,PDSI,SOIL}_2000_2024.csv
Output:
    figs/Fig12_climate_onepage.{png,pdf,tif}   (600 dpi PNG and TIFF)

Run from the repository root:  python make_fig12_onepage.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "figs", "Fig12_climate_onepage")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "font.weight": "bold",
    "axes.labelsize": 9, "axes.labelweight": "bold",
    "axes.titlesize": 8, "axes.titleweight": "bold",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MONTH_COL = {5: ("May", "#0072B2"), 6: ("June", "#E69F00"),
             7: ("July", "#009E73"), 8: ("August", "#D55E00")}
NWF_FACE, WF_FACE = "#B8CCE4", "#F2A9A2"
VERDICT_COL = {"increasing": "#1E7A3C", "decreasing": "#B2452C", "no trend": "#5A5A5A"}


def _terra(path, name):
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"])
    d["year"], d["month"] = d.date.dt.year, d.date.dt.month
    return d[["year", "month", "mean"]].rename(columns={"mean": name})


def load_merged():
    clim = pd.read_csv(os.path.join(DATA, "BC_2000_2024_monthly_climate_wide.csv"))
    sh = pd.read_csv(os.path.join(DATA, "BC_ERA5Land_monthly_specific_humidity_2000_2024.csv"))
    sh = sh.rename(columns={"specific_humidity_mean": "specific_humidity"})
    df = clim.merge(sh, on=["year", "month"], how="outer")
    for f, n in [("BC_TerraClimate_monthly_AET_2000_2024.csv", "AET"),
                 ("BC_TerraClimate_monthly_PDSI_2000_2024.csv", "PDSI"),
                 ("BC_TerraClimate_monthly_SOIL_2000_2024.csv", "soil_moisture")]:
        df = df.merge(_terra(os.path.join(DATA, f), n), on=["year", "month"], how="outer")
    return df.sort_values(["year", "month"]).reset_index(drop=True)


PANELS = [  # (column, label, unit, sci)
    ("max_temperature", "Max Temperature", "°C", False),
    ("avg_temperature", "Avg Temperature", "°C", False),
    ("min_temperature", "Min Temperature", "°C", False),
    ("specific_humidity", "Specific Humidity", "kg/kg", True),
    ("avg_wind_speed", "Avg Wind Speed", "m/s", False),
    ("PDSI", "PDSI", "unitless", False),
    ("AET", "AET", "mm/month", False),
    ("precipitation", "Precipitation", "mm", False),
    ("soil_moisture", "Soil Moisture", "mm", False),
]


def fnum(x, sci):
    return f"{x:+.2e}" if sci else f"{x:+.3f}"


def main():
    df = load_merged()
    wf = df[df.month.isin([5, 6, 7, 8])]
    nwf = df[~df.month.isin([5, 6, 7, 8])]

    pvals = []
    for col, *_ in PANELS:
        a = wf[col].dropna().to_numpy()
        b = nwf[col].dropna().to_numpy()
        pvals.append(stats.ttest_ind(a, b, equal_var=False).pvalue)
    qvals = multipletests(pvals, method="fdr_bh")[1]

    def star(q):
        return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "n.s."

    fig = plt.figure(figsize=(9.6, 13.3))
    outer = gridspec.GridSpec(5, 2, figure=fig,
                              left=0.085, right=0.982, top=0.940, bottom=0.045,
                              hspace=0.62, wspace=0.28)
    table = []
    for i, (col, label, unit, sci) in enumerate(PANELS):
        r, c = divmod(i, 2)
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[r, c], width_ratios=[2.5, 1.0], wspace=0.33)
        axl = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1])

        # --- left: monthly series + Theil-Sen trend with 95% CI band ---
        for m, (mname, cc) in MONTH_COL.items():
            s = df[df.month == m].dropna(subset=[col])
            axl.plot(s.year, s[col], "-o", color=cc, lw=1.2, ms=2.9,
                     markeredgecolor="white", markeredgewidth=0.35, zorder=3)
        ann = wf.groupby("year")[col].mean().dropna()
        yrs = ann.index.to_numpy(float)
        res = stats.theilslopes(ann.to_numpy(), yrs, alpha=0.95)
        slope, inter, lo, hi = res[0], res[1], res[2], res[3]
        xm = yrs.mean()
        y_at_xm = inter + slope * xm
        axl.fill_between(yrs, y_at_xm + lo * (yrs - xm), y_at_xm + hi * (yrs - xm),
                         color="0.45", alpha=0.16, lw=0, zorder=1)
        axl.plot(yrs, inter + slope * yrs, "k--", lw=1.5, zorder=4)
        verdict = "increasing" if lo > 0 else "decreasing" if hi < 0 else "no trend"
        u = "" if unit == "unitless" else f" {unit}"
        axl.set_title(f"Theil-Sen {fnum(slope, sci)}{u} yr$^{{-1}}$;  "
                      f"95% CI [{fnum(lo, sci)}, {fnum(hi, sci)}]",
                      loc="left", pad=4)
        # reserved headroom keeps the verdict tag clear of the data
        sub = wf[col].dropna()
        ymin, ymax = float(sub.min()), float(sub.max())
        rng = ymax - ymin
        axl.set_ylim(ymin - 0.05 * rng, ymax + 0.19 * rng)
        axl.text(0.972, 0.972, verdict, transform=axl.transAxes, ha="right",
                 va="top", fontsize=7.6, fontweight="bold",
                 color=VERDICT_COL[verdict],
                 bbox=dict(boxstyle="round,pad=0.24", facecolor="white",
                           edgecolor=VERDICT_COL[verdict], linewidth=0.6,
                           alpha=0.9))
        axl.text(-0.155, 1.14, f"({chr(97 + i)})", transform=axl.transAxes,
                 fontsize=12, fontweight="bold", va="bottom")
        axl.set_ylabel(f"{label} ({unit})")
        axl.set_xlabel("Year")
        axl.grid(alpha=0.25, lw=0.5)
        axl.set_xticks([2000, 2005, 2010, 2015, 2020, 2024])
        axl.set_xlim(1999, 2025)
        for sp in ("top", "right"):
            axl.spines[sp].set_visible(False)

        # --- right: wildfire vs non-wildfire boxes ---
        a = nwf[col].dropna().to_numpy()
        b = wf[col].dropna().to_numpy()
        p1, p2 = 0.85, 2.25
        bp = axr.boxplot([a, b], positions=[p1, p2], widths=0.5,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color="#1E7A3C", lw=1.6),
                         boxprops=dict(lw=0.7, edgecolor="0.25"),
                         whiskerprops=dict(lw=0.7, color="0.25"),
                         capprops=dict(lw=0.7, color="0.25"))
        for patch, fc in zip(bp["boxes"], [NWF_FACE, WF_FACE]):
            patch.set_facecolor(fc)
        for pos, v in [(p1, a), (p2, b)]:
            mn, ci = v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))
            axr.errorbar(pos, mn, yerr=ci, fmt="D", color="black", ms=3.8,
                         markeredgecolor="white", markeredgewidth=0.4,
                         capsize=2, lw=1, zorder=4)
            dec = 4 if sci else 2
            axr.text(pos + 0.32, mn, f"{mn:.{dec}f}", va="center",
                     ha="left", fontsize=7)
        top = max(a.max(), b.max())
        span = top - min(a.min(), b.min())
        y0 = top + 0.06 * span
        axr.plot([p1, p1, p2, p2],
                 [y0, y0 + 0.035 * span, y0 + 0.035 * span, y0],
                 color="black", lw=0.8)
        axr.text((p1 + p2) / 2, y0 + 0.045 * span, star(qvals[i]), ha="center",
                 va="bottom", fontsize=8.5, fontweight="bold")
        axr.set_ylim(top=y0 + 0.21 * span)
        axr.set_xlim(0.28, 3.15)
        axr.set_xticks([p1, p2])
        axr.set_xticklabels(["Non-Wildfire\nSeason", "Wildfire\nSeason"],
                            fontsize=6.4, linespacing=1.1)
        axr.grid(alpha=0.25, lw=0.5, axis="y")
        for sp in ("top", "right"):
            axr.spines[sp].set_visible(False)

        table.append((chr(97 + i), label, fnum(slope, sci), fnum(lo, sci),
                      fnum(hi, sci), verdict, f"{qvals[i]:.2e}", star(qvals[i])))

    # --- legend at the top, notes in the tenth cell ---
    handles = [Line2D([], [], color=cc, marker="o", ms=4.2, lw=1.4,
                      markeredgecolor="white", markeredgewidth=0.35, label=n)
               for m, (n, cc) in MONTH_COL.items()]
    handles += [Line2D([], [], color="black", ls="--", lw=1.5,
                       label="Theil-Sen (May-Aug mean)"),
                Patch(facecolor="0.45", alpha=0.25, label="95% CI of trend"),
                Patch(facecolor=NWF_FACE, edgecolor="0.25", lw=0.6,
                      label="Non-Wildfire season"),
                Patch(facecolor=WF_FACE, edgecolor="0.25", lw=0.6,
                      label="Wildfire season"),
                Line2D([], [], color="#1E7A3C", lw=1.6, label="Median"),
                Line2D([], [], color="black", marker="D", ls="none", ms=4.2,
                       markeredgecolor="white", markeredgewidth=0.4,
                       label="Mean ± 95% CI")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.998), fontsize=8.4,
               handletextpad=0.5, columnspacing=1.3, labelspacing=0.5)

    axleg = fig.add_subplot(outer[4, 1])
    axleg.axis("off")
    axleg.text(0.02, 0.90,
               "Box: IQR (Q1-Q3); whiskers: 1.5 × IQR.\n"
               "*** q < 0.001,  ** q < 0.01,  * q < 0.05\n"
               "(Welch tests, BH-adjusted); n.s. = not significant.\n"
               "Trend verdicts refer to the 95% CI of the\n"
               "Theil-Sen slope.",
               transform=axleg.transAxes, fontsize=7.8, va="top",
               fontweight="normal", linespacing=1.45)

    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT + ".tif", dpi=600, bbox_inches="tight", pad_inches=0.06,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print(f"\nWrote {OUT}.png / .pdf / .tif\n")
    print("Values on the figure (use these to align the manuscript text):")
    print(f"{'panel':5s} {'variable':18s} {'slope/yr':>12s} {'CI low':>12s} "
          f"{'CI high':>12s}  {'verdict':12s} {'BH q':>10s}  sig")
    for r in table:
        print(f"({r[0]})  {r[1]:18s} {r[2]:>12s} {r[3]:>12s} {r[4]:>12s}  "
              f"{r[5]:12s} {r[6]:>10s}  {r[7]}")


if __name__ == "__main__":
    main()
