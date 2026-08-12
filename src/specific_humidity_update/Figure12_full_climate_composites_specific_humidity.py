"""
Figure 12 - Monthly climate variables in BC (2000-2024), wildfire vs non-wildfire.
=================================================================================
SINGLE-FILE generator for the COMPLETE figure: all nine panels (a-i) across the
three composite images, with the humidity panel (d) showing SPECIFIC HUMIDITY
(kg/kg) instead of relative humidity.

  Image 1  (a) Max Temperature   (b) Avg Temperature   (c) Min Temperature
  Image 2  (d) Specific Humidity (e) Avg Wind Speed    (f) PDSI
  Image 3  (g) AET               (h) Precipitation     (i) Soil Moisture

Each panel: left = monthly fire-season (May-Aug) time series with Sen's slope and
Mann-Kendall (tau, p); right = season boxplots with median +/- 95% CI (green) and
mean +/- 95% CI (black), and a Benjamini-Hochberg-adjusted significance bracket.

Font: Times New Roman (manuscript body font). Liberation Serif / Nimbus Roman are
metric-compatible fallbacks used automatically where Times New Roman is absent;
run on a machine with Times New Roman installed for an exact embed.

Inputs (repo-relative; edit the five paths below if needed):
  data/BC_2000_2024_monthly_climate_wide.csv
  data/BC_ERA5Land_monthly_specific_humidity_2000_2024.csv   (ERA5-Land q, kg/kg)
  data/BC_TerraClimate_monthly_AET_2000_2024.csv             (GEE export, 'mean')
  data/BC_TerraClimate_monthly_PDSI_2000_2024.csv            (GEE export, 'mean')
  data/BC_TerraClimate_monthly_SOIL_2000_2024.csv            (GEE export, 'mean')

Outputs:
  figs/climate_v2/Fig12_temperature_abc.{png,jpg,pdf}
  figs/climate_v2/Fig12_atmospheric_def.{png,jpg,pdf}
  figs/climate_v2/Fig12_hydrology_ghi.{png,jpg,pdf}

The 2024 PDSI is an extreme outlier (monthly mean ~ -3.26 vs 2014-2023 climatology
~ -0.34); as in the original pipeline (infill_recent_year), a scale-mismatched
target year is replaced with the 2014-2023 monthly climatology. All BH-FDR q
values are computed across the nine variables together.

Run from repo root:
  python src/specific_humidity_update/Figure12_climate_boxplots_specific_humidity.py
Requires: numpy, pandas, scipy, seaborn, matplotlib, pillow
"""
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats as sps

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
CLIM = os.path.join(DATA, "BC_2000_2024_monthly_climate_wide.csv")
SH = os.path.join(DATA, "BC_ERA5Land_monthly_specific_humidity_2000_2024.csv")
AET = os.path.join(DATA, "BC_TerraClimate_monthly_AET_2000_2024.csv")
PDSI = os.path.join(DATA, "BC_TerraClimate_monthly_PDSI_2000_2024.csv")
SOIL = os.path.join(DATA, "BC_TerraClimate_monthly_SOIL_2000_2024.csv")
OUTDIR = os.path.join(REPO, "figs", "climate_v2")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})
sns.set_style("whitegrid")
sns.set_palette("colorblind")
GRID = dict(color="grey", linewidth=0.5, alpha=0.3)
WF = {5: "May", 6: "June", 7: "July", 8: "August"}

UNIT = {"max_temperature": "°C", "avg_temperature": "°C", "min_temperature": "°C",
        "specific_humidity": "kg/kg", "avg_wind_speed": "m/s", "PDSI": "unitless",
        "AET": "mm/month", "precipitation": "mm", "soil_moisture": "mm"}
DEC = {v: (4 if v == "specific_humidity" else 2) for v in UNIT}

GROUPS = [
    ("temperature", ["max_temperature", "avg_temperature", "min_temperature"], ["a", "b", "c"]),
    ("atmospheric", ["specific_humidity", "avg_wind_speed", "PDSI"], ["d", "e", "f"]),
    ("hydrology", ["AET", "precipitation", "soil_moisture"], ["g", "h", "i"]),
]
ALLVARS = [v for _, vs, _ in GROUPS for v in vs]


def ylabel(var):
    return "%s (%s)" % (var.replace("_", " ").title(), UNIT[var])


def _terra(path, name):
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d.date.dt.year
    d["month"] = d.date.dt.month
    return d[["year", "month", "mean"]].rename(columns={"mean": name})


def load_merged():
    clim = pd.read_csv(CLIM)
    sh = pd.read_csv(SH).rename(columns={"specific_humidity_mean": "specific_humidity"})
    df = clim.merge(sh, on=["year", "month"], how="outer")
    df = df.merge(_terra(AET, "AET"), on=["year", "month"], how="outer")
    df = df.merge(_terra(PDSI, "PDSI"), on=["year", "month"], how="outer")
    df = df.merge(_terra(SOIL, "soil_moisture"), on=["year", "month"], how="outer")
    return df.sort_values(["year", "month"]).reset_index(drop=True)


def infill_recent_year(df, win=(2014, 2023), lo=0.5, hi=2.0):
    ty = int(df.year.max())
    for v in [c for c in df.columns if c not in ("year", "month")
              and pd.api.types.is_numeric_dtype(df[c])]:
        tv = df.loc[df.year == ty, v].dropna()
        cv = df.loc[df.year.between(*win), v].dropna()
        if tv.empty or cv.empty or abs(cv.mean()) < 1e-9:
            continue
        if lo <= abs(tv.mean()) / abs(cv.mean()) <= hi:
            continue
        clim = df[df.year.between(*win)].groupby("month")[v].mean()
        for mo in range(1, 13):
            m = (df.year == ty) & (df.month == mo)
            if m.any():
                df.loc[m, v] = clim.get(mo, np.nan)
    return df.sort_values(["year", "month"]).reset_index(drop=True)


def cohens_d(x, y):
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]; nx, ny = len(x), len(y)
    sp = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp


def cliffs_delta(x, y):
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]; nx, ny = len(x), len(y)
    c = np.concatenate([x, y]); r = pd.Series(c).rank().values
    return 2 * (r[:nx].sum() - nx * (nx + 1) / 2) / (nx * ny) - 1


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    rk = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n); q[o] = np.minimum(rk, 1.0)
    return q


def season_stats(df):
    rows = {}
    p = []
    for v in ALLVARS:
        s = df[["month", v]].dropna()
        x = s.loc[s.month.isin(WF), v].astype(float).values
        y = s.loc[~s.month.isin(WF), v].astype(float).values
        t, tp = sps.ttest_ind(x, y, equal_var=False)
        rows[v] = dict(meanW=x.mean(), meanN=y.mean(), diff=x.mean() - y.mean(),
                       welch_t=t, welch_p=tp, d=cohens_d(x, y), delta=cliffs_delta(x, y),
                       nW=len(x), nN=len(y))
        p.append(tp)
    q = bh(p)
    for v, qq in zip(ALLVARS, q):
        rows[v]["q"] = qq
    return rows


def mean_ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < 2:
        return np.nan, np.nan
    return x.mean(), sps.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)


def mk_trend(ax, x, y, fs=10):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = len(y)
    if n < 3:
        return
    S = sum(np.sign(y[k + 1:] - y[k]).sum() for k in range(n - 1))
    _, c = np.unique(y, return_counts=True)
    varS = (n * (n - 1) * (2 * n + 5) - np.sum(c * (c - 1) * (2 * c + 5))) / 18
    if varS == 0:
        return
    Z = (S - 1) / math.sqrt(varS) if S > 0 else (S + 1) / math.sqrt(varS) if S < 0 else 0
    p = 2 * (1 - sps.norm.cdf(abs(Z))); tau = S / (n * (n - 1) / 2)
    sl = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]; dy = y[i + 1:] - y[i]; mm = dx != 0
        if np.any(mm):
            sl.extend((dy[mm] / dx[mm]).tolist())
    s = float(np.median(sl)); b = float(np.nanmean(y) - s * np.nanmean(x))
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, b + s * xs, "--", color="black", lw=1, zorder=3)
    ps = "p < 0.0001" if p < 1e-4 else ("p = %.4f" % p if p < 1e-3 else "p = %.3f" % p)
    ax.text(0.5, 0.02, "τ = %.3f, %s" % (tau, ps), transform=ax.transAxes, ha="center",
            va="bottom", fontsize=fs, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8), zorder=4)


def panel(df, var, letter, ax_ts, ax_box, q):
    sub = df[["year", "month", var]].dropna().copy()
    sub["MonthName"] = sub["month"].map(WF)
    sub["season"] = np.where(sub["month"].isin(WF), "Wildfire Season", "Non-Wildfire Season")
    dl = sub[sub["MonthName"].notna()].copy()
    yrs = np.sort(dl["year"].unique()); xt = yrs[::max(1, len(yrs) // 8 or 1)]
    ax_ts.grid(**GRID)
    sns.lineplot(data=dl, x="year", y=var, hue="MonthName",
                 hue_order=["May", "June", "July", "August"],
                 marker="o", markersize=4, linewidth=1.4, ax=ax_ts, legend=False)
    ann = dl.groupby("year", as_index=False)[var].mean()
    mk_trend(ax_ts, ann["year"].values, ann[var].values)
    ax_ts.set_xlabel("Year", weight="bold", fontsize=10)
    ax_ts.set_ylabel(ylabel(var), weight="bold", fontsize=10)
    ax_ts.set_xticks(xt); ax_ts.set_xticklabels(xt, rotation=45, weight="bold", fontsize=9)
    sns.despine(ax=ax_ts)
    ax_ts.text(-0.20, 1.02, "(%s)" % letter, transform=ax_ts.transAxes,
               fontsize=13, fontweight="bold", va="bottom", ha="left")

    ax_box.grid(**GRID)
    sns.boxplot(data=sub, x="season", y=var, hue="season",
                order=["Non-Wildfire Season", "Wildfire Season"], dodge=False,
                legend=False, showfliers=False, ax=ax_box, width=0.55,
                palette={"Non-Wildfire Season": "lightsteelblue", "Wildfire Season": "lightcoral"})
    for i, se in enumerate(["Non-Wildfire Season", "Wildfire Season"]):
        vals = sub.loc[sub["season"] == se, var].astype(float).values
        if len(vals) < 2:
            continue
        q1, q3 = np.percentile(vals, [25, 75]); med = float(np.median(vals))
        hw = 1.57 * (q3 - q1) / math.sqrt(len(vals))
        ax_box.plot([i - 0.18, i - 0.18], [med - hw, med + hw], color="#2E7D32",
                    lw=3, solid_capstyle="round", alpha=0.85, zorder=9)
        ax_box.plot(i - 0.18, med, marker="s", color="#2E7D32", markersize=5, zorder=10)
        m, h = mean_ci95(vals)
        if not np.isnan(m):
            ax_box.errorbar(i + 0.18, m, yerr=h, fmt="o", color="black",
                            markersize=4, capsize=3, lw=1.2, zorder=10)
            ax_box.annotate("%.*f" % (DEC[var], m), (i + 0.18, m), xytext=(6, 6),
                            textcoords="offset points", fontsize=8, fontweight="bold")
    ax_box.set_title(""); ax_box.set_xlabel(""); ax_box.set_ylabel("")
    lo = float(np.nanmin(sub[var])); hi = float(np.nanmax(sub[var])); span = hi - lo or 1
    ax_box.set_ylim(lo - 0.04 * span, hi + 0.20 * span); sns.despine(ax=ax_box)
    stars = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "n.s."
    yb = hi + 0.08 * span; yt = 0.015 * span
    ax_box.plot([0, 0, 1, 1], [yb - yt, yb, yb, yb - yt], color="black", lw=1.2, zorder=11)
    ax_box.text(0.5, yb + 0.005 * span, stars, ha="center", va="bottom",
                fontsize=12, fontweight="bold", zorder=12)
    ax_box.set_xticks([0, 1])
    ax_box.set_xticklabels(["Non-Wildfire\nSeason", "Wildfire\nSeason"], fontweight="bold", fontsize=9)


def legend_axis(fig):
    pal = sns.color_palette("colorblind", 4)
    r0 = [Line2D([0], [0], color=pal[i], lw=2, marker="o", ms=6, label=l)
          for i, l in enumerate(["May", "June", "July", "August"])]
    r0 += [Line2D([0], [0], color="black", lw=1.2, ls="--", label="Sen's slope (MK)"),
           mpatches.Patch(facecolor="lightsteelblue", edgecolor="black", label="Non-Wildfire"),
           mpatches.Patch(facecolor="lightcoral", edgecolor="black", label="Wildfire")]
    r1 = [mpatches.Patch(facecolor="white", edgecolor="black", label="Box: IQR (Q1-Q3)"),
          Line2D([0], [0], color="#2E7D32", lw=3, marker="s", ms=6, label="Median ± 95% CI"),
          Line2D([0], [0], color="black", lw=1.5, marker="o", ms=6, markerfacecolor="black", label="Mean ± 95% CI"),
          Line2D([0], [0], color="none", label="*** q < 0.001"),
          Line2D([0], [0], color="none", label="**  q < 0.01"),
          Line2D([0], [0], color="none", label="*    q < 0.05"),
          Line2D([0], [0], color="none", label="n.s. = not significant")]
    hs = []
    for a, b in zip(r0, r1):
        hs += [a, b]
    lax = fig.add_axes([0.04, 0.945, 0.92, 0.05]); lax.set_axis_off()
    leg = lax.legend(handles=hs, loc="center", ncol=7, frameon=False, fontsize=8,
                     handletextpad=0.5, columnspacing=1.0, borderpad=0.3, labelspacing=0.6)
    for t in leg.get_texts():
        t.set_fontfamily("serif"); t.set_fontweight("bold")


def build_group(df, stats, name, vars3, letters):
    fig = plt.figure(figsize=(7.6, 9.6))
    legend_axis(fig)
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=0.55, wspace=0.18,
                          left=0.10, right=0.97, top=0.90, bottom=0.05)
    for i, (var, letter) in enumerate(zip(vars3, letters)):
        panel(df, var, letter, fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]), stats[var]["q"])
    base = os.path.join(OUTDIR, "Fig12_%s_%s" % (name, "".join(letters)))
    fig.savefig(base + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(base + ".jpg", bbox_inches="tight", dpi=300, pil_kwargs={"quality": 95})
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("  saved", base + ".{png,jpg,pdf}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = infill_recent_year(load_merged())
    stats = season_stats(df)
    print("=" * 80)
    print(" STATISTICAL VERIFICATION - wildfire (May-Aug) vs non-wildfire, BC 2000-2024")
    print("=" * 80)
    print(" %-18s %9s %9s %10s %10s %8s %8s %4s" %
          ("Variable", "mean_W", "mean_NW", "Welch p", "BH q", "Cohen d", "Cliff d", "sig"))
    for v in ALLVARS:
        r = stats[v]
        sig = "***" if r["q"] < 0.001 else "**" if r["q"] < 0.01 else "*" if r["q"] < 0.05 else "n.s."
        dec = DEC[v]
        print(" %-18s %9.*f %9.*f %10.2e %10.2e %8.2f %8.2f %4s" %
              (v, dec, r["meanW"], dec, r["meanN"], r["welch_p"], r["q"], r["d"], r["delta"], sig))
    print("=" * 80)
    for name, vars3, letters in GROUPS:
        build_group(df, stats, name, vars3, letters)
    print("Done. Three composite images written to", OUTDIR)


if __name__ == "__main__":
    main()
