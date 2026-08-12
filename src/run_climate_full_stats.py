import os
import sys
import math
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats as sps
TNR_RC = {
    "font.family":          "Times New Roman",
    "font.serif":           ["Times New Roman", "Times", "serif"],
    "mathtext.fontset":     "stix",
    "font.size":            12,
    "font.weight":          "bold",
    "axes.labelweight":     "bold",
    "axes.titleweight":     "bold",
    "axes.titlesize":       12,
    "axes.labelsize":       12,
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,
    "legend.fontsize":      10,
    "legend.title_fontsize":11,
    "legend.frameon":       False,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
    "svg.fonttype":         "none",
}
plt.rcParams.update(TNR_RC)
sns.set_style("whitegrid")
sns.set_palette("colorblind")
GRID_KW = dict(color="grey", linewidth=0.5, alpha=0.3)
WILDFIRE_MONTHS = {5: "May", 6: "June", 7: "July", 8: "August"}
VARIABLE_RENAME = {
    "avg_relative_humidity": "relative_humidity",
}
UNIT_MAP = {
    "AET":                  "mm/month",
    "PDSI":                 "unitless",
    "soil_moisture":        "mm",
    "max_temperature":      "°C",
    "avg_temperature":      "°C",
    "min_temperature":      "°C",
    "avg_wind_speed":       "m/s",
    "relative_humidity":    "%",
    "precipitation":        "mm",
}
PRETTY_NAME = {
    "AET":                  "Evapotranspiration (mm/mo)",
    "PDSI":                 "Palmer Drought Severity Index",
    "soil_moisture":        "Soil moisture (mm)",
    "max_temperature":      "Max temperature (°C)",
    "avg_temperature":      "Mean temperature (°C)",
    "min_temperature":      "Min temperature (°C)",
    "avg_wind_speed":       "Wind speed (m/s)",
    "relative_humidity":    "Relative humidity (%)",
    "precipitation":        "Precipitation (mm)",
}
COMPOSITE_GROUPS = [
    ("temperature",  ["max_temperature", "avg_temperature", "min_temperature"]),
    ("hydrology",    ["AET",             "precipitation",   "soil_moisture"]),
    ("atmospheric",  ["relative_humidity","avg_wind_speed", "PDSI"]),
]
def fmt_p(p, decimals=3):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ""
    if p < 0.0001: return "<0.0001"
    if p < 0.001:  return f"{p:.4f}"
    return f"{p:.{decimals}f}"
def fmt_num(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    if abs(v) >= 1000: return f"{v:,.0f}"
    if abs(v) >= 100:  return f"{v:.1f}"
    return f"{v:.{decimals}f}"
def stars_for_q(q):
    if q is None or (isinstance(q, float) and np.isnan(q)): return "n.s."
    if q < 0.001: return "***"
    if q < 0.01:  return "**"
    if q < 0.05:  return "*"
    return "n.s."
_TK = None
def _tk():
    global _TK
    if _TK is None:
        _TK = tk.Tk(); _TK.withdraw()
    return _TK
def pick_merged_or_files():
    _tk()
    choice = messagebox.askyesno(
        "Climate data source",
        "Use the already-merged monthly CSV?\n\n"
        "Yes = pick merged_monthly_climate_2000_2024.csv (faster)\n"
        "No  = pick raw files (3 TerraClimate CSVs + wide-form xlsx OR GEE CSV)",
    )
    if choice:
        path = filedialog.askopenfilename(
            title="Pick merged_monthly_climate_2000_2024.csv",
            filetypes=[("CSV", "*.csv")],
        )
        return ("merged", Path(path) if path else None)
    paths = filedialog.askopenfilenames(
        title="Pick 3 TerraClimate CSVs + wide-form CSV (or xlsx)",
        filetypes=[("Data files", "*.csv *.xlsx *.xls")],
    )
    return ("raw", [Path(p) for p in paths] if paths else None)
def pick_project_root():
    _tk()
    p = filedialog.askdirectory(
        title="Pick project root (wildfire-bc-bilstm-pso)",
        initialdir=r"C:\Users\saadz\Documents",
    )
    return Path(p) if p else None
def remerge(paths):
    parts = []
    for p in paths:
        if p.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(p)
            df = df.rename(columns={c: c.strip() for c in df.columns})
            df = df.groupby(["year", "month"], as_index=False).mean(numeric_only=True)
            parts.append(("wide", df, p.name))
        else:
            df = pd.read_csv(p)
            if 'mean' in df.columns and 'variable' in df.columns:
                name = p.stem.replace("BC_TerraClimate_monthly_", "").replace("_2000_2024", "")
                upper = name.upper()
                if   "SOIL" in upper: name = "soil_moisture"
                elif "AET"  in upper: name = "AET"
                elif "PDSI" in upper: name = "PDSI"
                keep = df[["year", "month", "mean"]].rename(columns={"mean": name})
                keep = keep.groupby(["year", "month"], as_index=False).mean()
                parts.append(("terra", keep, p.name))
            elif 'year' in df.columns and 'month' in df.columns:
                df = df.select_dtypes(include=[np.number])
                df = df.groupby(["year", "month"], as_index=False).mean()
                parts.append(("wide", df, p.name))
                print(f"  [WIDE CSV] {p.name}: {df.shape[0]} rows, "
                      f"cols={[c for c in df.columns if c not in ('year','month')]}")
            else:
                print(f"  [WARN] unknown CSV format: {p.name}")
    wide_groups = {}
    for kind, df, name in parts:
        if kind != "wide":
            continue
        key = tuple(sorted(c for c in df.columns if c not in ("year", "month")))
        wide_groups.setdefault(key, []).append(df)
    wide_combined = []
    for key, group in wide_groups.items():
        if len(group) > 1:
            stacked = pd.concat(group, ignore_index=True)
            stacked = stacked.groupby(["year", "month"], as_index=False).mean()
            wide_combined.append(stacked)
        else:
            wide_combined.append(group[0])
    terra_parts = [df for kind, df, _ in parts if kind == "terra"]
    merged = None
    for part in terra_parts + wide_combined:
        merged = part if merged is None else pd.merge(merged, part,
                                                     on=["year", "month"],
                                                     how="outer")
    merged = merged.rename(columns=VARIABLE_RENAME)
    return merged.sort_values(["year", "month"]).reset_index(drop=True)
def infill_recent_year(df, target_year=None, clim_window=(2014, 2023),
                       scale_lo=0.5, scale_hi=2.0):
    if target_year is None:
        target_year = int(df["year"].max())
    num_cols = [c for c in df.columns if c not in ("year", "month")
                and pd.api.types.is_numeric_dtype(df[c])]
    missing = []
    scale_mismatch = []
    target_present = target_year in set(df["year"].astype(int).tolist())
    for var in num_cols:
        target_vals = (df.loc[df["year"] == target_year, var].dropna()
                       if target_present else pd.Series(dtype=float))
        clim_vals = df.loc[df["year"].between(*clim_window), var].dropna()
        if target_vals.empty:
            action = "missing"
        elif clim_vals.empty:
            action = "use_provided"
        else:
            t_abs = abs(target_vals.mean())
            c_abs = abs(clim_vals.mean())
            if c_abs < 1e-9:
                action = "use_provided"
            else:
                ratio = t_abs / c_abs
                if scale_lo <= ratio <= scale_hi:
                    action = "use_provided"
                else:
                    action = "scale_mismatch"
        if action == "use_provided":
            continue
        clim = (df[df["year"].between(*clim_window)]
                .groupby("month")[var].mean())
        for month in range(1, 13):
            mask = (df["year"] == target_year) & (df["month"] == month)
            if mask.any():
                df.loc[mask, var] = clim.get(month, np.nan)
            else:
                new_row = {c: np.nan for c in df.columns}
                new_row.update({"year": target_year, "month": month,
                                var: clim.get(month, np.nan)})
                df = pd.concat([df, pd.DataFrame([new_row])],
                               ignore_index=True)
        if action == "missing":
            missing.append(var)
        else:
            scale_mismatch.append(var)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df, missing, scale_mismatch
def cohens_d(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0: return np.nan
    return (x.mean() - y.mean()) / sp
def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0: return np.nan
    combined = np.concatenate([x, y])
    ranks = pd.Series(combined).rank().values
    rank_x = ranks[:nx].sum()
    U = rank_x - nx * (nx + 1) / 2
    return float((2 * U) / (nx * ny) - 1)
def cohens_d_mag(d):
    a = abs(d) if not np.isnan(d) else 0
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"
def cliffs_delta_mag(d):
    a = abs(d) if not np.isnan(d) else 0
    if a < 0.147: return "negligible"
    if a < 0.33:  return "small"
    if a < 0.474: return "medium"
    return "large"
def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0: return np.array([]), np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)
    q = np.empty(n); q[order] = q_ranked
    return q, q < alpha
def mean_ci95(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 2: return np.nan, np.nan
    m = x.mean(); se = x.std(ddof=1) / math.sqrt(n)
    t_crit = sps.t.ppf(0.975, df=n - 1)
    return m, t_crit * se
def variable_stats(df, var):
    sub = df[["year", "month", var]].dropna().copy()
    if sub.empty: return None
    sub["season"] = np.where(sub["month"].isin(WILDFIRE_MONTHS.keys()),
                             "Wildfire", "NonWildfire")
    x = sub.loc[sub["season"] == "Wildfire", var].astype(float).values
    y = sub.loc[sub["season"] == "NonWildfire", var].astype(float).values
    sw_x = sps.shapiro(x).pvalue if len(x) >= 3 else np.nan
    sw_y = sps.shapiro(y).pvalue if len(y) >= 3 else np.nan
    lev  = sps.levene(x, y, center="median").pvalue if min(len(x), len(y)) >= 2 else np.nan
    t_stat, t_p = sps.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    u_stat, u_p = sps.mannwhitneyu(x, y, alternative="two-sided")
    d = cohens_d(x, y); cdelta = cliffs_delta(x, y)
    mx, mx_ci = mean_ci95(x); my, my_ci = mean_ci95(y)
    return {
        "variable": var,
        "n_wildfire": int(len(x)), "n_nonwildfire": int(len(y)),
        "mean_wildfire": float(mx), "mean_nonwildfire": float(my),
        "ci95_wildfire_lo": float(mx - mx_ci) if not np.isnan(mx_ci) else np.nan,
        "ci95_wildfire_hi": float(mx + mx_ci) if not np.isnan(mx_ci) else np.nan,
        "ci95_nonfire_lo":  float(my - my_ci) if not np.isnan(my_ci) else np.nan,
        "ci95_nonfire_hi":  float(my + my_ci) if not np.isnan(my_ci) else np.nan,
        "diff_mean": float(mx - my),
        "shapiro_p_wild": float(sw_x), "shapiro_p_nonfire": float(sw_y),
        "levene_p": float(lev),
        "welch_t": float(t_stat), "welch_p": float(t_p),
        "mw_U": float(u_stat), "mw_p": float(u_p),
        "cohens_d": float(d), "cohens_d_mag": cohens_d_mag(d),
        "cliffs_delta": float(cdelta), "cliffs_delta_mag": cliffs_delta_mag(cdelta),
    }
def _render_panels(df, var, row, ax_ts, ax_box, small=False):
    sub = df[["year", "month", var]].dropna().copy()
    if sub.empty: return
    sub["MonthName"] = sub["month"].map(WILDFIRE_MONTHS)
    sub["season"] = np.where(sub["month"].isin(WILDFIRE_MONTHS.keys()),
                             "Wildfire Season", "Non-Wildfire Season")
    unit = UNIT_MAP.get(var, "")
    y_label = f"{var.replace('_',' ').title()}{' (' + unit + ')' if unit else ''}"
    ts_label_fs   = 10 if small else 12
    ts_tick_fs    = 9  if small else 11
    box_tick_fs   = 9  if small else 10
    mean_text_fs  = 8  if small else 9
    mk_text_fs    = 9  if small else 10
    stars_fs      = 12 if small else 14
    marker_size   = 4  if small else 6
    line_w        = 1.4 if small else 2.0
    df_line = sub[sub["MonthName"].notna()].copy()
    years = (np.sort(df_line["year"].unique()) if not df_line.empty
             else np.sort(sub["year"].unique()))
    xticks = years[::max(1, len(years)//8 or 1)]
    ax_ts.grid(**GRID_KW)
    if not df_line.empty:
        sns.lineplot(data=df_line, x="year", y=var, hue="MonthName",
                     hue_order=["May", "June", "July", "August"],
                     marker="o", markersize=marker_size,
                     linewidth=line_w, ax=ax_ts, legend=False)
        annual = df_line.groupby("year", as_index=False)[var].mean()
        _add_mk_trend(ax_ts, annual["year"].values, annual[var].values,
                      fontsize=mk_text_fs)
    else:
        g = sub.groupby("year", as_index=False)[var].mean()
        ax_ts.plot(g["year"], g[var], marker="o", lw=line_w)
        _add_mk_trend(ax_ts, g["year"].values, g[var].values,
                      fontsize=mk_text_fs)
    ax_ts.set_xlabel("Year", weight="bold", fontsize=ts_label_fs)
    ax_ts.set_ylabel(y_label, weight="bold", fontsize=ts_label_fs)
    ax_ts.set_xticks(xticks)
    ax_ts.set_xticklabels(xticks, rotation=45, weight="bold", fontsize=ts_tick_fs)
    sns.despine(ax=ax_ts)
    ax_box.grid(**GRID_KW)
    sns.boxplot(
        data=sub, x="season", y=var, hue="season",
        order=["Non-Wildfire Season", "Wildfire Season"],
        dodge=False, legend=False, notch=False,
        palette={"Non-Wildfire Season": "lightsteelblue",
                 "Wildfire Season":     "lightcoral"},
        showfliers=False, ax=ax_box, width=0.55,
    )
    for i, season in enumerate(["Non-Wildfire Season", "Wildfire Season"]):
        vals = sub.loc[sub["season"] == season, var].astype(float).values
        if len(vals) < 2: continue
        q1, q3 = np.percentile(vals, [25, 75])
        med = float(np.median(vals))
        med_hw = 1.57 * (q3 - q1) / math.sqrt(len(vals))
        ax_box.plot([i - 0.18, i - 0.18], [med - med_hw, med + med_hw],
                    color="#2E7D32", lw=3.0, solid_capstyle="round",
                    alpha=0.85, zorder=9)
        ax_box.plot(i - 0.18, med, marker="s", color="#2E7D32",
                    markersize=5, zorder=10)
        m, hw = mean_ci95(vals)
        if not np.isnan(m):
            ax_box.errorbar(i + 0.18, m, yerr=hw, fmt="o", color="black",
                            markersize=marker_size, capsize=3, lw=1.2,
                            zorder=10)
            ax_box.annotate(f"{m:.2f}" if small else f"{m:.3f}",
                            (i + 0.18, m), xytext=(6, 6),
                            textcoords="offset points", fontsize=mean_text_fs,
                            fontweight="bold")
    ax_box.set_title(""); ax_box.set_xlabel(""); ax_box.set_ylabel("")
    overall_min = float(np.nanmin(sub[var].values))
    overall_max = float(np.nanmax(sub[var].values))
    span = overall_max - overall_min or 1.0
    ax_box.set_ylim(overall_min - 0.04 * span, overall_max + 0.20 * span)
    sns.despine(ax=ax_box)
    q_w = row.get("welch_q_BH", np.nan)
    stars = stars_for_q(q_w)
    y_bracket = overall_max + 0.08 * span
    y_tick = 0.015 * span
    ax_box.plot([0, 0, 1, 1],
                [y_bracket - y_tick, y_bracket, y_bracket, y_bracket - y_tick],
                color="black", lw=1.2, zorder=11)
    ax_box.text(0.5, y_bracket + 0.005 * span, stars,
                ha="center", va="bottom",
                fontsize=stars_fs, fontweight="bold", zorder=12)
    ax_box.set_xticks([0, 1])
    ax_box.set_xticklabels(
        ["Non-Wildfire\nSeason", "Wildfire\nSeason"],
        fontweight="bold", fontsize=box_tick_fs,
    )
def _add_mk_trend(ax, x, y, fontsize=10):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = len(y)
    if n < 3: return
    S = 0
    for k in range(n - 1):
        S += np.sign(y[k+1:] - y[k]).sum()
    unique_y, counts = np.unique(y, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2*counts + 5))
    varS = (n*(n-1)*(2*n+5) - tie) / 18.0
    if varS == 0: return
    if S > 0:   Z = (S - 1) / math.sqrt(varS)
    elif S < 0: Z = (S + 1) / math.sqrt(varS)
    else:       Z = 0.0
    p = 2 * (1 - sps.norm.cdf(abs(Z)))
    tau = S / (n * (n - 1) / 2.0)
    slopes = []
    for i in range(n - 1):
        dx = x[i+1:] - x[i]; dy = y[i+1:] - y[i]
        m = dx != 0
        if np.any(m): slopes.extend((dy[m] / dx[m]).tolist())
    if not slopes: return
    slope = float(np.median(slopes))
    intercept = float(np.nanmean(y) - slope * np.nanmean(x))
    xs = np.array([x.min(), x.max()], dtype=float)
    ax.plot(xs, intercept + slope * xs, "--", color="black", lw=1, zorder=3)
    if p < 0.0001:  p_str = "p < 0.0001"
    elif p < 0.001: p_str = f"p = {p:.4f}"
    else:           p_str = f"p = {p:.3f}"
    ax.text(0.5, 0.02, f"τ = {tau:.3f}, {p_str}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8), zorder=4)
def plot_variable_v2(df, row, out_folder):
    var = row["variable"]
    fig, (ax_ts, ax_box) = plt.subplots(
        1, 2, figsize=(10, 4),
        gridspec_kw={"width_ratios": [3, 1]},
        constrained_layout=True,
    )
    _render_panels(df, var, row, ax_ts, ax_box, small=False)
    base = os.path.join(out_folder, var)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight",
                    dpi=(300 if ext == "png" else None))
    plt.close(fig)
    print(f"  [FIG] {var}.{{svg,pdf,png}}")
def build_composite_figure(df, stats_df, group_name, var_list, out_folder):
    fig = plt.figure(figsize=(7.5, 9.5))
    fig.suptitle(
        f"Figure 11 — Monthly climate variables, BC 2000–2024  "
        f"({group_name.capitalize()} group)",
        fontsize=12, weight="bold", y=0.99,
    )
    gs = fig.add_gridspec(
        3, 2, width_ratios=[3, 1],
        hspace=0.55, wspace=0.18,
        left=0.10, right=0.97, top=0.94, bottom=0.04,
    )
    for i, var in enumerate(var_list):
        if var not in df.columns:
            print(f"  [WARN] {var} not in dataframe, skipping panel")
            continue
        row = (stats_df[stats_df["variable"] == var].iloc[0].to_dict()
               if var in stats_df["variable"].values else {})
        ax_ts = fig.add_subplot(gs[i, 0])
        ax_box = fig.add_subplot(gs[i, 1])
        _render_panels(df, var, row, ax_ts, ax_box, small=True)
    base = os.path.join(out_folder, f"Fig11_{group_name}")
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base}.jpg", bbox_inches="tight", dpi=300,
                pil_kwargs={"quality": 95})
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  [COMPOSITE] Fig11_{group_name}.{{png,jpg,pdf}}")
def build_legend_figure(out_folder):
    palette = sns.color_palette("colorblind", 4)
    row0 = [
        Line2D([0], [0], color=palette[0], lw=2, marker="o", markersize=6, label="May"),
        Line2D([0], [0], color=palette[1], lw=2, marker="o", markersize=6, label="June"),
        Line2D([0], [0], color=palette[2], lw=2, marker="o", markersize=6, label="July"),
        Line2D([0], [0], color=palette[3], lw=2, marker="o", markersize=6, label="August"),
        Line2D([0], [0], color="black", lw=1.2, linestyle="--",
               label="Sen's slope (MK)"),
        mpatches.Patch(facecolor="lightsteelblue", edgecolor="black",
                       label="Non-Wildfire"),
        mpatches.Patch(facecolor="lightcoral", edgecolor="black",
                       label="Wildfire"),
    ]
    row1 = [
        mpatches.Patch(facecolor="white", edgecolor="black",
                       label="Box: IQR (Q1–Q3)"),
        Line2D([0], [0], color="#2E7D32", lw=3.0, marker="s", markersize=6,
               label="Median ± 95% CI"),
        Line2D([0], [0], color="black", lw=1.5, marker="o", markersize=6,
               markerfacecolor="black", label="Mean ± 95% CI"),
        Line2D([0], [0], color="none", label="*** q < 0.001"),
        Line2D([0], [0], color="none", label="**  q < 0.01"),
        Line2D([0], [0], color="none", label="*    q < 0.05"),
        Line2D([0], [0], color="none", label="n.s. = not significant"),
    ]
    handles = []
    for h0, h1 in zip(row0, row1):
        handles.append(h0); handles.append(h1)
    fig, ax = plt.subplots(figsize=(12, 1.5))
    ax.set_axis_off()
    leg = ax.legend(
        handles=handles, loc="center", ncol=7,
        frameon=False, fontsize=10,
        handletextpad=0.5, columnspacing=1.2,
        borderpad=0.3, labelspacing=0.7,
    )
    for txt in leg.get_texts():
        txt.set_fontfamily("Times New Roman")
        txt.set_fontweight("bold")
    fig.tight_layout()
    base = os.path.join(out_folder, "Fig11_legend")
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base}.jpg", bbox_inches="tight", dpi=300,
                pil_kwargs={"quality": 95})
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  [LEGEND] Fig11_legend.{{png,jpg,pdf,svg}}")
def write_publication_table(stats_df, out_tables):
    rows = []
    for _, r in stats_df.iterrows():
        var = r["variable"]
        rows.append({
            "Variable":              PRETTY_NAME.get(var, var),
            "n (W)":                 int(r["n_wildfire"]),
            "n (NW)":                int(r["n_nonwildfire"]),
            "Mean ± 95% CI (W)":     f"{r['mean_wildfire']:.2f} [{r['ci95_wildfire_lo']:.2f}, {r['ci95_wildfire_hi']:.2f}]",
            "Mean ± 95% CI (NW)":    f"{r['mean_nonwildfire']:.2f} [{r['ci95_nonfire_lo']:.2f}, {r['ci95_nonfire_hi']:.2f}]",
            "Shapiro-Wilk p (W)":    fmt_p(r["shapiro_p_wild"]),
            "Shapiro-Wilk p (NW)":   fmt_p(r["shapiro_p_nonfire"]),
            "Levene p":              fmt_p(r["levene_p"]),
            "Welch t":               fmt_num(r["welch_t"]),
            "Welch p":               fmt_p(r["welch_p"]),
            "Welch q (BH)":          fmt_p(r["welch_q_BH"]),
            "M-W U":                 fmt_num(r["mw_U"]),
            "M-W p":                 fmt_p(r["mw_p"]),
            "M-W q (BH)":            fmt_p(r["mw_q_BH"]),
            "Cohen's d":             fmt_num(r["cohens_d"]),
            "Magnitude (d)":         r["cohens_d_mag"],
            "Cliff's δ":             fmt_num(r["cliffs_delta"]),
            "Magnitude (δ)":         r["cliffs_delta_mag"],
            "FDR-q significance":    stars_for_q(r["welch_q_BH"]),
        })
    pub = pd.DataFrame(rows)
    csv_path = out_tables / "T_climate_stats_publication.csv"
    pub.to_csv(csv_path, index=False)
    print(f" [PUBLICATION CSV] -> {csv_path}")
    html_path = out_tables / "T_climate_stats_publication.html"
    css = """
    <style>
      body { font-family: 'Times New Roman', serif; padding: 20px; }
      h2   { font-size: 16px; }
      table { border-collapse: collapse; font-size: 11px; font-family: 'Times New Roman', serif; }
      th, td { border: 1px solid #888; padding: 5px 8px; text-align: center; }
      th { background: #2E4A7B; color: white; font-weight: bold; }
      tr:nth-child(even) { background: #F4F6F9; }
      td:first-child, th:first-child { text-align: left; font-weight: bold; }
      caption { caption-side: top; font-weight: bold; font-size: 12px;
                text-align: left; padding-bottom: 8px; font-family: 'Times New Roman', serif; }
    </style>
    """
    caption = (
        "Table SX. Statistical comparison of nine monthly climate variables "
        "between wildfire (May–August) and non-wildfire seasons in British "
        "Columbia, 2000–2024. Maximum/minimum/mean temperature, wind speed, "
        "and precipitation from TerraClimate (Abatzoglou et al. 2018); "
        "relative humidity derived from ERA5-Land 2m temperature and "
        "2m dewpoint temperature via the Magnus formula (Alduchov &amp; "
        "Eskridge 1996); PDSI, AET, and soil moisture from TerraClimate. "
        "Assumption tests: Shapiro–Wilk (normality, per group), Levene "
        "(homogeneity of variance). Parametric inference: Welch's t-test. "
        "Non-parametric inference: Mann–Whitney U. Multiple-testing "
        "correction: Benjamini–Hochberg FDR at q &lt; 0.05 across the nine "
        "variables. Effect sizes: Cohen's d and Cliff's δ, with magnitude "
        "labels per Cohen (1988) and Romano et al. (2006). Significance "
        "code: *** q &lt; 0.001, ** q &lt; 0.01, * q &lt; 0.05, n.s. = "
        "not significant."
    )
    html_table = pub.to_html(index=False, escape=False, border=0)
    html_table = html_table.replace(
        "<table border=\"0\" class=\"dataframe\">",
        f"<table>\n  <caption>{caption}</caption>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'>{css}</head>"
                f"<body><h2>Climate seasonal contrasts — publication table</h2>"
                f"{html_table}</body></html>")
    print(f" [PUBLICATION HTML] -> {html_path}")
def main():
    print("=" * 75)
    print(" Climate Stats - FULL battery for Comment 12 (monthly-aggregate)")
    print(f" Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 75)
    mode, picks = pick_merged_or_files()
    if not picks:
        print("No files picked. Exiting."); return 1
    if mode == "merged":
        df = pd.read_csv(picks)
        df = df.rename(columns=VARIABLE_RENAME)
        print(f" loaded merged: {picks.name}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    else:
        df = remerge(picks)
        print(f" re-merged {len(picks)} files: shape {df.shape}")
    df, infilled_missing, infilled_scale = infill_recent_year(
        df, clim_window=(2014, 2023), scale_lo=0.5, scale_hi=2.0)
    target_year = int(df["year"].max())
    if infilled_missing:
        print(f" [INFILL-MISSING] {target_year} missing for: {infilled_missing}")
        print(f"                  filled with 2014-2023 monthly climatology")
    if infilled_scale:
        print(f" [INFILL-SCALE]   {target_year} scale-mismatch with climatology for:")
        print(f"                  {infilled_scale}")
        print(f"                  replaced with 2014-2023 monthly climatology")
    if not infilled_missing and not infilled_scale:
        print(f" [INFILL] not needed - {target_year} fully populated and scale-matched")
    root = pick_project_root()
    if not root:
        print("No project root selected. Exiting."); return 1
    out_tables = root / "tables"
    out_figs = root / "figs" / "climate_v2"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    num_cols = [c for c in df.columns if c not in ("year", "month")
                and pd.api.types.is_numeric_dtype(df[c])]
    print(f" testing {len(num_cols)} variables: {num_cols}")
    rows = []
    for var in num_cols:
        r = variable_stats(df, var)
        if r is not None: rows.append(r)
    out = pd.DataFrame(rows)
    q_w, _ = benjamini_hochberg(out["welch_p"].values, alpha=0.05)
    out["welch_q_BH"] = q_w
    out["welch_sig_q05"] = out["welch_q_BH"] < 0.05
    q_mw, _ = benjamini_hochberg(out["mw_p"].values, alpha=0.05)
    out["mw_q_BH"] = q_mw
    out["mw_sig_q05"] = out["mw_q_BH"] < 0.05
    out["abs_cliffs"] = out["cliffs_delta"].abs()
    out = out.sort_values("abs_cliffs", ascending=False).drop(columns="abs_cliffs")
    raw_csv = out_tables / "T_climate_stats_full.csv"
    out.to_csv(raw_csv, index=False, float_format="%.6g")
    print(f"\n [RAW CSV] -> {raw_csv}")
    write_publication_table(out, out_tables)
    print("\n Summary (top of table):")
    print(out[["variable", "n_wildfire", "n_nonwildfire",
               "shapiro_p_wild", "shapiro_p_nonfire", "levene_p",
               "welch_p", "welch_q_BH", "mw_p", "mw_q_BH",
               "cohens_d", "cohens_d_mag",
               "cliffs_delta", "cliffs_delta_mag"]
              ].to_string(index=False))
    print(f"\n Single-variable figures -> {out_figs} ...")
    for _, row in out.iterrows():
        plot_variable_v2(df, row.to_dict(), str(out_figs))
    print(f"\n Composite 3-panel figures (one Word page each) -> {out_figs} ...")
    for group_name, var_list in COMPOSITE_GROUPS:
        build_composite_figure(df, out, group_name, var_list, str(out_figs))
    print(f"\n Two-row horizontal legend -> {out_figs} ...")
    build_legend_figure(str(out_figs))
    print("\n" + "=" * 75)
    print(f" DONE.")
    print(f"   Raw CSV:           {raw_csv}")
    print(f"   Publication CSV:   {out_tables / 'T_climate_stats_publication.csv'}")
    print(f"   Publication HTML:  {out_tables / 'T_climate_stats_publication.html'}")
    print(f"   Single-var figs:   {out_figs}  ({len(out)} variables x 3 formats)")
    print(f"   Composite figs:    {out_figs}  (3 page-sized PNG+JPG+PDF)")
    print(f"   2-row legend:      {out_figs / 'Fig11_legend.png'}")
    print("=" * 75)
    try:
        _tk()
        messagebox.showinfo(
            "Climate stats complete",
            "All outputs written. Check console for infill status."
        )
    except Exception:
        pass
    return 0
if __name__ == "__main__":
    sys.exit(main())
