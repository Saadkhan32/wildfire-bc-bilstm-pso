from __future__ import annotations
import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                     "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 9, "font.family": "DejaVu Sans"})
_TK_ROOT = None
def _tk():
    global _TK_ROOT
    if _TK_ROOT is None:
        _TK_ROOT = Tk()
        _TK_ROOT.withdraw()
    return _TK_ROOT
def ask_directory(title: str, initial: str | None = None) -> Path | None:
    _tk()
    p = filedialog.askdirectory(title=title, initialdir=initial or "")
    return Path(p) if p else None
def ask_file(title: str, filetypes, initial: str | None = None) -> Path | None:
    _tk()
    p = filedialog.askopenfilename(title=title, filetypes=filetypes,
                                   initialdir=initial or "")
    return Path(p) if p else None
def info(title: str, msg: str):
    _tk()
    messagebox.showinfo(title, msg)
def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    return (x.mean() - y.mean()) / sp
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    nx, ny = len(x), len(y)
    if nx * ny > 5_000_000:
        combined = np.concatenate([x, y])
        ranks = pd.Series(combined).rank().values
        rank_x = ranks[:nx].sum()
        U = rank_x - nx * (nx + 1) / 2
        return (2 * U) / (nx * ny) - 1
    diffs = np.sign(x[:, None] - y[None, :])
    return float(diffs.mean())
def cliffs_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"
def cohens_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"
def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * n / (np.arange(1, n + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)
    q = np.empty(n)
    q[order] = q_ranked
    return q, q < alpha
CLIMATE_CANDIDATES = [
    "tmax", "Tmax", "TMAX", "tmean", "Tmean",
    "precipitation", "Precipitation", "Precip", "PRECIP", "ppt",
    "soil_moisture", "Soil_Moisture", "soilmoist", "SoilMoist", "sm",
    "aet", "AET", "Aet",
    "pdsi", "PDSI", "Pdsi",
    "ndvi", "NDVI", "Ndvi",
    "twi", "TWI", "Twi",
    "wind", "Wind", "WIND", "windspeed",
    "relhum", "RelHum", "RH",
]
def find_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    found = []
    seen = set()
    for c in candidates:
        if c in df.columns and c.lower() not in seen:
            found.append(c)
            seen.add(c.lower())
    return found
def phase_d(project_root: Path) -> dict:
    print("\n" + "=" * 75)
    print("PHASE D  - Welch's t + BH-FDR + Cohen's d + Cliff's delta  (Comment 12)")
    print("=" * 75)
    fire_shp = ask_file(
        "Phase D: Fire-presence shapefile (with climate columns)",
        filetypes=[("Shapefiles", "*.shp"), ("CSV", "*.csv"), ("All", "*.*")],
        initial=str(project_root / "data" / "processed"),
    )
    if not fire_shp:
        print("Phase D skipped (no fire file).")
        return {"status": "skipped"}
    nonfire_shp = ask_file(
        "Phase D: Non-fire (background) shapefile (with climate columns)",
        filetypes=[("Shapefiles", "*.shp"), ("CSV", "*.csv"), ("All", "*.*")],
        initial=str(fire_shp.parent),
    )
    if not nonfire_shp:
        print("Phase D skipped (no non-fire file).")
        return {"status": "skipped"}
    def load(p: Path) -> pd.DataFrame:
        if p.suffix.lower() == ".shp":
            import geopandas as gpd
            return pd.DataFrame(gpd.read_file(p).drop(columns="geometry", errors="ignore"))
        return pd.read_csv(p)
    df_fire = load(fire_shp)
    df_non = load(nonfire_shp)
    print(f"  fires   : {len(df_fire):,} rows, {len(df_fire.columns)} cols")
    print(f"  non-fire: {len(df_non):,} rows, {len(df_non.columns)} cols")
    common = sorted(set(df_fire.columns) & set(df_non.columns))
    climate_cols = find_columns(df_fire[common], CLIMATE_CANDIDATES)
    if not climate_cols:
        climate_cols = [
            c for c in common
            if pd.api.types.is_numeric_dtype(df_fire[c])
            and pd.api.types.is_numeric_dtype(df_non[c])
            and df_fire[c].notna().sum() > 50 and df_non[c].notna().sum() > 50
        ]
    print(f"  testing {len(climate_cols)} predictors: {climate_cols}")
    from scipy import stats as sps
    rows = []
    for col in climate_cols:
        x = df_fire[col].astype(float).dropna().values
        y = df_non[col].astype(float).dropna().values
        if len(x) < 5 or len(y) < 5:
            continue
        sw_x = sps.shapiro(x[:5000]).pvalue if len(x) >= 3 else np.nan
        sw_y = sps.shapiro(y[:5000]).pvalue if len(y) >= 3 else np.nan
        lev = sps.levene(x, y, center="median").pvalue
        t_stat, t_p = sps.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        u_stat, u_p = sps.mannwhitneyu(x, y, alternative="two-sided")
        d = cohens_d(x, y)
        delta = cliffs_delta(x, y)
        rows.append({
            "predictor": col,
            "n_fire": len(x),
            "n_nonfire": len(y),
            "fire_mean": float(np.mean(x)),
            "nonfire_mean": float(np.mean(y)),
            "fire_sd": float(np.std(x, ddof=1)),
            "nonfire_sd": float(np.std(y, ddof=1)),
            "shapiro_p_fire": float(sw_x) if not np.isnan(sw_x) else None,
            "shapiro_p_nonfire": float(sw_y) if not np.isnan(sw_y) else None,
            "levene_p": float(lev),
            "welch_t": float(t_stat),
            "welch_p": float(t_p),
            "mw_U": float(u_stat),
            "mw_p": float(u_p),
            "cohens_d": float(d),
            "cohens_d_mag": cohens_magnitude(d),
            "cliffs_delta": float(delta),
            "cliffs_delta_mag": cliffs_magnitude(delta),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("  No predictors testable.")
        return {"status": "no_data"}
    q_welch, rej_welch = benjamini_hochberg(df["welch_p"].values, alpha=0.05)
    df["welch_q_BH"] = q_welch
    df["welch_significant_q05"] = rej_welch
    q_mw, rej_mw = benjamini_hochberg(df["mw_p"].values, alpha=0.05)
    df["mw_q_BH"] = q_mw
    df["mw_significant_q05"] = rej_mw
    df["abs_cliffs"] = df["cliffs_delta"].abs()
    df = df.sort_values("abs_cliffs", ascending=False).drop(columns="abs_cliffs")
    out = project_root / "tables" / "T_welch_fdr_effect_sizes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6g")
    print(f"  wrote -> {out}  ({len(df)} predictors)")
    return {"status": "ok", "n_predictors": int(len(df)), "out": str(out)}
def phase_e(project_root: Path) -> dict:
    print("\n" + "=" * 75)
    print("PHASE E  - Theil-Sen + Mann-Kendall on Fig 5a annual trend  (Comment 15b)")
    print("=" * 75)
    ts_file = ask_file(
        "Phase E: Annual burned-area time series CSV (year, area_ha)",
        filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        initial=str(project_root / "data" / "processed"),
    )
    if not ts_file:
        print("  No CSV picked. Attempting to derive annual burned area from "
              "fire_centroids_70ha.shp ...")
        shp = project_root / "data" / "processed" / "fire_centroids_70ha.shp"
        if not shp.exists():
            print("  fire_centroids_70ha.shp not found. Phase E skipped.")
            return {"status": "skipped"}
        import geopandas as gpd
        gdf = gpd.read_file(shp)
        year_col = next((c for c in ["YEAR", "Year", "year", "FIRE_YEAR"] if c in gdf.columns), None)
        area_col = next((c for c in ["SIZE_HA", "AREA_HA", "AREA", "area_ha", "Hectares"] if c in gdf.columns), None)
        if not year_col or not area_col:
            print(f"  Could not find year/area columns in shapefile. Cols: {list(gdf.columns)[:20]}")
            return {"status": "skipped"}
        df_ts = (gdf.groupby(year_col)[area_col].sum()
                 .reset_index()
                 .rename(columns={year_col: "year", area_col: "area_ha"}))
    else:
        df_ts = pd.read_csv(ts_file)
        df_ts.columns = [c.strip().lower() for c in df_ts.columns]
        if "year" not in df_ts.columns:
            df_ts.rename(columns={df_ts.columns[0]: "year"}, inplace=True)
        if "area_ha" not in df_ts.columns:
            df_ts.rename(columns={df_ts.columns[1]: "area_ha"}, inplace=True)
    df_ts = df_ts.dropna().sort_values("year").reset_index(drop=True)
    df_ts["year"] = df_ts["year"].astype(int)
    df_ts["area_ha"] = df_ts["area_ha"].astype(float)
    print(f"  series: {len(df_ts)} years ({df_ts['year'].min()}-{df_ts['year'].max()})")
    x = df_ts["year"].values.astype(float)
    y = df_ts["area_ha"].values.astype(float)
    from scipy import stats as sps
    ols = sps.linregress(x, y)
    ts_slope, ts_intercept, ts_lo, ts_hi = sps.theilslopes(y, x, 0.95)
    try:
        import pymannkendall as mk
        mkres = mk.original_test(y)
        mk_trend = mkres.trend
        mk_p = mkres.p
        mk_tau = mkres.Tau
        mk_z = mkres.z
        mk_s = mkres.s
    except ImportError:
        n = len(y)
        s = 0
        for i in range(n - 1):
            s += np.sign(y[i + 1:] - y[i]).sum()
        var_s = n * (n - 1) * (2 * n + 5) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        mk_p = 2 * (1 - sps.norm.cdf(abs(z)))
        tau = s / (0.5 * n * (n - 1))
        mk_trend = "increasing" if z > 0 and mk_p < 0.05 else "decreasing" if z < 0 and mk_p < 0.05 else "no trend"
        mk_z, mk_s, mk_tau = float(z), int(s), float(tau)
    summary = pd.DataFrame([{
        "method": "OLS",
        "slope_ha_per_year": ols.slope,
        "intercept": ols.intercept,
        "p_value": ols.pvalue,
        "ci95_lo": ols.slope - 1.96 * ols.stderr,
        "ci95_hi": ols.slope + 1.96 * ols.stderr,
        "test_statistic": ols.rvalue,
        "notes": "Assumes Gaussian residuals - sensitive to outliers",
    }, {
        "method": "Theil-Sen",
        "slope_ha_per_year": ts_slope,
        "intercept": ts_intercept,
        "p_value": mk_p,
        "ci95_lo": ts_lo,
        "ci95_hi": ts_hi,
        "test_statistic": mk_z,
        "notes": "Non-parametric, robust to ~29% outliers; p-value from Mann-Kendall",
    }, {
        "method": "Mann-Kendall",
        "slope_ha_per_year": np.nan,
        "intercept": np.nan,
        "p_value": mk_p,
        "ci95_lo": np.nan,
        "ci95_hi": np.nan,
        "test_statistic": mk_z,
        "notes": f"Tau={mk_tau:.3f}, S={mk_s}, trend={mk_trend}",
    }])
    out_csv = project_root / "tables" / "T_fig5a_trend_methods.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False, float_format="%.6g")
    print(f"  wrote -> {out_csv}")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x, y / 1000.0, color="#c44e52", alpha=0.85, width=0.7,
           label="Annual burned area")
    ax.plot(x, (ols.intercept + ols.slope * x) / 1000.0, color="#888888",
            ls="--", lw=1.2, label=f"OLS slope = {ols.slope:.0f} ha/yr (p={ols.pvalue:.3f})")
    ax.plot(x, (ts_intercept + ts_slope * x) / 1000.0, color="#1f77b4",
            ls="-", lw=2.0,
            label=f"Theil-Sen slope = {ts_slope:.0f} ha/yr "
                  f"[{ts_lo:.0f}, {ts_hi:.0f}], MK p={mk_p:.3f}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Burned area (thousand ha)")
    ax.set_title(
        f"Fig 5a (revised). Annual burned area in BC, "
        f"{int(df_ts['year'].min())}-{int(df_ts['year'].max())}.\n"
        f"Mann-Kendall trend: {mk_trend} (tau={mk_tau:.3f})"
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_out = project_root / "figs" / "Fig5a_theilsen_revised.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote -> {fig_out}")
    return {"status": "ok",
            "ols_slope": float(ols.slope), "ols_p": float(ols.pvalue),
            "ts_slope": float(ts_slope), "mk_p": float(mk_p),
            "mk_trend": str(mk_trend),
            "csv": str(out_csv), "fig": str(fig_out)}
def phase_f(project_root: Path) -> dict:
    print("\n" + "=" * 75)
    print("PHASE F  - Fig 11 redraw with bootstrap 95% CI bands  (Comment 16)")
    print("=" * 75)
    oof_csv = ask_file(
        "Phase F: OOF predictions CSV (y_true, y_pred_oof, ...)",
        filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        initial=str(project_root),
    )
    if not oof_csv:
        print("  Phase F skipped (no OOF predictions).")
        return {"status": "skipped"}
    df = pd.read_csv(oof_csv)
    y_true_col = next((c for c in ["y_true", "label", "y", "fire", "target"] if c in df.columns), None)
    y_pred_col = next((c for c in ["y_pred_oof", "y_pred", "y_proba", "proba", "pred", "score"]
                       if c in df.columns), None)
    if not y_true_col or not y_pred_col:
        print(f"  Columns not auto-detected. Cols: {list(df.columns)}")
        return {"status": "no_columns"}
    y_true = df[y_true_col].astype(int).values
    y_score = df[y_pred_col].astype(float).values
    print(f"  loaded: {len(y_true):,} OOF predictions, "
          f"{y_true.sum():,} positives ({100*y_true.mean():.1f}% prevalence)")
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prec)
    n_iter = 1000
    rng = np.random.default_rng(42)
    fpr_grid = np.linspace(0, 1, 201)
    rec_grid = np.linspace(0, 1, 201)
    tpr_boot = np.empty((n_iter, len(fpr_grid)))
    prec_boot = np.empty((n_iter, len(rec_grid)))
    n = len(y_true)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        ys = y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            tpr_boot[i] = np.nan
            prec_boot[i] = np.nan
            continue
        f, t, _ = roc_curve(yt, ys)
        tpr_boot[i] = np.interp(fpr_grid, f, t)
        p, r, _ = precision_recall_curve(yt, ys)
        order = np.argsort(r)
        prec_boot[i] = np.interp(rec_grid, r[order], p[order])
    tpr_lo = np.nanpercentile(tpr_boot, 2.5, axis=0)
    tpr_hi = np.nanpercentile(tpr_boot, 97.5, axis=0)
    prec_lo = np.nanpercentile(prec_boot, 2.5, axis=0)
    prec_hi = np.nanpercentile(prec_boot, 97.5, axis=0)
    auc_boot = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]; ys = y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            auc_boot[i] = np.nan; continue
        f, t, _ = roc_curve(yt, ys)
        auc_boot[i] = auc(f, t)
    auc_lo, auc_hi = np.nanpercentile(auc_boot, [2.5, 97.5])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.0,
            label=f"BiLSTM-PSO (AUC = {roc_auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}])")
    ax.fill_between(fpr_grid, tpr_lo, tpr_hi, color="#1f77b4", alpha=0.20,
                    label="95% CI (1000 bootstrap)")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="Random")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(a) ROC curve with bootstrap 95% CI")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(rec, prec, color="#2ca02c", lw=2.0,
            label=f"BiLSTM-PSO (PR-AUC = {pr_auc:.3f})")
    ax.fill_between(rec_grid, prec_lo, prec_hi, color="#2ca02c", alpha=0.20,
                    label="95% CI (1000 bootstrap)")
    ax.axhline(y_true.mean(), color="#888", ls="--", lw=1,
               label=f"Random baseline = {y_true.mean():.2f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(b) Precision-Recall curve with bootstrap 95% CI")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.3)
    fig.suptitle("Fig 11 (revised). BiLSTM-PSO model discrimination "
                 "with bootstrap-derived 95% confidence intervals "
                 f"(n_OOF = {n:,}, n_bootstrap = {n_iter}).",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig_out = project_root / "figs" / "Fig11_revised_with_ci.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote -> {fig_out}")
    print(f"  AUC = {roc_auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]"
          f"   PR-AUC = {pr_auc:.3f}")
    return {"status": "ok",
            "auc": float(roc_auc), "auc_lo": float(auc_lo), "auc_hi": float(auc_hi),
            "pr_auc": float(pr_auc),
            "fig": str(fig_out)}
def phase_g(project_root: Path) -> dict:
    print("\n" + "=" * 75)
    print("PHASE G  - Regional District fire-density ranking + choropleth  (Comment 1)")
    print("=" * 75)
    rd_shp = ask_file(
        "Phase G: BC Regional Districts shapefile",
        filetypes=[("Shapefiles", "*.shp"), ("All", "*.*")],
        initial=str(project_root / "data"),
    )
    if not rd_shp:
        print("  Phase G skipped (no Regional Districts file).")
        return {"status": "skipped"}
    fire_shp = ask_file(
        "Phase G: Fire centroids shapefile (1,839 BC fires >= 70 ha)",
        filetypes=[("Shapefiles", "*.shp"), ("All", "*.*")],
        initial=str(project_root / "data" / "processed"),
    )
    if not fire_shp:
        print("  Phase G skipped (no fire centroids).")
        return {"status": "skipped"}
    import geopandas as gpd
    rd = gpd.read_file(rd_shp)
    fires = gpd.read_file(fire_shp)
    print(f"  RDs: {len(rd)}  fires: {len(fires)}")
    print(f"  RD CRS: {rd.crs}   fire CRS: {fires.crs}")
    target_crs = "EPSG:3005"
    if rd.crs is None:
        print("  WARNING: RD shapefile has no CRS - assuming EPSG:3005")
        rd.set_crs(target_crs, inplace=True)
    elif rd.crs.to_string() != target_crs:
        rd = rd.to_crs(target_crs)
    if fires.crs is None:
        fires.set_crs(target_crs, inplace=True)
    elif fires.crs.to_string() != target_crs:
        fires = fires.to_crs(target_crs)
    name_candidates = ["ADMIN_AREA_NAME", "REGIONAL_DISTRICT_NAME", "NAME",
                       "Name", "RD_NAME", "ADMIN_AREA_ABBREVIATION",
                       "AA_NAME", "RDName"]
    name_col = next((c for c in name_candidates if c in rd.columns), None)
    if name_col is None:
        for c in rd.columns:
            if rd[c].dtype == object:
                name_col = c; break
    print(f"  using RD name column: '{name_col}'")
    rd["_area_km2"] = rd.geometry.area / 1e6
    joined = gpd.sjoin(fires, rd[[name_col, "_area_km2", "geometry"]],
                       how="left", predicate="within")
    size_col = next((c for c in ["SIZE_HA", "AREA_HA", "AREA", "Hectares"]
                     if c in fires.columns), None)
    agg = joined.groupby(name_col).size().rename("n_fires").to_frame()
    if size_col:
        agg["burned_ha"] = joined.groupby(name_col)[size_col].sum()
        agg["mean_fire_size_ha"] = joined.groupby(name_col)[size_col].mean()
    agg = agg.merge(
        rd[[name_col, "_area_km2"]].set_index(name_col),
        left_index=True, right_index=True, how="left",
    )
    agg["fires_per_1000km2"] = agg["n_fires"] / agg["_area_km2"] * 1000
    if size_col:
        agg["burned_ha_per_km2"] = agg["burned_ha"] / agg["_area_km2"]
    agg = agg.rename(columns={"_area_km2": "rd_area_km2"})
    agg = agg.reset_index().rename(columns={name_col: "regional_district"})
    agg["fire_density_rank"] = agg["fires_per_1000km2"].rank(ascending=False).astype(int)
    agg = agg.sort_values("fires_per_1000km2", ascending=False)
    out_csv = project_root / "tables" / "T_regional_district_ranking.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False, float_format="%.4g")
    print(f"  wrote -> {out_csv}  ({len(agg)} RDs)")
    rd_plot = rd[[name_col, "geometry"]].merge(
        agg.rename(columns={"regional_district": name_col}),
        on=name_col, how="left",
    )
    rd_plot["fires_per_1000km2"] = rd_plot["fires_per_1000km2"].fillna(0)
    fig, ax = plt.subplots(figsize=(8, 9))
    rd_plot.plot(column="fires_per_1000km2", cmap="YlOrRd", linewidth=0.4,
                 edgecolor="white", ax=ax, legend=True,
                 legend_kwds={"label": "Fires per 1,000 km²", "shrink": 0.55})
    ax.set_axis_off()
    ax.set_title(
        "BC Regional District wildfire density, 2000-2024\n"
        f"(BiLSTM-PSO study area, n = {len(fires):,} fires >= 70 ha)",
        fontsize=11,
    )
    top5 = agg.head(5)
    for _, row in top5.iterrows():
        sub = rd_plot[rd_plot[name_col] == row["regional_district"]]
        if len(sub) == 0:
            continue
        c = sub.geometry.unary_union.centroid
        ax.annotate(f"#{int(row['fire_density_rank'])} {row['regional_district']}",
                    xy=(c.x, c.y), fontsize=7, ha="center",
                    color="#333", weight="bold")
    fig.tight_layout()
    fig_out = project_root / "figs" / "Fig_RD_choropleth.png"
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote -> {fig_out}")
    return {"status": "ok",
            "n_rds": int(len(agg)), "csv": str(out_csv), "fig": str(fig_out)}
def main():
    print("=" * 75)
    print(" Wildfire BC BiLSTM-PSO  -  Day 4 analysis  (Phases D, E, F, G)")
    print(f" Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 75)
    root = ask_directory(
        "Pick project root folder (wildfire-bc-bilstm-pso)",
        initial=os.path.expandvars(r"%USERPROFILE%\Documents"),
    )
    if not root:
        print("No project root selected. Exiting.")
        return 1
    print(f" project root: {root}\n")
    results = {"started": datetime.now().isoformat(), "project_root": str(root)}
    try:
        results["phase_d"] = phase_d(root)
    except Exception as e:
        results["phase_d"] = {"status": "error", "error": repr(e)}
        print(f"  Phase D ERROR: {e}")
    try:
        results["phase_e"] = phase_e(root)
    except Exception as e:
        results["phase_e"] = {"status": "error", "error": repr(e)}
        print(f"  Phase E ERROR: {e}")
    try:
        results["phase_f"] = phase_f(root)
    except Exception as e:
        results["phase_f"] = {"status": "error", "error": repr(e)}
        print(f"  Phase F ERROR: {e}")
    try:
        results["phase_g"] = phase_g(root)
    except Exception as e:
        results["phase_g"] = {"status": "error", "error": repr(e)}
        print(f"  Phase G ERROR: {e}")
    results["ended"] = datetime.now().isoformat()
    log_path = root / "tables" / "T_day4_automation_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 75)
    print(f" Run log: {log_path}")
    print("=" * 75)
    info("Day 4 complete",
         f"Phases D / E / F / G finished.\n\nLog: {log_path}\n\n"
         "Open tables\\ and figs\\ to inspect outputs.")
    return 0
if __name__ == "__main__":
    sys.exit(main())
