from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.stats import pearsonr, spearmanr
ROOT = Path(r".")
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIGS = ROOT / "figs"
TABLES = ROOT / "tables"
for d in (DATA_PROC, FIGS, TABLES): d.mkdir(parents=True, exist_ok=True)
raw = (DATA_RAW / "meiv2.txt").read_text()
m_map = {1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:11, 11:12, 12:1}  
data = []
for ln in raw.splitlines():
    parts = ln.split()
    if len(parts) == 13 and parts[0].isdigit() and 1900 <= int(parts[0]) <= 2100:
        year = int(parts[0])
        for slot in range(1, 13):
            try:
                v = float(parts[slot])
                m = m_map[slot]
                y = year + 1 if slot == 12 else year  
                if v == -999.0: v = np.nan
                data.append((y, m, v))
            except ValueError:
                continue
mei = pd.DataFrame(data, columns=["year","month","mei"])
mei = mei[(mei.year >= 2000) & (mei.year <= 2024)].sort_values(["year","month"]).reset_index(drop=True)
print(f"MEI rows: {len(mei)}, range {mei.year.min()}-{mei.year.max()}")
raw = (DATA_RAW / "pdo.csv").read_text()
rows = []
for ln in raw.splitlines():
    parts = ln.split()
    if len(parts) == 13 and parts[0].isdigit() and len(parts[0]) == 4:
        try:
            year = int(parts[0])
            for m in range(1, 13):
                v = float(parts[m])
                if v == 99.99 or v == -99.99 or v == -9999.0:
                    v = np.nan
                rows.append((year, m, v))
        except ValueError:
            continue
pdo = pd.DataFrame(rows, columns=["year","month","pdo"])
pdo = pdo[(pdo.year >= 2000) & (pdo.year <= 2024)].sort_values(["year","month"]).reset_index(drop=True)
print(f"PDO rows: {len(pdo)}, range {pdo.year.min()}-{pdo.year.max()}")
cal = pd.MultiIndex.from_product([range(2000,2025), range(1,13)], names=["year","month"]).to_frame(index=False)
tele = cal.merge(mei, on=["year","month"], how="left").merge(pdo, on=["year","month"], how="left")
for lag in [3, 6, 9, 12]:
    tele[f"mei_lag{lag:02d}"] = tele["mei"].shift(lag)
    tele[f"pdo_lag{lag:02d}"] = tele["pdo"].shift(lag)
out = DATA_PROC / "teleconnections_monthly.csv"
tele.to_csv(out, index=False)
print(f"Wrote {out}  ({len(tele)} rows, {len(tele.columns)} cols)")
print(f"  MEI coverage: {tele.mei.notna().sum()}/{len(tele)} months")
print(f"  PDO coverage: {tele.pdo.notna().sum()}/{len(tele)} months")
tele["date"] = pd.to_datetime(dict(year=tele.year, month=tele.month, day=1))
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for ax, col, lbl, c in zip(axes, ["mei","pdo"], ["MEIv2","PDO"], ["#d7191c","#2c7bb6"]):
    ax.plot(tele.date, tele[col], color=c, lw=1.0)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel(lbl)
    for yr in range(2000, 2025):
        ax.axvspan(pd.Timestamp(yr, 5, 1), pd.Timestamp(yr, 9, 30), color="orange", alpha=0.06)
axes[-1].set_xlabel("Date")
plt.suptitle("ENSO (MEIv2) and PDO 2000-2024 (BC fire seasons May-Sep shaded)", y=1.02)
plt.tight_layout()
fig_out = FIGS / "S_enso_pdo_timeseries.png"
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Wrote {fig_out}")
fires = gpd.read_file(DATA_PROC / "fires_geq_70ha.shp")
year_col = next((c for c in ("YEAR","YEAR_","FIRE_YEAR") if c in fires.columns), None)
size_col = next((c for c in ("SIZE_HA","AREA_HA") if c in fires.columns), None)
print(f"Using fire-year column: {year_col}, size column: {size_col}")
annual = fires.groupby(year_col)[size_col].sum().reset_index().rename(columns={year_col:"year", size_col:"burned_ha"})
annual["year"] = annual["year"].astype(int)
print(f"Annual fire records: {len(annual)} years, {annual.year.min()}-{annual.year.max()}")
fs = tele[tele.month.isin([5,6,7,8,9])].groupby("year", as_index=False).agg(
    mei_fs_mean=("mei","mean"), pdo_fs_mean=("pdo","mean"))
fs["year"] = fs["year"].astype(int)
joined = annual.merge(fs, on="year", how="inner")
print(f"After merge: {len(joined)} years")
joined_mei = joined.dropna(subset=["burned_ha","mei_fs_mean"])
joined_pdo = joined.dropna(subset=["burned_ha","pdo_fs_mean"])
corr_rows = []
if len(joined_mei) >= 2:
    r, p = pearsonr(joined_mei["mei_fs_mean"], joined_mei["burned_ha"])
    rho, ps = spearmanr(joined_mei["mei_fs_mean"], joined_mei["burned_ha"])
    corr_rows.append({"index":"mei_fs_mean","pearson_r":round(r,4),"pearson_p":round(p,4),"spearman_rho":round(rho,4),"spearman_p":round(ps,4),"n":len(joined_mei)})
if len(joined_pdo) >= 2:
    r, p = pearsonr(joined_pdo["pdo_fs_mean"], joined_pdo["burned_ha"])
    rho, ps = spearmanr(joined_pdo["pdo_fs_mean"], joined_pdo["burned_ha"])
    corr_rows.append({"index":"pdo_fs_mean","pearson_r":round(r,4),"pearson_p":round(p,4),"spearman_rho":round(rho,4),"spearman_p":round(ps,4),"n":len(joined_pdo)})
corr = pd.DataFrame(corr_rows)
corr.to_csv(TABLES / "T_teleconnection_burnedarea_corr.csv", index=False)
print("\nCorrelation results (annual BC burned area vs mean fire-season teleconnection):")
print(corr.to_string(index=False) if len(corr) else "  No correlations - insufficient data overlap")
print("\nDone.")
