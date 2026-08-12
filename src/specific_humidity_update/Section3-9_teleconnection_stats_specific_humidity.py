"""
Section 3.9 teleconnection analysis, recomputed with SPECIFIC HUMIDITY.

Fire-season (May-August) annual means of each climate predictor are correlated
(Spearman) with fire-season ONI and PDO over 2000-2024 (n = 25 years), and the
ONI series is also detrended to separate interannual covariation from shared
trends.

Key result vs the old relative-humidity analysis:
  - Relative humidity (old):  ONI rho = -0.61 (significant negative)
  - Specific humidity (new):  ONI rho = +0.21 (p=0.32, n.s.),
                              PDO rho = +0.05 (p=0.80, n.s.)
  => specific humidity is NOT significantly coupled to ENSO/PDO (warm air simply
     holds more moisture), so it moves out of the "lower in warm phases" group.
     Precipitation (ONI rho=-0.47) and drought/PDSI (ONI rho=-0.51) carry the
     warm-phase drying signal reported in Fig. 18a-b.

Inputs (repo-relative):
  - wide climate CSV (year, month, ... , precipitation)
  - ERA5-Land specific humidity CSV (year, month, specific_humidity_mean)
  - data/raw/pdo.csv  (ERSST PDO index: 'Year' + Jan..Dec columns; 1 title line)
  - TerraClimate PDSI CSV (for the drought correlation)
ONI is hard-coded below (fire-season AMJ/MJJ/JJA/JAS averages, NOAA CPC).

Run:  python src/specific_humidity_update/teleconnection_specific_humidity.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WIDE_CLIMATE_CSV = os.path.join(REPO, "data", "BC_2000_2024_monthly_climate_wide.csv")
SPECIFIC_HUMIDITY_CSV = os.path.join(REPO, "data", "BC_ERA5Land_monthly_specific_humidity_2000_2024.csv")
PDO_CSV = os.path.join(REPO, "data", "raw", "pdo.csv")
PDSI_CSV = os.path.join(REPO, "reviewer_response_C2_C15", "data", "raw",
                        "BC_TerraClimate_monthly_PDSI_2000_2024.csv")

# Fire-season ONI (mean of AMJ, MJJ, JJA, JAS anomalies), NOAA CPC, 2000-2024.
ONI = {2000: -0.6025, 2001: -0.145, 2002: 0.6825, 2003: -0.0325, 2004: 0.39,
       2005: 0.05, 2006: 0.0575, 2007: -0.555, 2008: -0.5125, 2009: 0.33,
       2010: -0.8075, 2011: -0.46, 2012: 0.15, 2013: -0.31, 2014: 0.1875,
       2015: 1.4375, 2016: -0.08, 2017: 0.2275, 2018: 0.085, 2019: 0.42,
       2020: -0.2775, 2021: -0.3725, 2022: -0.8275, 2023: 0.975, 2024: 0.18}
FS_MONTHS = [5, 6, 7, 8]


def fire_season_mean(df, col):
    return df[df.month.isin(FS_MONTHS)].groupby("year")[col].mean()


def main():
    years = sorted(ONI)
    onv = np.array([ONI[y] for y in years])

    # PDO: skip the 1-line title, whitespace-delimited, wide Year x Month
    pdo = pd.read_csv(PDO_CSV, skiprows=1, sep=r"\s+")
    mon = {m: i + 1 for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
    pl = pdo.melt(id_vars="Year", value_vars=list(mon), var_name="m", value_name="pdo")
    pl["month"] = pl.m.map(mon)
    pdo_fs = pl[pl.month.isin(FS_MONTHS)].groupby("Year")["pdo"].mean()
    pdv = np.array([pdo_fs[y] for y in years])

    clim = pd.read_csv(WIDE_CLIMATE_CSV)
    sh = pd.read_csv(SPECIFIC_HUMIDITY_CSV).rename(columns={"specific_humidity_mean": "specific_humidity"})
    p = pd.read_csv(PDSI_CSV); p["date"] = pd.to_datetime(p["date"])
    p["year"] = p.date.dt.year; p["month"] = p.date.dt.month
    pdsi = p.rename(columns={"mean": "PDSI"})

    series = {
        "Specific humidity": fire_season_mean(sh, "specific_humidity"),
        "Relative humidity (old)": fire_season_mean(clim, "avg_relative_humidity"),
        "Precipitation": fire_season_mean(clim, "precipitation"),
        "Drought (PDSI)": fire_season_mean(pdsi, "PDSI"),
        "Max temperature": fire_season_mean(clim, "max_temperature"),
    }

    def detrend(v):
        t = np.arange(len(v))
        return v - np.polyval(np.polyfit(t, v, 1), t)

    print(f"{'Variable':24s} {'ONI rho (p)':>18s} {'PDO rho (p)':>18s} {'ONI detr rho (p)':>20s}")
    for name, s in series.items():
        v = np.array([s[y] for y in years])
        ro, po = sps.spearmanr(onv, v)
        rp, pp = sps.spearmanr(pdv, v)
        rd, pd_ = sps.spearmanr(detrend(onv), detrend(v))
        print(f"{name:24s} {f'{ro:+.2f} ({po:.2f})':>18s} "
              f"{f'{rp:+.2f} ({pp:.2f})':>18s} {f'{rd:+.2f} ({pd_:.2f})':>20s}")


if __name__ == "__main__":
    main()
