#!/usr/bin/env python3
"""
Reproducibility smoke test for: wildfire-bc-bilstm-pso
=====================================================
Run from the package root (assembled Zenodo download) or the repository root:

    conda env create -f environment.yml      # one time
    conda activate wildfire
    python test_reproducibility.py

It does what a reviewer does: checks the environment, confirms the shipped
files are present, and re-runs a deterministic analysis to confirm it
reproduces the numbers reported in the manuscript.
Exit code 0 = all critical checks passed. Data-dependent checks are skipped
with a warning when data.zip has not been unpacked (e.g. a git-only clone).
"""
import sys, os, importlib

PASS, WARN, FAIL = [], [], []
def ok(m):   PASS.append(m); print("  [PASS]", m)
def warn(m): WARN.append(m); print("  [WARN]", m)
def fail(m): FAIL.append(m); print("  [FAIL]", m)

print("\n1) Python & core packages")
v = sys.version_info
(ok if v[:2] >= (3, 9) else warn)(f"Python {v.major}.{v.minor} (env pins 3.10)")
for pkg in ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "sklearn", "statsmodels"]:
    try: importlib.import_module(pkg); ok(f"import {pkg}")
    except Exception: fail(f"missing {pkg}  -> conda env not active / not created")

print("\n2) Optional packages (needed by specific scripts)")
for opt in ["tensorflow", "keras", "geopandas", "rasterio", "shap",
            "esda", "libpysal", "pingouin", "pymannkendall"]:
    try: importlib.import_module(opt); ok(f"import {opt}")
    except Exception: warn(f"optional {opt} not installed (some scripts need it)")

print("\n3) Key files present")
for f in ["environment.yml", "README.md", "src/run_climate_full_stats.py",
          "src/fig_wildfire_trend.py", "src/seeds.py"]:
    (ok if os.path.exists(f) else fail)(f"exists {f}")
for f in ["data/analysis_dataset.csv", "data/training_points_70ha_seed42.csv",
          "data/shap/SHAP_BiLSTM_PSO_values.pkl",
          "src/build_shap_beeswarm.py", "src/cross_border_c4.py",
          "src/specific_humidity_update/Figure17_SHAP_beeswarm_specific_humidity.py"]:
    (ok if os.path.exists(f) else warn)(f"repro asset {f} (unpack data.zip / code.zip)")
for d in ["data/rasters", "data/susceptibility"]:
    (ok if os.path.isdir(d) else warn)(f"dir   {d} (unpack data.zip)")

print("\n4) Inputs some scripts require (warn if absent)")
for f in ["data/raw/meiv2.txt", "data/raw/pdo.csv",
          "data/processed/fires_geq_70ha.shp"]:
    (ok if os.path.exists(f) else warn)(f"input {f}")

print("\n5) Deterministic reproduction: seasonal climate trends (Theil-Sen)")
CLIM = "data/climate/merged_monthly_climate_2000_2024.csv"
if not os.path.exists(CLIM):
    warn(f"{CLIM} not found - unpack data.zip to run the numeric check")
else:
    try:
        import pandas as pd
        from scipy import stats
        c = pd.read_csv(CLIM)
        sea = c[c["month"].between(5, 8)].groupby("year").mean(numeric_only=True)
        checks = [("avg_temperature", 0.0702, 0.005, "degC/yr"),
                  ("specific_humidity", 2.88e-05, 5e-06, "kg/kg/yr")]
        for col, expect, tol, unit in checks:
            slope = stats.theilslopes(sea[col].values, sea.index.values, 0.95)[0]
            msg = f"May-Aug {col}: Theil-Sen {slope:+.4g} {unit} (manuscript {expect:+.4g})"
            (ok if abs(slope - expect) <= tol else fail)(msg)
    except Exception as e:
        fail(f"numeric check errored: {e}")

print("\n" + "=" * 60)
print(f"SUMMARY:  {len(PASS)} pass | {len(WARN)} warn | {len(FAIL)} fail")
if FAIL:
    print("CRITICAL failures above must be fixed for a reviewer to reproduce.")
elif WARN:
    print("No blockers, but review the WARN items (optional deps / data.zip).")
else:
    print("All checks passed.")
print("=" * 60)
sys.exit(1 if FAIL else 0)
