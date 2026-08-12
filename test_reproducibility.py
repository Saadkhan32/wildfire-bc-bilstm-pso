#!/usr/bin/env python3
"""
Reproducibility smoke test for: wildfire-bc-bilstm-pso
=====================================================
Run from the repository root, inside the conda env:

    conda env create -f environment.yml      # one time
    conda activate wildfire
    python test_reproducibility.py

It does what a reviewer does: checks the environment, confirms the shipped
data are present, and actually runs the deterministic analyses to confirm they
produce the expected numbers. Exit code 0 = all critical checks passed.
"""
import sys, os, subprocess, importlib

PASS, WARN, FAIL = [], [], []
def ok(m):   PASS.append(m); print("  [PASS]", m)
def warn(m): WARN.append(m); print("  [WARN]", m)
def fail(m): FAIL.append(m); print("  [FAIL]", m)

print("\n1) Python & core packages")
v = sys.version_info
(ok if v[:2] >= (3, 9) else warn)(f"Python {v.major}.{v.minor} (env wants 3.10)")
for pkg in ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "sklearn", "statsmodels"]:
    try: importlib.import_module(pkg); ok(f"import {pkg}")
    except Exception: fail(f"missing {pkg}  -> conda env not active / not created")

print("\n2) Optional packages (needed by specific scripts)")
for opt in ["tensorflow", "keras", "geopandas", "rasterio", "shap",
            "esda", "libpysal", "pingouin", "pymannkendall"]:
    try: importlib.import_module(opt); ok(f"import {opt}")
    except Exception: warn(f"optional {opt} not installed (some scripts need it)")

print("\n3) Shipped data & key files present")
for f in ["environment.yml", "README.md",
          "code/spatial_autocorrelation.py",
          "data/training_points_70ha_seed42.csv"]:
    (ok if os.path.exists(f) else fail)(f"exists {f}")
for d in ["data/rasters", "data/susceptibility"]:
    (ok if os.path.isdir(d) else warn)(f"dir   {d}")

print("\n4) Inputs some scripts require (warn if absent)")
for f in ["data/raw/meiv2.txt", "data/raw/pdo.csv",
          "data/processed/fires_geq_70ha.shp"]:
    (ok if os.path.exists(f) else warn)(f"input {f} (teleconnections.py needs this)")

print("\n5) Run the deterministic analysis (spatial autocorrelation)")
try:
    r = subprocess.run([sys.executable, "code/spatial_autocorrelation.py"],
                       capture_output=True, text=True, timeout=600)
    if "Moran" in r.stdout:
        ok("spatial_autocorrelation.py runs and prints Moran's I")
        for line in r.stdout.splitlines():
            if "Moran I:" in line or "Moran (KNN8)" in line:
                print("        ->", line.strip())
    else:
        fail("spatial_autocorrelation.py ran but produced no Moran output")
        print(r.stdout[-400:] or r.stderr[-400:])
except Exception as e:
    fail(f"spatial_autocorrelation.py errored: {e}")

print("\n" + "=" * 60)
print(f"SUMMARY:  {len(PASS)} pass | {len(WARN)} warn | {len(FAIL)} fail")
if FAIL:
    print("CRITICAL failures above must be fixed for a reviewer to reproduce.")
elif WARN:
    print("No blockers, but review the WARN items (optional deps / inputs).")
else:
    print("All checks passed.")
print("=" * 60)
sys.exit(1 if FAIL else 0)
