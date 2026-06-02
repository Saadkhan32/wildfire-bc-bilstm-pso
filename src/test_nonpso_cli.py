# -*- coding: utf-8 -*-
"""
test_nonpso_cli.py
==================
Focused test of c8c11_non_pso_cli.py.

This script AUTO-LOCATES the wildfire conda env on disk and uses ITS
python.exe to run the subprocess.  That way it does NOT matter which
interpreter VS Code is configured with -- press Ctrl+F5 in VS Code with
ANY interpreter, and this script will still launch the subprocess inside
the wildfire env where joblib/tensorflow/etc. are installed.

If the wildfire env is in a non-standard location, edit WILDFIRE_PY at
the top of this file.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# ============================================================
# AUTO-LOCATE wildfire env's python.exe (edit if non-standard)
# ============================================================
CANDIDATE_WILDFIRE_PATHS = [
    r"C:\Users\saadz\anaconda3\envs\wildfire\python.exe",
    r"C:\Users\saadz\miniconda3\envs\wildfire\python.exe",
    r"C:\ProgramData\anaconda3\envs\wildfire\python.exe",
    r"C:\ProgramData\miniconda3\envs\wildfire\python.exe",
    # Add more if conda is installed somewhere else:
    r"C:\anaconda3\envs\wildfire\python.exe",
    r"C:\miniconda3\envs\wildfire\python.exe",
]

WILDFIRE_PY = None
for candidate in CANDIDATE_WILDFIRE_PATHS:
    if Path(candidate).exists():
        WILDFIRE_PY = candidate
        break

if WILDFIRE_PY is None:
    print("=" * 70, flush=True)
    print("ERROR: could not auto-locate the wildfire conda env.", flush=True)
    print("=" * 70, flush=True)
    print("\nTried these paths (none exist):", flush=True)
    for c in CANDIDATE_WILDFIRE_PATHS:
        print(f"  - {c}", flush=True)
    print("\nTo find the correct path, open Anaconda Prompt and run:", flush=True)
    print("    conda activate wildfire", flush=True)
    print("    where python", flush=True)
    print("\nThen edit the CANDIDATE_WILDFIRE_PATHS list at the top of", flush=True)
    print("this file and add that path as the first entry.", flush=True)
    sys.exit(2)

# ============================================================
# CONFIG
# ============================================================
REPO        = Path(__file__).resolve().parent.parent
CLI_SCRIPT  = REPO / "src" / "c8c11_non_pso_cli.py"
CSV_FILE    = REPO / "revision_c8c11" / "03_Training_Tables" / "training_thr70_seed42.csv"
OUT_DIR     = REPO / "revision_c8c11" / "05_Model_Results_PREFLIGHT"
EPOCHS      = 3
GRID_KM     = 50
MODELS      = "lstm,bilstm"
THR         = 70
SEED        = 42

print("=" * 70, flush=True)
print("test_nonpso_cli.py  --  non-PSO CLI focused test", flush=True)
print("=" * 70, flush=True)
print(f"  wildfire env python : {WILDFIRE_PY}",                                flush=True)
print(f"  (script was launched by : {sys.executable})",                        flush=True)
print(f"  CLI script           : {CLI_SCRIPT}",                                flush=True)
print(f"  data                 : {CSV_FILE}",                                  flush=True)
print(f"  out dir              : {OUT_DIR}",                                   flush=True)
print(f"  epochs               : {EPOCHS}  (tiny test)",                       flush=True)
print(f"  models               : {MODELS}",                                    flush=True)
print("=" * 70, flush=True)

missing = []
if not CLI_SCRIPT.exists(): missing.append(f"CLI script: {CLI_SCRIPT}")
if not CSV_FILE.exists():   missing.append(f"CSV file:   {CSV_FILE}")
if missing:
    print("\nERROR: missing required file(s):", flush=True)
    for m in missing: print(f"  - {m}", flush=True)
    sys.exit(2)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Quick sanity: does the wildfire env actually have joblib + tensorflow?
print("\nVerifying wildfire env has required packages ...", flush=True)
probe = subprocess.run(
    [WILDFIRE_PY, "-c",
     "import joblib, tensorflow, sklearn, pandas, numpy; "
     "print('joblib', joblib.__version__); "
     "print('tensorflow', tensorflow.__version__); "
     "print('sklearn', sklearn.__version__)"],
    capture_output=True, text=True
)
if probe.returncode != 0:
    print("ERROR: wildfire env is missing packages:", flush=True)
    print(probe.stdout, flush=True)
    print(probe.stderr, flush=True)
    print("\nFix by running this in Anaconda Prompt:", flush=True)
    print("  conda activate wildfire", flush=True)
    print("  pip install joblib tensorflow scikit-learn pandas numpy", flush=True)
    sys.exit(3)
print("  OK:", flush=True)
for line in probe.stdout.strip().splitlines():
    print(f"    {line}", flush=True)

# ============================================================
# Run subprocess with streaming
# ============================================================
cmd = [
    WILDFIRE_PY, str(CLI_SCRIPT),
    "--data",         str(CSV_FILE),
    "--base_out_dir", str(OUT_DIR),
    "--thr",          str(THR),
    "--seed",         str(SEED),
    "--epochs",       str(EPOCHS),
    "--grid_km",      str(GRID_KM),
    "--models",       MODELS,
]
print("\nCommand:", flush=True)
print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
print("\nStreaming subprocess output:\n", flush=True)

t0 = time.time()
proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, encoding="utf-8", errors="replace",
)
for line in proc.stdout:
    print("  " + line.rstrip(), flush=True)
proc.wait()
dt = (time.time() - t0) / 60

# ============================================================
# Verdict
# ============================================================
print("\n" + "=" * 70, flush=True)
print("VERDICT", flush=True)
print("=" * 70, flush=True)
print(f"  subprocess return code : {proc.returncode}", flush=True)
print(f"  wall time              : {dt:.1f} min",       flush=True)

lstm_json   = OUT_DIR / "LSTM"   / f"thr{THR}" / f"seed{SEED}" / "metrics_summary.json"
bilstm_json = OUT_DIR / "BiLSTM" / f"thr{THR}" / f"seed{SEED}" / "metrics_summary.json"

print(f"  LSTM   metrics.json     : {'YES' if lstm_json.exists()   else 'NO  '}  ({lstm_json})",   flush=True)
print(f"  BiLSTM metrics.json     : {'YES' if bilstm_json.exists() else 'NO  '}  ({bilstm_json})", flush=True)

all_ok = proc.returncode == 0 and lstm_json.exists() and bilstm_json.exists()
print("\n" + "=" * 70, flush=True)
if all_ok:
    print("PASS.  Non-PSO CLI works.  You can launch production STEP 3 now.", flush=True)
else:
    print("FAIL.  See streamed output above for the error.", flush=True)
print("=" * 70, flush=True)
