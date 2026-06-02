# -*- coding: utf-8 -*-
"""
revision_step3_HEADLESS.py - split-workload production training.

Usage:
    python src/revision_step3_HEADLESS.py --mode pso       # runs BiLSTM_PSO + LSTM_PSO only
    python src/revision_step3_HEADLESS.py --mode nonpso    # runs non-PSO LSTM + BiLSTM only
    python src/revision_step3_HEADLESS.py --mode all       # runs everything (default)

Paths auto-detected from this script's location (no cache file needed).
Resumable: skips combos with existing metrics_summary.json.
"""
import os
import sys
import json
import time
import argparse
import subprocess

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

# ----- CLI args -----
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["all", "pso", "nonpso"], default="all",
                help="Which models to train: 'pso' (BiLSTM_PSO+LSTM_PSO), 'nonpso' (LSTM+BiLSTM), or 'all'")
ap.add_argument("--python-exe", default=sys.executable,
                help="Python interpreter to use for subprocesses (default: current python)")
args = ap.parse_args()

MODE = args.mode
PYTHON_EXE = args.python_exe

# ----- Auto-detect paths -----
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
TABLES      = os.path.join(REPO_ROOT, "revision_c8c11", "03_Training_Tables")
MODEL_DIR   = os.path.join(REPO_ROOT, "revision_c8c11", "04_Model_Scripts")
BILSTM_PSO  = os.path.join(MODEL_DIR, "BiLSTM PSO FE.py")
LSTM_PSO    = os.path.join(MODEL_DIR, "LSTM PSO FE.py")
NONPSO_CLI  = os.path.join(SCRIPT_DIR, "c8c11_non_pso_cli.py")
OUT_ROOT    = os.path.join(REPO_ROOT, "revision_c8c11", "05_Model_Results")

# ----- Design -----
SEEDS_AT_THR70    = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
THRESHOLDS_AT_S42 = [100, 200]
THR_MAIN          = 70
SEED_MAIN         = 42

# ----- PSO production budget -----
OBJECTIVE      = "roc"
GRID_KM        = 50
PROGRESS       = "none"
PSO_PARTICLES  = 6
PSO_ITERS      = 6
PSO_FOLDS      = 2
SEARCH_EPOCHS  = 30
RETRAIN_EPOCHS = 30
NONPSO_EPOCHS  = 30
FORCE_RERUN    = False

def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{stamp()}] {msg}", flush=True)

log("=" * 70)
log(f"STEP 3 PRODUCTION (HEADLESS) -- MODE={MODE}")
log("=" * 70)
log(f"repo       = {REPO_ROOT}")
log(f"tables     = {TABLES}")
log(f"bilstm_pso = {BILSTM_PSO}")
log(f"lstm_pso   = {LSTM_PSO}")
log(f"nonpso_cli = {NONPSO_CLI}")
log(f"out_root   = {OUT_ROOT}")
log(f"python     = {PYTHON_EXE}")

# Validate
required_paths = {"tables": TABLES, "python_exe": PYTHON_EXE}
if MODE in ("all", "pso"):
    required_paths["bilstm_pso"] = BILSTM_PSO
    required_paths["lstm_pso"] = LSTM_PSO
if MODE in ("all", "nonpso"):
    required_paths["nonpso_cli"] = NONPSO_CLI

for k, v in required_paths.items():
    if not os.path.exists(v):
        log(f"FATAL: missing [{k}]: {v}")
        sys.exit(2)

os.makedirs(OUT_ROOT, exist_ok=True)

# Build combo plan
combos = [(THR_MAIN, s) for s in SEEDS_AT_THR70]
for thr in THRESHOLDS_AT_S42:
    combos.append((thr, SEED_MAIN))

plan = []
for thr, s in combos:
    csv = os.path.join(TABLES, f"training_thr{thr}_seed{s}.csv")
    if not os.path.exists(csv):
        log(f"FATAL: missing CSV: {csv}")
        sys.exit(2)
    plan.append((thr, s, csv))

def needs(name, thr, s):
    d = os.path.join(OUT_ROOT, name, f"thr{thr}", f"seed{s}")
    return FORCE_RERUN or not os.path.exists(os.path.join(d, "metrics_summary.json"))

# Build task list filtered by MODE
task_list = []
for ci, (thr, s, csv) in enumerate(plan):
    if MODE in ("all", "pso"):
        if needs("BiLSTM_PSO", thr, s):
            task_list.append(("pso", thr, s, csv, "BiLSTM_PSO", BILSTM_PSO))
        if needs("LSTM_PSO", thr, s):
            task_list.append(("pso", thr, s, csv, "LSTM_PSO", LSTM_PSO))
    if MODE in ("all", "nonpso"):
        if needs("LSTM", thr, s) or needs("BiLSTM", thr, s):
            which = []
            if needs("LSTM", thr, s):   which.append("lstm")
            if needs("BiLSTM", thr, s): which.append("bilstm")
            task_list.append(("nonpso", thr, s, csv,
                               "+".join(w.upper() for w in which), NONPSO_CLI, which))

n_pso    = sum(1 for t in task_list if t[0] == "pso")
n_nonpso = sum(1 for t in task_list if t[0] == "nonpso")
log(f"Plan (mode={MODE}): {len(task_list)} subprocess calls")
log(f"  PSO subprocess calls:     {n_pso}")
log(f"  non-PSO subprocess calls: {n_nonpso}")

def safe_print(prefix, line):
    try:
        print(prefix + line.rstrip(), flush=True)
    except (UnicodeEncodeError, UnicodeDecodeError):
        safe = line.rstrip().encode("ascii", "replace").decode("ascii")
        print(prefix + safe, flush=True)

def run_subprocess(cmd, log_path):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as runlog:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            runlog.write(line); runlog.flush()
            safe_print("    ", line)
        proc.wait()
    return proc.returncode, time.time() - t0

t_total = time.time()
ok = 0
fail = 0
fail_logs = []

for i, entry in enumerate(task_list, 1):
    kind = entry[0]
    log("-" * 70)
    log(f"[{i:>3}/{len(task_list)}]  {entry[4]}  thr{entry[1]}  seed{entry[2]}")
    log("-" * 70)
    if kind == "pso":
        kind, thr, s, csv, name, script = entry
        od = os.path.join(OUT_ROOT, name, f"thr{thr}", f"seed{s}")
        os.makedirs(od, exist_ok=True)
        cmd = [PYTHON_EXE, script, "--data", csv, "--out_dir", od,
               "--objective", OBJECTIVE, "--grid_km", str(GRID_KM),
               "--pso_particles", str(PSO_PARTICLES),
               "--pso_iters", str(PSO_ITERS), "--pso_folds", str(PSO_FOLDS),
               "--search_epochs", str(SEARCH_EPOCHS),
               "--retrain_epochs", str(RETRAIN_EPOCHS),
               "--progress", PROGRESS]
        log_path = os.path.join(od, "run_log.txt")
        rc, dt = run_subprocess(cmd, log_path)
        if rc == 0 and os.path.exists(os.path.join(od, "metrics_summary.json")):
            ok += 1
            log(f"  >>> {name} thr{thr} s{s}: OK ({dt/60:.1f} min)")
        else:
            fail += 1; fail_logs.append(log_path)
            log(f"  >>> {name} thr{thr} s{s}: FAIL ({dt/60:.1f} min)")
    else:
        kind, thr, s, csv, name, script, which = entry
        log_dir = os.path.join(OUT_ROOT, "__nonpso_logs__", f"thr{thr}_seed{s}")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "run_log.txt")
        cmd = [PYTHON_EXE, script,
               "--data", csv,
               "--base_out_dir", OUT_ROOT,
               "--thr", str(thr), "--seed", str(s),
               "--epochs", str(NONPSO_EPOCHS),
               "--grid_km", str(GRID_KM),
               "--models", ",".join(which)]
        rc, dt = run_subprocess(cmd, log_path)
        per_ok = 0
        for sub_lower in which:
            sub = "LSTM" if sub_lower == "lstm" else "BiLSTM"
            msj = os.path.join(OUT_ROOT, sub, f"thr{thr}", f"seed{s}", "metrics_summary.json")
            if rc == 0 and os.path.exists(msj):
                per_ok += 1
        if rc == 0 and per_ok == len(which):
            ok += len(which)
            log(f"  >>> {name} thr{thr} s{s}: OK ({dt/60:.1f} min)")
        else:
            fail += len(which); fail_logs.append(log_path)
            log(f"  >>> {name} thr{thr} s{s}: FAIL ({dt/60:.1f} min)")

log("=" * 70)
log(f"STEP 3 FINISHED (mode={MODE}) in {(time.time()-t_total)/3600:.2f} hours")
log(f"  OK:   {ok}")
log(f"  FAIL: {fail}")
log("=" * 70)
