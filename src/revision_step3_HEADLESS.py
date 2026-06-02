# -*- coding: utf-8 -*-
"""
revision_step3_HEADLESS.py
==========================
Headless version of revision_step3_run_all_models.py.

  - NO Tkinter dialogs (reads all paths from .revision_step3_paths.json)
  - NO confirmation popup (auto-yes)
  - All output goes to stdout/stderr -- redirect to a log file for monitoring
  - Same resumable behaviour: skips combos with existing metrics_summary.json
  - Same training behaviour: 6 PSO particles x 6 iters x 30 epochs

Use this from Task Scheduler so it runs invisibly to other lab users.

Run (interactive test):
    set TF_FORCE_GPU_ALLOW_GROWTH=true
    python src\revision_step3_HEADLESS.py

Run (real -- via Task Scheduler or detached cmd, write log to OneDrive):
    cmd /C "set TF_FORCE_GPU_ALLOW_GROWTH=true && python src\revision_step3_HEADLESS.py > "%OneDrive%\wildfire_training.log" 2>&1"
"""
import os
import sys
import json
import time
import subprocess

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Design
SEEDS_AT_THR70    = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
THRESHOLDS_AT_S42 = [100, 200]
THR_MAIN          = 70
SEED_MAIN         = 42

# PSO budget (production)
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

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".revision_step3_paths.json")

def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{stamp()}] {msg}", flush=True)

log("=" * 70)
log("STEP 3 PRODUCTION (HEADLESS)")
log("=" * 70)

# Load cached paths
if not os.path.exists(CONFIG_PATH):
    log(f"FATAL: cache file not found: {CONFIG_PATH}")
    log("Run revision_step3_PREFLIGHT.py once interactively first")
    log("to populate the cache, then re-run this headless script.")
    sys.exit(2)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cache = json.load(f)

required = ["tables", "bilstm_pso", "lstm_pso", "nonpso_cli", "python_exe"]
missing = [k for k in required if not cache.get(k)]
if missing:
    log(f"FATAL: cache missing keys: {missing}")
    sys.exit(2)

tables      = cache["tables"]
bilstm_pso  = cache["bilstm_pso"]
lstm_pso    = cache["lstm_pso"]
nonpso_cli  = cache["nonpso_cli"]
python_exe  = cache["python_exe"]

# Out_root: prefer "out_root" if present, else derive from "parent_dir"
out_root = cache.get("out_root")
if not out_root:
    parent = cache.get("parent_dir")
    if not parent:
        log("FATAL: cache has no 'out_root' or 'parent_dir'")
        sys.exit(2)
    out_root = os.path.join(parent, "05_Model_Results")
os.makedirs(out_root, exist_ok=True)

log(f"tables     = {tables}")
log(f"bilstm_pso = {bilstm_pso}")
log(f"lstm_pso   = {lstm_pso}")
log(f"nonpso_cli = {nonpso_cli}")
log(f"out_root   = {out_root}")
log(f"python     = {python_exe}")

# Validate each path
for k, v in [("tables", tables), ("bilstm_pso", bilstm_pso),
             ("lstm_pso", lstm_pso), ("nonpso_cli", nonpso_cli),
             ("python_exe", python_exe)]:
    if not os.path.exists(v):
        log(f"FATAL: path does not exist [{k}]: {v}")
        sys.exit(2)

# Build the 12-combo plan
combos = [(THR_MAIN, s) for s in SEEDS_AT_THR70]
for thr in THRESHOLDS_AT_S42:
    combos.append((thr, SEED_MAIN))

plan = []
for thr, s in combos:
    csv = os.path.join(tables, f"training_thr{thr}_seed{s}.csv")
    if not os.path.exists(csv):
        log(f"FATAL: missing CSV: {csv}")
        sys.exit(2)
    plan.append((thr, s, csv))

def needs(name, thr, s):
    d = os.path.join(out_root, name, f"thr{thr}", f"seed{s}")
    return FORCE_RERUN or not os.path.exists(os.path.join(d, "metrics_summary.json"))

# Build task list (only what's not done)
task_list = []
for ci, (thr, s, csv) in enumerate(plan):
    if needs("BiLSTM_PSO", thr, s):
        task_list.append(("pso", thr, s, csv, "BiLSTM_PSO", bilstm_pso))
    if needs("LSTM_PSO", thr, s):
        task_list.append(("pso", thr, s, csv, "LSTM_PSO", lstm_pso))
    if needs("LSTM", thr, s) or needs("BiLSTM", thr, s):
        which = []
        if needs("LSTM", thr, s):   which.append("lstm")
        if needs("BiLSTM", thr, s): which.append("bilstm")
        task_list.append(("nonpso", thr, s, csv,
                           "+".join(w.upper() for w in which), nonpso_cli, which))

n_pso    = sum(1 for t in task_list if t[0] == "pso")
n_nonpso = sum(1 for t in task_list if t[0] == "nonpso")

log(f"Plan: 4 models x {len(plan)} combos = {4*len(plan)} runs total")
log(f"  PSO remaining:     {n_pso}")
log(f"  non-PSO remaining: {n_nonpso}")
log(f"  Already complete:  {len(plan)*4 - n_pso - 2*n_nonpso}")

def run_subprocess(cmd, log_path):
    """Run cmd, stream every stdout line to BOTH our log AND the per-run log."""
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as runlog:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            runlog.write(line); runlog.flush()
            print("    " + line.rstrip(), flush=True)
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
        od = os.path.join(out_root, name, f"thr{thr}", f"seed{s}")
        os.makedirs(od, exist_ok=True)
        cmd = [python_exe, script, "--data", csv, "--out_dir", od,
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
            log(f"  >>> {name} thr{thr} s{s}: FAIL ({dt/60:.1f} min)  log: {log_path}")
    else:
        kind, thr, s, csv, name, script, which = entry
        log_dir = os.path.join(out_root, "__nonpso_logs__", f"thr{thr}_seed{s}")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "run_log.txt")
        cmd = [python_exe, script,
               "--data", csv,
               "--base_out_dir", out_root,
               "--thr", str(thr), "--seed", str(s),
               "--epochs", str(NONPSO_EPOCHS),
               "--grid_km", str(GRID_KM),
               "--models", ",".join(which)]
        rc, dt = run_subprocess(cmd, log_path)
        per_ok = 0
        for sub_lower in which:
            sub = "LSTM" if sub_lower == "lstm" else "BiLSTM"
            msj = os.path.join(out_root, sub, f"thr{thr}", f"seed{s}", "metrics_summary.json")
            if rc == 0 and os.path.exists(msj):
                per_ok += 1
        if rc == 0 and per_ok == len(which):
            ok += len(which)
            log(f"  >>> {name} thr{thr} s{s}: OK ({dt/60:.1f} min)")
        else:
            fail += len(which); fail_logs.append(log_path)
            log(f"  >>> {name} thr{thr} s{s}: FAIL ({dt/60:.1f} min)  log: {log_path}")

log("=" * 70)
log(f"STEP 3 FINISHED in {(time.time()-t_total)/3600:.2f} hours")
log(f"  OK:   {ok}")
log(f"  FAIL: {fail}")
log("=" * 70)
if fail_logs:
    log("Failed run logs:")
    for lg in fail_logs[:10]:
        log(f"  {lg}")
log("Next: STEP 4 (summarize results)")
