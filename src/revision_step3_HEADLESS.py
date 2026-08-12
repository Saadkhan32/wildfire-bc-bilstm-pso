import os
import sys
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
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["all", "pso", "nonpso"], default="all",
                help="Which models to train")
ap.add_argument("--python-exe", default=sys.executable)
ap.add_argument("--verbose", action="store_true",
                help="Stream ALL subprocess output (default: only key lines)")
args = ap.parse_args()
MODE       = args.mode
PYTHON_EXE = args.python_exe
VERBOSE    = args.verbose
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found; installing...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm",
                            "--quiet", "--disable-pip-version-check"])
    from tqdm import tqdm
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
TABLES      = os.path.join(REPO_ROOT, "revision_c8c11", "03_Training_Tables")
MODEL_DIR   = os.path.join(REPO_ROOT, "revision_c8c11", "04_Model_Scripts")
BILSTM_PSO  = os.path.join(MODEL_DIR, "BiLSTM PSO FE.py")
LSTM_PSO    = os.path.join(MODEL_DIR, "LSTM PSO FE.py")
NONPSO_CLI  = os.path.join(SCRIPT_DIR, "c8c11_non_pso_cli.py")
OUT_ROOT    = os.path.join(REPO_ROOT, "revision_c8c11", "05_Model_Results")
SEEDS_AT_THR70    = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
THRESHOLDS_AT_S42 = [100, 200]
THR_MAIN, SEED_MAIN = 70, 42
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
def stamp(): return time.strftime("%H:%M:%S")
def log(msg): print(f"[{stamp()}] {msg}", flush=True)
print("=" * 70, flush=True)
print(f"STEP 3 PRODUCTION  --  MODE={MODE.upper()}", flush=True)
print("=" * 70, flush=True)
print(f"repo       : {REPO_ROOT}", flush=True)
print(f"out_root   : {OUT_ROOT}", flush=True)
print(f"python     : {PYTHON_EXE}", flush=True)
print(f"verbose    : {VERBOSE}", flush=True)
print(flush=True)
required = {"tables": TABLES, "python_exe": PYTHON_EXE}
if MODE in ("all", "pso"):
    required["bilstm_pso"] = BILSTM_PSO
    required["lstm_pso"]   = LSTM_PSO
if MODE in ("all", "nonpso"):
    required["nonpso_cli"] = NONPSO_CLI
for k, v in required.items():
    if not os.path.exists(v):
        log(f"FATAL: missing [{k}]: {v}"); sys.exit(2)
os.makedirs(OUT_ROOT, exist_ok=True)
combos = [(THR_MAIN, s) for s in SEEDS_AT_THR70]
for thr in THRESHOLDS_AT_S42:
    combos.append((thr, SEED_MAIN))
plan = []
for thr, s in combos:
    csv = os.path.join(TABLES, f"training_thr{thr}_seed{s}.csv")
    if not os.path.exists(csv):
        log(f"FATAL: missing CSV: {csv}"); sys.exit(2)
    plan.append((thr, s, csv))
def needs(name, thr, s):
    d = os.path.join(OUT_ROOT, name, f"thr{thr}", f"seed{s}")
    return FORCE_RERUN or not os.path.exists(os.path.join(d, "metrics_summary.json"))
task_list = []
for thr, s, csv in plan:
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
log(f"Plan ({MODE}): {len(task_list)} subprocess calls")
log(f"  PSO     : {sum(1 for t in task_list if t[0] == 'pso')}")
log(f"  non-PSO : {sum(1 for t in task_list if t[0] == 'nonpso')}")
print(flush=True)
INTERESTING_KEYWORDS = [
    "epoch ", "fold ", "Final CV", "particle", "iter", "best loss",
    "best ROC", "best PR", "best cost", "[CV10]", "[FINAL]", "[BEST]",
    "[DONE]", "ROC-AUC", "PR-AUC", "Training LSTM", "Training BILSTM",
    "Training BiLSTM", "SKIP", "ERROR", "Error", "FAIL", "Traceback",
    "OSError", "RuntimeError", "OOM", "MemoryError"
]
NOISE_KEYWORDS = ["tensorflow/core/", "cuda_dnn", "cuda_blas", "cpu_feature_guard",
                  "Loaded cuDNN", "Created device", "compute capability",
                  "oneDNN custom", "rebuild TensorFlow", "WARNING:tensorflow",
                  "tf.compat.v1"]
def is_interesting(line):
    low = line.lower()
    if any(n.lower() in low for n in NOISE_KEYWORDS): return False
    if any(k.lower() in low for k in INTERESTING_KEYWORDS): return True
    return False
def safe_print(prefix, line):
    try:
        print(prefix + line.rstrip(), flush=True)
    except (UnicodeEncodeError, UnicodeDecodeError):
        safe = line.rstrip().encode("ascii", "replace").decode("ascii")
        print(prefix + safe, flush=True)
def run_subprocess(cmd, log_path, label):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as runlog:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            runlog.write(line); runlog.flush()
            if VERBOSE or is_interesting(line):
                safe_print(f"  [{label}] ", line)
        proc.wait()
    return proc.returncode, time.time() - t0
t_total = time.time()
ok = fail = 0
fail_logs = []
pbar = tqdm(total=len(task_list), desc=f"STEP 3 ({MODE})",
            unit="combo",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")
for i, entry in enumerate(task_list, 1):
    kind = entry[0]
    label_short = f"{entry[4]} thr{entry[1]} s{entry[2]}"
    pbar.set_postfix_str(label_short)
    print(f"\n{'=' * 70}", flush=True)
    print(f"[{i:>3}/{len(task_list)}] {label_short}", flush=True)
    print(f"{'=' * 70}", flush=True)
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
        rc, dt = run_subprocess(cmd, log_path, name)
        if rc == 0 and os.path.exists(os.path.join(od, "metrics_summary.json")):
            ok += 1
            msj_path = os.path.join(od, "metrics_summary.json")
            try:
                import json
                with open(msj_path, "r", encoding="utf-8") as f:
                    msj = json.load(f)
                roc = msj.get("cv_oof", {}).get("roc_auc", float("nan"))
                pr  = msj.get("cv_oof", {}).get("pr_auc",  float("nan"))
                print(f"\n  >>> {label_short}: OK ({dt/60:.1f} min)  cv_oof ROC={roc:.4f}  PR={pr:.4f}\n", flush=True)
            except Exception:
                print(f"\n  >>> {label_short}: OK ({dt/60:.1f} min)\n", flush=True)
        else:
            fail += 1; fail_logs.append(log_path)
            print(f"\n  >>> {label_short}: FAIL ({dt/60:.1f} min)  log: {log_path}\n", flush=True)
    else:
        kind, thr, s, csv, name, script, which = entry
        log_dir = os.path.join(OUT_ROOT, "__nonpso_logs__", f"thr{thr}_seed{s}")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "run_log.txt")
        cmd = [PYTHON_EXE, script,
               "--data", csv, "--base_out_dir", OUT_ROOT,
               "--thr", str(thr), "--seed", str(s),
               "--epochs", str(NONPSO_EPOCHS),
               "--grid_km", str(GRID_KM),
               "--models", ",".join(which)]
        rc, dt = run_subprocess(cmd, log_path, label_short)
        per_ok = 0
        for sub_lower in which:
            sub = "LSTM" if sub_lower == "lstm" else "BiLSTM"
            msj = os.path.join(OUT_ROOT, sub, f"thr{thr}", f"seed{s}", "metrics_summary.json")
            if rc == 0 and os.path.exists(msj):
                per_ok += 1
                try:
                    import json
                    with open(msj, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    roc = d.get("cv_oof", {}).get("roc_auc", float("nan"))
                    pr  = d.get("cv_oof", {}).get("pr_auc",  float("nan"))
                    print(f"  >>> {sub} thr{thr} s{s}: OK   cv_oof ROC={roc:.4f}  PR={pr:.4f}", flush=True)
                except Exception: pass
        if rc == 0 and per_ok == len(which):
            ok += len(which)
            print(f"\n  >>> {label_short}: OK ({dt/60:.1f} min)\n", flush=True)
        else:
            fail += len(which); fail_logs.append(log_path)
            print(f"\n  >>> {label_short}: FAIL ({dt/60:.1f} min)  log: {log_path}\n", flush=True)
    pbar.update(1)
pbar.close()
print("\n" + "=" * 70, flush=True)
print(f"STEP 3 FINISHED (mode={MODE}) in {(time.time()-t_total)/3600:.2f} hours", flush=True)
print(f"  OK:   {ok}", flush=True)
print(f"  FAIL: {fail}", flush=True)
print("=" * 70, flush=True)
if fail_logs:
    print("\nFailed run logs:", flush=True)
    for lg in fail_logs[:10]:
        print(f"  {lg}", flush=True)
