import os
import sys
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
import json
import time
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found; installing into current env ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm",
                            "--quiet", "--disable-pip-version-check"])
    from tqdm import tqdm
SEEDS_AT_THR70    = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
THRESHOLDS_AT_S42 = [100, 200]
THR_MAIN          = 70
SEED_MAIN         = 42
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
def _root():
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True); return r
def pick_file(t, ft, default=None):
    r = _root()
    initdir = os.path.dirname(default) if default and os.path.exists(default) else None
    initfile = os.path.basename(default) if default else None
    p = filedialog.askopenfilename(title=t, filetypes=ft,
                                    initialdir=initdir, initialfile=initfile)
    r.destroy(); return p
def pick_folder(t, default=None):
    r = _root()
    initdir = default if default and os.path.exists(default) else None
    p = filedialog.askdirectory(title=t, initialdir=initdir)
    r.destroy(); return p
def confirm(t, m):
    r = _root(); a = messagebox.askyesno(t, m); r.destroy(); return a
def load_cache():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception: pass
    return {}
def save_cache(c):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(c, f, indent=2)
    except Exception: pass
def run_streaming(cmd, log_path, pbar=None, label=""):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            logf.write(line); logf.flush()
            stripped = line.rstrip()
            tqdm.write(f"      [{label}] {stripped}")
            low = stripped.lower()
            if any(k in low for k in ("epoch", "fold", "particle", "iter", "auc")):
                if pbar is not None:
                    pbar.set_postfix_str(f"{label}: {stripped[:60]}")
        proc.wait()
    return proc.returncode, time.time() - t0
print("=" * 70, flush=True)
print("STEP 3 / 5: Run ALL 4 models on 12 datasets (Option C lean)", flush=True)
print("  - 48 runs total, resumable", flush=True)
print("  - Live progress bar + streaming model output", flush=True)
print("=" * 70, flush=True)
cache = load_cache()
print("\nDialog 1/6: pick 03_Training_Tables folder", flush=True)
tables = pick_folder("STEP 3 1/6: 03_Training_Tables", cache.get("tables"))
if not tables: sys.exit(1)
cache["tables"] = tables
print("Dialog 2/6: pick BiLSTM PSO FE.py", flush=True)
bilstm_pso = pick_file("STEP 3 2/6: BiLSTM PSO FE.py",
                       [("Python", "*.py")], cache.get("bilstm_pso"))
if not bilstm_pso: sys.exit(1)
cache["bilstm_pso"] = bilstm_pso
print("Dialog 3/6: pick LSTM PSO FE.py", flush=True)
lstm_pso = pick_file("STEP 3 3/6: LSTM PSO FE.py",
                     [("Python", "*.py")], cache.get("lstm_pso"))
if not lstm_pso: sys.exit(1)
cache["lstm_pso"] = lstm_pso
print("Dialog 4/6: pick c8c11_non_pso_cli.py", flush=True)
nonpso_cli = pick_file("STEP 3 4/6: c8c11_non_pso_cli.py",
                       [("Python", "*.py")], cache.get("nonpso_cli"))
if not nonpso_cli: sys.exit(1)
cache["nonpso_cli"] = nonpso_cli
print("Dialog 5/6: pick 05_Model_Results folder", flush=True)
out_root = pick_folder("STEP 3 5/6: 05_Model_Results", cache.get("out_root"))
if not out_root: sys.exit(1)
cache["out_root"] = out_root
print("Dialog 6/6: pick wildfire env python.exe", flush=True)
python_exe = pick_file("STEP 3 6/6: python.exe",
    [("Executable", "python.exe"), ("All", "*.*")], cache.get("python_exe"))
if not python_exe: python_exe = sys.executable
cache["python_exe"] = python_exe
save_cache(cache)
print(f"\n[paths cached to {CONFIG_PATH}]", flush=True)
combos = [(THR_MAIN, s) for s in SEEDS_AT_THR70]
for thr in THRESHOLDS_AT_S42:
    combos.append((thr, SEED_MAIN))
plan = []
for thr, s in combos:
    csv = os.path.join(tables, f"training_thr{thr}_seed{s}.csv")
    if not os.path.exists(csv):
        print(f"MISSING CSV: {csv}", flush=True); sys.exit(2)
    plan.append((thr, s, csv))
def needs(name, thr, s):
    d = os.path.join(out_root, name, f"thr{thr}", f"seed{s}")
    return FORCE_RERUN or not os.path.exists(os.path.join(d, "metrics_summary.json"))
task_list = []
for ci, (thr, s, csv) in enumerate(plan):
    if needs("BiLSTM_PSO", thr, s):
        task_list.append((ci, thr, s, csv, "BiLSTM_PSO", "pso", bilstm_pso))
    if needs("LSTM_PSO", thr, s):
        task_list.append((ci, thr, s, csv, "LSTM_PSO",   "pso", lstm_pso))
    if needs("LSTM", thr, s) or needs("BiLSTM", thr, s):
        which = []
        if needs("LSTM", thr, s):   which.append("lstm")
        if needs("BiLSTM", thr, s): which.append("bilstm")
        task_list.append((ci, thr, s, csv, "+".join(w.upper() for w in which),
                          "nonpso", nonpso_cli, which))
n_pso    = sum(1 for t in task_list if t[5] == "pso")
n_nonpso = sum(1 for t in task_list if t[5] == "nonpso")
est_h = (n_pso * 45 + n_nonpso * 25) / 60
print(f"\nPlan: 4 models x {len(plan)} combos = {4*len(plan)} runs.", flush=True)
print(f"  PSO subprocess runs remaining:      {n_pso}", flush=True)
print(f"  non-PSO subprocess runs remaining:  {n_nonpso}", flush=True)
print(f"  Estimated wall-time:                ~{est_h:.0f} hours", flush=True)
if not confirm("Ready?",
    f"4 models x {len(plan)} combos.\nPSO remaining: {n_pso}\n"
    f"non-PSO remaining: {n_nonpso}\nEstimate: ~{est_h:.0f} hours.\n\nProceed?"):
    sys.exit(0)
t_total = time.time()
ok = 0; fail = 0
fail_logs = []
pbar = tqdm(total=len(task_list), desc="STEP 3", unit="run",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
for entry in task_list:
    if entry[5] == "pso":
        ci, thr, s, csv, name, kind, script = entry
        od = os.path.join(out_root, name, f"thr{thr}", f"seed{s}")
        os.makedirs(od, exist_ok=True)
        cmd = [python_exe, script, "--data", csv, "--out_dir", od,
               "--objective", OBJECTIVE, "--grid_km", str(GRID_KM),
               "--pso_particles", str(PSO_PARTICLES),
               "--pso_iters", str(PSO_ITERS), "--pso_folds", str(PSO_FOLDS),
               "--search_epochs", str(SEARCH_EPOCHS),
               "--retrain_epochs", str(RETRAIN_EPOCHS),
               "--progress", PROGRESS]
        log = os.path.join(od, "run_log.txt")
        label = f"{name} thr{thr} s{s}"
        pbar.set_postfix_str(f"starting {label}")
        rc, dt = run_streaming(cmd, log, pbar=pbar, label=label)
        if rc == 0 and os.path.exists(os.path.join(od, "metrics_summary.json")):
            ok += 1
            tqdm.write(f"  >>> {label}: OK ({dt/60:.1f} min)")
        else:
            fail += 1; fail_logs.append(log)
            tqdm.write(f"  >>> {label}: FAIL ({dt/60:.1f} min)  log: {log}")
    else:
        ci, thr, s, csv, name, kind, script, which = entry
        log_dir = os.path.join(out_root, "__nonpso_logs__", f"thr{thr}_seed{s}")
        os.makedirs(log_dir, exist_ok=True)
        log = os.path.join(log_dir, "run_log.txt")
        cmd = [python_exe, script,
               "--data", csv,
               "--base_out_dir", out_root,
               "--thr", str(thr), "--seed", str(s),
               "--epochs", str(NONPSO_EPOCHS),
               "--grid_km", str(GRID_KM),
               "--models", ",".join(which)]
        label = f"{'+'.join(w.upper() for w in which)} thr{thr} s{s}"
        pbar.set_postfix_str(f"starting {label}")
        rc, dt = run_streaming(cmd, log, pbar=pbar, label=label)
        per_ok = 0
        for sub_lower in which:
            sub = "LSTM" if sub_lower == "lstm" else "BiLSTM"
            msj = os.path.join(out_root, sub, f"thr{thr}", f"seed{s}",
                                "metrics_summary.json")
            if rc == 0 and os.path.exists(msj):
                per_ok += 1
        if rc == 0 and per_ok == len(which):
            ok += len(which)
            tqdm.write(f"  >>> {label}: OK ({dt/60:.1f} min)")
        else:
            fail += len(which); fail_logs.append(log)
            tqdm.write(f"  >>> {label}: FAIL ({dt/60:.1f} min)  log: {log}")
    pbar.update(1)
pbar.close()
print("\n" + "=" * 70, flush=True)
print(f"STEP 3 finished in {(time.time()-t_total)/3600:.2f} hours", flush=True)
print(f"  OK:   {ok}", flush=True)
print(f"  FAIL: {fail}", flush=True)
print("=" * 70, flush=True)
if fail_logs:
    print("\nFailed logs (first 5):", flush=True)
    for lg in fail_logs[:5]:
        print(f"  {lg}", flush=True)
print("\nNext: STEP 4", flush=True)
print("  python src\\revision_step4_summarize_all_models.py", flush=True)
try:
    a = _root()
    if fail == 0:
        messagebox.showinfo("STEP 3 done", f"OK: {ok}  FAIL: 0")
    else:
        messagebox.showwarning("STEP 3 done with failures",
            f"OK: {ok}\nFAIL: {fail}\n\nSee log paths in terminal.")
    a.destroy()
except Exception: pass
