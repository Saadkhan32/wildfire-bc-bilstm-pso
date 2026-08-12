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
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found; installing into current env ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm",
                            "--quiet", "--disable-pip-version-check"])
    from tqdm import tqdm
PREFLIGHT_THR  = 70
PREFLIGHT_SEED = 42
OBJECTIVE      = "roc"
GRID_KM        = 50
PROGRESS       = "none"
PSO_PARTICLES  = 2
PSO_ITERS      = 1
PSO_FOLDS      = 2
SEARCH_EPOCHS  = 3
RETRAIN_EPOCHS = 3
NONPSO_EPOCHS  = 3
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
    last_useful = ""
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
                last_useful = stripped[:60]
                if pbar is not None:
                    pbar.set_postfix_str(f"{label}: {last_useful}")
        proc.wait()
    return proc.returncode, time.time() - t0
print("=" * 70, flush=True)
print("PRE-FLIGHT TEST for STEP 3 (Visual Studio / VS Code friendly)", flush=True)
print("  - 1 combo (thr70 seed42), tiny PSO budget", flush=True)
print("  - Output in 05_Model_Results_PREFLIGHT (does NOT touch real results)", flush=True)
print("  - Live progress bar + streaming model output", flush=True)
print("=" * 70, flush=True)
cache = load_cache()
print("\nDialog 1/6: pick 03_Training_Tables folder", flush=True)
tables = pick_folder("PREFLIGHT 1/6: 03_Training_Tables", cache.get("tables"))
if not tables: sys.exit(1)
cache["tables"] = tables
print("Dialog 2/6: pick BiLSTM PSO FE.py", flush=True)
bilstm_pso = pick_file("PREFLIGHT 2/6: BiLSTM PSO FE.py",
                       [("Python", "*.py")], cache.get("bilstm_pso"))
if not bilstm_pso: sys.exit(1)
cache["bilstm_pso"] = bilstm_pso
print("Dialog 3/6: pick LSTM PSO FE.py", flush=True)
lstm_pso = pick_file("PREFLIGHT 3/6: LSTM PSO FE.py",
                     [("Python", "*.py")], cache.get("lstm_pso"))
if not lstm_pso: sys.exit(1)
cache["lstm_pso"] = lstm_pso
print("Dialog 4/6: pick c8c11_non_pso_cli.py", flush=True)
nonpso_cli = pick_file("PREFLIGHT 4/6: c8c11_non_pso_cli.py",
                       [("Python", "*.py")], cache.get("nonpso_cli"))
if not nonpso_cli: sys.exit(1)
cache["nonpso_cli"] = nonpso_cli
print("Dialog 5/6: pick parent of 05_Model_Results (= revision_c8c11)", flush=True)
parent_dir = pick_folder("PREFLIGHT 5/6: revision_c8c11 (parent of 05_Model_Results)",
                          cache.get("parent_dir"))
if not parent_dir: sys.exit(1)
cache["parent_dir"] = parent_dir
print("Dialog 6/6: pick wildfire env python.exe", flush=True)
python_exe = pick_file("PREFLIGHT 6/6: python.exe",
    [("Executable", "python.exe"), ("All", "*.*")], cache.get("python_exe"))
if not python_exe: python_exe = sys.executable
cache["python_exe"] = python_exe
save_cache(cache)
print(f"\n[paths cached to {CONFIG_PATH}; future runs will pre-fill the dialogs]", flush=True)
out_root = os.path.join(parent_dir, "05_Model_Results_PREFLIGHT")
os.makedirs(out_root, exist_ok=True)
print(f"\npreflight output root: {out_root}", flush=True)
csv = os.path.join(tables, f"training_thr{PREFLIGHT_THR}_seed{PREFLIGHT_SEED}.csv")
if not os.path.exists(csv):
    print(f"MISSING CSV: {csv}", flush=True); sys.exit(2)
if not confirm("Ready for pre-flight?",
    f"Will run all 4 models on ONE combo (thr{PREFLIGHT_THR} seed{PREFLIGHT_SEED}).\n\n"
    f"Tiny budget -> ~5-15 min.\nLive progress bar in your terminal.\n\nProceed?"):
    sys.exit(0)
results = {}
t_total = time.time()
tasks = [
    ("BiLSTM_PSO", "pso",    bilstm_pso),
    ("LSTM_PSO",   "pso",    lstm_pso),
    ("nonPSO",     "nonpso", nonpso_cli),
]
pbar = tqdm(total=len(tasks), desc="Pre-flight", unit="model",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
for name, kind, script in tasks:
    pbar.set_postfix_str(f"starting {name} ...")
    if kind == "pso":
        od = os.path.join(out_root, name,
                          f"thr{PREFLIGHT_THR}", f"seed{PREFLIGHT_SEED}")
        os.makedirs(od, exist_ok=True)
        cmd = [python_exe, script, "--data", csv, "--out_dir", od,
               "--objective", OBJECTIVE, "--grid_km", str(GRID_KM),
               "--pso_particles", str(PSO_PARTICLES),
               "--pso_iters", str(PSO_ITERS),
               "--pso_folds", str(PSO_FOLDS),
               "--search_epochs", str(SEARCH_EPOCHS),
               "--retrain_epochs", str(RETRAIN_EPOCHS),
               "--progress", PROGRESS]
        log = os.path.join(od, "run_log.txt")
        rc, dt = run_streaming(cmd, log, pbar=pbar, label=name)
        msj = os.path.join(od, "metrics_summary.json")
        results[name] = ("OK" if (rc == 0 and os.path.exists(msj)) else "FAIL",
                         dt, od, log)
    else:
        log_dir = os.path.join(out_root, "__nonpso_logs__",
                                f"thr{PREFLIGHT_THR}_seed{PREFLIGHT_SEED}")
        os.makedirs(log_dir, exist_ok=True)
        log = os.path.join(log_dir, "run_log.txt")
        cmd = [python_exe, script,
               "--data", csv,
               "--base_out_dir", out_root,
               "--thr", str(PREFLIGHT_THR), "--seed", str(PREFLIGHT_SEED),
               "--epochs", str(NONPSO_EPOCHS),
               "--grid_km", str(GRID_KM),
               "--models", "lstm,bilstm"]
        rc, dt = run_streaming(cmd, log, pbar=pbar, label="LSTM+BiLSTM")
        for sub in ("LSTM", "BiLSTM"):
            od = os.path.join(out_root, sub,
                              f"thr{PREFLIGHT_THR}", f"seed{PREFLIGHT_SEED}")
            msj = os.path.join(od, "metrics_summary.json")
            results[sub] = ("OK" if (rc == 0 and os.path.exists(msj)) else "FAIL",
                            dt, od, log)
    pbar.update(1)
pbar.close()
print("\n" + "=" * 70, flush=True)
print("PRE-FLIGHT VERDICT", flush=True)
print("=" * 70, flush=True)
all_ok = True
for name in ("BiLSTM_PSO", "LSTM_PSO", "LSTM", "BiLSTM"):
    if name not in results: continue
    status, dt, od, log = results[name]
    msj_exists = os.path.exists(os.path.join(od, "metrics_summary.json"))
    flag = "OK" if (status == "OK" and msj_exists) else "FAIL"
    if flag != "OK": all_ok = False
    print(f"  {name:12s}  {flag:4s}  ({dt/60:5.1f} min)   log: {log}",
          flush=True)
print(f"\nTotal: {(time.time()-t_total)/60:.1f} min", flush=True)
if all_ok:
    print("\n" + "=" * 70, flush=True)
    print("PRE-FLIGHT PASSED.  You can launch the production run:", flush=True)
    print("  python src\\revision_step3_run_all_models.py", flush=True)
    print("=" * 70, flush=True)
else:
    print("\n" + "=" * 70, flush=True)
    print("PRE-FLIGHT FAILED.  Open the log(s) listed above and send", flush=True)
    print("=" * 70, flush=True)
try:
    a = _root()
    if all_ok:
        messagebox.showinfo("PRE-FLIGHT PASSED",
            "All 4 models trained successfully.\n\n"
            "Launch production STEP 3 now.")
    else:
        messagebox.showerror("PRE-FLIGHT FAILED",
            "One or more models failed. Check the run_log.txt files.")
    a.destroy()
except Exception: pass
