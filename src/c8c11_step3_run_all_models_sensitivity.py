import os
import sys
import time
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
THRESHOLDS_HA = [70, 100, 200]
PSO_OBJECTIVE = "roc"
GRID_KM = 50
PROGRESS = "none"
PSO_PARTICLES, PSO_ITERS, PSO_FOLDS = 6, 6, 2
SEARCH_EPOCHS, RETRAIN_EPOCHS = 30, 30
NONPSO_EPOCHS = 30
FORCE_RERUN = False
def pick_file(title, ft):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=title, filetypes=ft)
    r.destroy(); return p
def pick_folder(title):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=title)
    r.destroy(); return p
def confirm(title, msg):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    a = messagebox.askyesno(title, msg)
    r.destroy(); return a
print("=" * 60)
print("STEP 3 / 5: Run all 4 models on 30 datasets (Option C)")
print("=" * 60)
print("\nDialog 1: pick 03_Training_Tables folder")
tables_folder = pick_folder("STEP 3: pick 03_Training_Tables")
if not tables_folder: sys.exit(1)
print(f"  tables: {tables_folder}")
print("\nDialog 2: pick the BiLSTM-PSO script (e.g. 'BiLSTM PSO FE.py')")
bilstm_pso = pick_file("STEP 3: pick BiLSTM PSO FE.py",
                       [("Python", "*.py")])
if not bilstm_pso: sys.exit(1)
print(f"  BiLSTM-PSO: {bilstm_pso}")
print("\nDialog 3: pick the LSTM-PSO script (e.g. 'LSTM PSO FE.py')")
lstm_pso = pick_file("STEP 3: pick LSTM PSO FE.py",
                     [("Python", "*.py")])
if not lstm_pso: sys.exit(1)
print(f"  LSTM-PSO: {lstm_pso}")
print("\nDialog 4: pick c8c11_non_pso_cli.py")
nonpso_cli = pick_file("STEP 3: pick c8c11_non_pso_cli.py",
                       [("Python", "*.py")])
if not nonpso_cli: sys.exit(1)
print(f"  non-PSO CLI: {nonpso_cli}")
print("\nDialog 5: pick 05_Model_Results folder")
out_root = pick_folder("STEP 3: pick 05_Model_Results")
if not out_root: sys.exit(1)
print(f"  output root: {out_root}")
print("\nDialog 6: pick python.exe (wildfire env)")
python_exe = pick_file("STEP 3: pick python.exe",
                       [("Executable", "python.exe"), ("All", "*.*")])
if not python_exe:
    python_exe = sys.executable
print(f"  python: {python_exe}")
plan = []
for thr in THRESHOLDS_HA:
    for s in SEEDS:
        p = os.path.join(tables_folder, f"training_thr{thr}_seed{s}.csv")
        if os.path.exists(p):
            plan.append((thr, s, p))
print(f"\nFound {len(plan)} of {len(THRESHOLDS_HA) * len(SEEDS)} training CSVs.")
if not plan:
    print("Nothing to run."); sys.exit(2)
def needs_run(name, thr, s):
    d = os.path.join(out_root, name, f"thr{thr}", f"seed{s}")
    return FORCE_RERUN or not os.path.exists(os.path.join(d, "metrics_summary.json"))
remain_pso = sum(needs_run("BiLSTM_PSO", t, s) for t, s, _ in plan) \
           + sum(needs_run("LSTM_PSO",   t, s) for t, s, _ in plan)
remain_nps = sum(needs_run("LSTM",  t, s) or needs_run("BiLSTM", t, s)
                 for t, s, _ in plan)
print(f"  PSO runs remaining:     {remain_pso}")
print(f"  non-PSO calls remaining: {remain_nps}  (each trains LSTM + BiLSTM)")
est_h = (remain_pso * 45 + remain_nps * 30) / 60
ok = confirm("Run all 4 models?",
    f"Will run:\n"
    f"  BiLSTM-PSO  +  LSTM-PSO    on every (thr, seed)\n"
    f"  c8c11_non_pso_cli.py       on every (thr, seed)\n\n"
    f"PSO runs remaining:       {remain_pso}\n"
    f"non-PSO calls remaining:  {remain_nps}\n\n"
    f"Wall-time estimate: ~{est_h:.0f} hours.\n"
    f"Script is resumable; you can stop and restart.\n\nProceed?")
if not ok: sys.exit(0)
def run_subprocess(cmd, log_path):
    with open(log_path, "w", encoding="utf-8") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
t_total = time.time()
total_done = 0
total_skip = 0
total_fail = 0
for i, (thr, s, csv) in enumerate(plan, 1):
    print(f"\n{'='*70}")
    print(f"[{i:>2}/{len(plan)}]  thr{thr}  seed{s}")
    print('=' * 70)
    name = "BiLSTM_PSO"
    out_dir = os.path.join(out_root, name, f"thr{thr}", f"seed{s}")
    if needs_run(name, thr, s):
        os.makedirs(out_dir, exist_ok=True)
        cmd = [python_exe, bilstm_pso, "--data", csv, "--out_dir", out_dir,
               "--objective", PSO_OBJECTIVE, "--grid_km", str(GRID_KM),
               "--pso_particles", str(PSO_PARTICLES), "--pso_iters", str(PSO_ITERS),
               "--pso_folds", str(PSO_FOLDS),
               "--search_epochs", str(SEARCH_EPOCHS),
               "--retrain_epochs", str(RETRAIN_EPOCHS), "--progress", PROGRESS]
        print(f"  running {name} ...")
        t0 = time.time()
        r = run_subprocess(cmd, os.path.join(out_dir, "run_log.txt"))
        print(f"  {name} done in {(time.time()-t0)/60:.1f} min "
              f"(returncode {r.returncode})")
        total_done += 1 if r.returncode == 0 else 0
        total_fail += 1 if r.returncode != 0 else 0
    else:
        print(f"  SKIP {name}")
        total_skip += 1
    name = "LSTM_PSO"
    out_dir = os.path.join(out_root, name, f"thr{thr}", f"seed{s}")
    if needs_run(name, thr, s):
        os.makedirs(out_dir, exist_ok=True)
        cmd = [python_exe, lstm_pso, "--data", csv, "--out_dir", out_dir,
               "--objective", PSO_OBJECTIVE, "--grid_km", str(GRID_KM),
               "--pso_particles", str(PSO_PARTICLES), "--pso_iters", str(PSO_ITERS),
               "--pso_folds", str(PSO_FOLDS),
               "--search_epochs", str(SEARCH_EPOCHS),
               "--retrain_epochs", str(RETRAIN_EPOCHS), "--progress", PROGRESS]
        print(f"  running {name} ...")
        t0 = time.time()
        r = run_subprocess(cmd, os.path.join(out_dir, "run_log.txt"))
        print(f"  {name} done in {(time.time()-t0)/60:.1f} min "
              f"(returncode {r.returncode})")
        total_done += 1 if r.returncode == 0 else 0
        total_fail += 1 if r.returncode != 0 else 0
    else:
        print(f"  SKIP {name}")
        total_skip += 1
    need_lstm   = needs_run("LSTM",   thr, s)
    need_bilstm = needs_run("BiLSTM", thr, s)
    if need_lstm or need_bilstm:
        which = []
        if need_lstm:   which.append("lstm")
        if need_bilstm: which.append("bilstm")
        cmd = [python_exe, nonpso_cli, "--data", csv,
               "--base_out_dir", out_root,
               "--thr", str(thr), "--seed", str(s),
               "--epochs", str(NONPSO_EPOCHS),
               "--grid_km", str(GRID_KM),
               "--models", ",".join(which)]
        log_dir = os.path.join(out_root, "__nonpso_logs__", f"thr{thr}_seed{s}")
        os.makedirs(log_dir, exist_ok=True)
        print(f"  running non-PSO CLI ({','.join(which)}) ...")
        t0 = time.time()
        r = run_subprocess(cmd, os.path.join(log_dir, "run_log.txt"))
        print(f"  non-PSO done in {(time.time()-t0)/60:.1f} min "
              f"(returncode {r.returncode})")
        total_done += len(which) if r.returncode == 0 else 0
        total_fail += len(which) if r.returncode != 0 else 0
    else:
        print(f"  SKIP LSTM + BiLSTM (both done)")
        total_skip += 2
print(f"\n{'=' * 70}")
print(f"STEP 3 done in {(time.time()-t_total)/3600:.1f} hours.")
print(f"  successes: {total_done}")
print(f"  skipped:   {total_skip}")
print(f"  failures:  {total_fail}")
print('=' * 70)
print("\nNext: run STEP 4 to collect results:")
print("  python src/c8c11_step4_collect_and_summarize.py")
try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo("STEP 3 done",
        f"successes: {total_done}\nskipped: {total_skip}\nfailures: {total_fail}")
    a.destroy()
except Exception:
    pass
