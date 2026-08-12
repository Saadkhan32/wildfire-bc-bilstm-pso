import os
import sys
import time
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
THRESHOLDS_HA = [70, 100, 200]
OBJECTIVE = "roc"
GRID_KM = 50
PROGRESS = "none"
PSO_PARTICLES  = 6
PSO_ITERS      = 6
PSO_FOLDS      = 2
SEARCH_EPOCHS  = 30
RETRAIN_EPOCHS = 30
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
print("STEP 3 / 5: Run BiLSTM-PSO on all 30 training datasets")
print("=" * 60)
print("\nDialog 1 of 4: Pick the folder with training CSVs.")
print("(Wildfire_Reviewer_Response\\03_Training_Tables)")
tables_folder = pick_folder("STEP 3 dialog 1: pick 03_Training_Tables folder")
if not tables_folder:
    print("CANCELLED."); sys.exit(1)
print(f"  tables folder: {tables_folder}")
print("\nDialog 2 of 4: Pick your PSO training script (.py).")
print("Typically 'BiLSTM PSO FE.py' or its renamed copy 'bilstm_pso_fe.py'.")
pso_script = pick_file("STEP 3 dialog 2: pick PSO training script (.py)",
                        [("Python file", "*.py")])
if not pso_script:
    print("CANCELLED."); sys.exit(1)
print(f"  PSO script: {pso_script}")
print("\nDialog 3 of 4: Pick Wildfire_Reviewer_Response\\05_Model_Results.")
out_root = pick_folder("STEP 3 dialog 3: pick 05_Model_Results folder")
if not out_root:
    print("CANCELLED."); sys.exit(1)
print(f"  output root: {out_root}")
print("\nDialog 4 of 4: Pick the Python interpreter to use.")
print(os.path.expandvars("(your wildfire conda env's python.exe; typically %USERPROFILE%\\"))
print(" anaconda3\\envs\\wildfire\\python.exe or similar)")
python_exe = pick_file("STEP 3 dialog 4: pick python.exe",
                        [("Executable", "python.exe"), ("All", "*.*")])
if not python_exe:
    print("Using sys.executable as fallback:", sys.executable)
    python_exe = sys.executable
print(f"  python interpreter: {python_exe}")
plan = []
missing = []
for thr in THRESHOLDS_HA:
    for s in SEEDS:
        csv_path = os.path.join(tables_folder, f"training_thr{thr}_seed{s}.csv")
        if os.path.exists(csv_path):
            plan.append((thr, s, csv_path))
        else:
            missing.append((thr, s, csv_path))
print(f"\nFound {len(plan)} of {len(THRESHOLDS_HA) * len(SEEDS)} expected CSVs.")
if missing:
    print(f"Missing {len(missing)}:")
    for thr, s, p in missing[:5]:
        print(f"  - thr{thr} seed{s}")
    if len(missing) > 5:
        print(f"  ... and {len(missing) - 5} more")
if not plan:
    print("Nothing to run. Did STEP 2 complete?"); sys.exit(2)
to_run = 0
already = 0
for thr, s, csv_path in plan:
    out_dir = os.path.join(out_root, f"thr{thr}", f"seed{s}")
    summary_json = os.path.join(out_dir, "metrics_summary.json")
    if os.path.exists(summary_json) and not FORCE_RERUN:
        already += 1
    else:
        to_run += 1
print(f"\n{already} runs already complete (will skip).")
print(f"{to_run} runs remaining.")
ok = confirm("Ready to run STEP 3?",
    f"Will run BiLSTM-PSO on {to_run} dataset(s).\n\n"
    f"Each PSO run takes ~30-60 minutes on a single GPU.\n"
    f"Total wall-time estimate: ~{to_run * 45 // 60} hours.\n\n"
    "You can stop the script at any time; completed runs are skipped on restart.\n\n"
    "Proceed?")
if not ok:
    sys.exit(0)
t_start = time.time()
for i, (thr, s, csv_path) in enumerate(plan, 1):
    out_dir = os.path.join(out_root, f"thr{thr}", f"seed{s}")
    os.makedirs(out_dir, exist_ok=True)
    summary_json = os.path.join(out_dir, "metrics_summary.json")
    log_path = os.path.join(out_dir, "run_log.txt")
    print("\n" + "=" * 60)
    print(f"[{i:>2} / {len(plan)}]  thr{thr} seed{s}")
    print("=" * 60)
    if os.path.exists(summary_json) and not FORCE_RERUN:
        print(f"  SKIP (already done): {summary_json}")
        continue
    cmd = [
        python_exe, pso_script,
        "--data", csv_path,
        "--out_dir", out_dir,
        "--objective", OBJECTIVE,
        "--grid_km", str(GRID_KM),
        "--pso_particles", str(PSO_PARTICLES),
        "--pso_iters", str(PSO_ITERS),
        "--pso_folds", str(PSO_FOLDS),
        "--search_epochs", str(SEARCH_EPOCHS),
        "--retrain_epochs", str(RETRAIN_EPOCHS),
        "--progress", PROGRESS,
    ]
    print("  cmd:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  FAILED in {dt/60:.1f} min. See log: {log_path}")
        if not confirm("PSO run failed",
            f"thr{thr} seed{s} failed.\nCheck:\n{log_path}\n\nContinue with remaining runs?"):
            sys.exit(3)
    else:
        print(f"  DONE in {dt/60:.1f} min.")
        print(f"  metrics_summary.json: {summary_json}")
total = time.time() - t_start
print(f"\nAll runs finished in {total/3600:.1f} hours.")
print("\nNext: run STEP 4 to collect results into summary tables:")
print("  python src/c8c11_step4_collect_and_summarize.py")
try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo("STEP 3 complete",
        f"Ran BiLSTM-PSO on {len(plan)} datasets.\nNext: STEP 4.")
    a.destroy()
except Exception:
    pass
