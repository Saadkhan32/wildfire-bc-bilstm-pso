import os
import sys
import time
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
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
FORCE_RERUN    = False
def pick_file(t, ft):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=t, filetypes=ft)
    r.destroy(); return p
def pick_folder(t):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=t)
    r.destroy(); return p
def confirm(t, m):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    a = messagebox.askyesno(t, m); r.destroy(); return a
print("=" * 60)
print("STEP 3 / 5: Run BiLSTM-PSO 12 times")
print("=" * 60)
print("\nDialog 1: pick 03_Training_Tables folder")
tables_folder = pick_folder("STEP 3 dialog 1")
if not tables_folder: sys.exit(1)
print("\nDialog 2: pick 'BiLSTM PSO FE.py' (your PSO training script)")
pso_script = pick_file("STEP 3 dialog 2", [("Python", "*.py")])
if not pso_script: sys.exit(1)
print("\nDialog 3: pick 05_Model_Results folder")
out_root = pick_folder("STEP 3 dialog 3")
if not out_root: sys.exit(1)
print("\nDialog 4: pick python.exe of the wildfire conda env")
python_exe = pick_file("STEP 3 dialog 4",
    [("Executable", "python.exe"), ("All", "*.*")])
if not python_exe:
    python_exe = sys.executable
    print(f"  fallback to sys.executable: {python_exe}")
combos = [(THR_MAIN, s) for s in SEEDS_AT_THR70]
for thr in THRESHOLDS_AT_S42:
    combos.append((thr, SEED_MAIN))
plan = []
missing_csv = []
for thr, s in combos:
    csv = os.path.join(tables_folder, f"training_thr{thr}_seed{s}.csv")
    if os.path.exists(csv):
        plan.append((thr, s, csv))
    else:
        missing_csv.append((thr, s, csv))
if missing_csv:
    print("\nERROR: missing training CSVs from STEP 2:")
    for thr, s, p in missing_csv:
        print(f"  - thr{thr} seed{s}: {p}")
    sys.exit(2)
remaining = []
for thr, s, csv in plan:
    od = os.path.join(out_root, f"thr{thr}", f"seed{s}")
    sj = os.path.join(od, "metrics_summary.json")
    if FORCE_RERUN or not os.path.exists(sj):
        remaining.append((thr, s, csv, od))
print(f"\nPlan: {len(plan)} runs total, {len(remaining)} remaining.")
est_h = len(remaining) * 45 / 60
if not confirm("Ready?",
    f"Will run BiLSTM-PSO on {len(remaining)} dataset(s).\n\n"
    f"Wall-time estimate: ~{est_h:.1f} hours.\n"
    f"Resumable; you can stop and restart.\n\nProceed?"):
    sys.exit(0)
t_total = time.time()
ok = 0; fail = 0
for i, (thr, s, csv, od) in enumerate(remaining, 1):
    os.makedirs(od, exist_ok=True)
    cmd = [python_exe, pso_script,
           "--data", csv, "--out_dir", od,
           "--objective", OBJECTIVE, "--grid_km", str(GRID_KM),
           "--pso_particles", str(PSO_PARTICLES),
           "--pso_iters", str(PSO_ITERS),
           "--pso_folds", str(PSO_FOLDS),
           "--search_epochs", str(SEARCH_EPOCHS),
           "--retrain_epochs", str(RETRAIN_EPOCHS),
           "--progress", PROGRESS]
    log = os.path.join(od, "run_log.txt")
    print(f"\n[{i:>2}/{len(remaining)}]  thr{thr} seed{s}")
    print(f"  data: {csv}")
    print(f"  out : {od}")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    dt = (time.time() - t0) / 60
    if r.returncode == 0:
        print(f"  done in {dt:.1f} min")
        ok += 1
    else:
        print(f"  FAILED in {dt:.1f} min. See {log}")
        fail += 1
print("\n" + "=" * 60)
print(f"STEP 3 finished in {(time.time()-t_total)/3600:.1f} hours.")
print(f"  successes: {ok}")
print(f"  failures:  {fail}")
print("=" * 60)
print("\nNext: STEP 4 (collect + summarize):")
print("  python src/revision_step4_summarize.py")
try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo("STEP 3 done", f"ok: {ok}\nfail: {fail}")
    a.destroy()
except Exception:
    pass
