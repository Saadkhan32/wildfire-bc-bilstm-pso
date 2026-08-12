import os
import tkinter as tk
from tkinter import filedialog, messagebox
def main():
    print("=" * 60)
    print("STEP 1 / 5: Set up folder structure")
    print("=" * 60)
    print()
    print("A folder picker will open. Choose where to keep all the")
    print("Comment-8 working files. Recommended: G:\\Wildfire_C8")
    print()
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    parent = filedialog.askdirectory(
        title="STEP 1: Pick a parent folder for C8 working files",
    )
    root.destroy()
    if not parent:
        print("CANCELLED. Re-run the script when ready.")
        return
    base = os.path.join(parent, "Wildfire_C8")
    subdirs = [
        "01_GIS",
        "01_GIS/rasters",
        "02_Seed_Tables",
        "03_Model_Results",
        "04_Assumption_Checks",
    ]
    print(f"\nCreating folder tree under: {base}")
    for s in subdirs:
        d = os.path.join(base, s)
        os.makedirs(d, exist_ok=True)
        print(f"  OK   {d}")
    cfg_path = os.path.join(base, "c8_config.txt")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(base)
    print(f"\n[saved] base-path config: {cfg_path}")
    print()
    print("=" * 60)
    print("DONE with STEP 1.")
    print("=" * 60)
    print()
    print("Next: put your input files in these folders:")
    print(f"  - BC boundary shapefile      -> {base}\\01_GIS\\")
    print(f"  - 17 predictor rasters       -> {base}\\01_GIS\\rasters\\")
    print(f"  - fires_geq_70ha.shp source  -> stays in your project repo")
    print()
    print("Then run STEP 2:  python src/c8_step2_make_wildfire_points.py")
    try:
        tk_alert = tk.Tk(); tk_alert.withdraw(); tk_alert.attributes("-topmost", True)
        messagebox.showinfo(
            "STEP 1 complete",
            f"Folder structure created at:\n{base}\n\nNext: run STEP 2.",
        )
        tk_alert.destroy()
    except Exception:
        pass
if __name__ == "__main__":
    main()
