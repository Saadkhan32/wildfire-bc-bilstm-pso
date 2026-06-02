# -*- coding: utf-8 -*-
"""
c8c11_step1_setup_folders.py
============================
STEP 1 of 5 -- Reviewer Comments 8 (random-seed sensitivity)
                            + 11 (70 ha threshold sensitivity).

What this does:
  Asks you (with a folder picker) where to put all the C8+C11 working
  files. Creates the six-folder layout and a config text file.

Why this matters for the reviewer:
  Comments 8 and 11 share the same data pipeline. We address them in one
  workflow: 3 area thresholds (>=70, >=100, >=200 ha) x 10 random seeds
  = 30 reproducible training datasets.

Run this script anywhere (any Python env, no special packages):
    python src/c8c11_step1_setup_folders.py
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def main():
    print("=" * 60)
    print("STEP 1 / 5: Set up folder structure")
    print("=" * 60)
    print()
    print("A folder picker will open. Choose where to keep all the")
    print("C8 + C11 working files. Recommended: G:\\")
    print()

    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    parent = filedialog.askdirectory(
        title="STEP 1: Pick a parent folder for the working files",
    )
    root.destroy()
    if not parent:
        print("CANCELLED."); return

    base = os.path.join(parent, "Wildfire_Reviewer_Response")
    subdirs = [
        "01_Input_Data",
        "01_Input_Data/rasters",
        "02_GIS_Output",
        "03_Training_Tables",
        "04_Model_Scripts",
        "05_Model_Results",
        "06_Final_Tables",
    ]
    print(f"\nCreating folder tree under: {base}")
    for s in subdirs:
        d = os.path.join(base, s)
        os.makedirs(d, exist_ok=True)
        print(f"  OK   {d}")

    cfg = os.path.join(base, "c8c11_config.txt")
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(base)
    print(f"\n[saved] base-path config: {cfg}")

    print()
    print("=" * 60)
    print("DONE with STEP 1.")
    print("=" * 60)
    print()
    print("Before STEP 2, place these files manually:")
    print(f"  - BC boundary shapefile           -> {base}\\01_Input_Data\\")
    print(f"  - Wildfire layer (perimeter or")
    print(f"    centroid; with YEAR + AREA_HA)  -> {base}\\01_Input_Data\\")
    print(f"  - 17 predictor rasters            -> {base}\\01_Input_Data\\rasters\\")
    print(f"  - your PSO script bilstm_pso_fe.py-> {base}\\04_Model_Scripts\\")
    print()
    print("Required raster filenames (rename your TIFFs to match):")
    print("  Aspect.tif, Slope.tif, Elevation.tif, TWI.tif,")
    print("  Profile_Curvature.tif, Plan_Curvature.tif, NDVI.tif, LULC.tif,")
    print("  Max_Temperature.tif, Precipitation.tif, WS.tif,")
    print("  Relative_Humidity.tif,   <- IMPORTANT: relative, not specific")
    print("  AET.tif, DSI.tif, Soil_Moisture.tif,")
    print("  Distance_roads.tif, Distance_rivers.tif, Distance_households.tif")
    print()
    print("Then run STEP 2 in ArcGIS Pro Python window.")

    try:
        a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
        messagebox.showinfo(
            "STEP 1 complete",
            f"Folder structure created at:\n{base}\n\n"
            "Place input data in 01_Input_Data, then run STEP 2.",
        )
        a.destroy()
    except Exception:
        pass

if __name__ == "__main__":
    main()
