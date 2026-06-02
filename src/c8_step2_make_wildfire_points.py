# -*- coding: utf-8 -*-
"""
c8_step2_make_wildfire_points.py
================================
STEP 2 of 5 -- Reviewer Comment 8.

What this does:
  Takes your fires_geq_70ha.shp, filters to SRC_AGENCY='BC' (2000-2024),
  converts the polygons to centroid points, saves
  wildfire_points_1992.shp into your 01_GIS folder.

Why this matters for the reviewer:
  This is THE wildfire training set referenced throughout the rebuttal:
  n = 1,992 BC-agency fires of >= 70 ha, 2000-2024. Every later step uses it.

How to run:
  Open ArcGIS Pro -> Catalog or any project.
  Analysis tab -> Python window.
  Paste:

    exec(open(r"C:\\Users\\saadz\\Documents\\wildfire-bc-bilstm-pso\\src\\c8_step2_make_wildfire_points.py").read())
"""
import os
import sys
import arcpy
import tkinter as tk
from tkinter import filedialog, messagebox

arcpy.env.overwriteOutput = True

def pick_file(title, filetypes, initialdir=None):
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=title, filetypes=filetypes,
                                    initialdir=initialdir)
    root.destroy()
    return p

def pick_folder(title, initialdir=None):
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    p = filedialog.askdirectory(title=title, initialdir=initialdir)
    root.destroy()
    return p

print("=" * 60)
print("STEP 2 / 5: Make the 1,992-point wildfire centroid file")
print("=" * 60)

# 1. Pick fires_geq_70ha.shp
print("\nDialog 1 of 2: Pick your fires_geq_70ha.shp source file.")
print("(probably under <your-repo>\\data\\processed\\fires_geq_70ha.shp)")
fires = pick_file(
    "STEP 2 dialog 1: Pick fires_geq_70ha.shp",
    [("Shapefile", "*.shp")],
    initialdir=r"C:\Users\saadz\Documents\wildfire-bc-bilstm-pso\data\processed",
)
if not fires:
    print("CANCELLED."); sys.exit(1)
print(f"  picked: {fires}")

# 2. Pick the Wildfire_C8\01_GIS output folder
print("\nDialog 2 of 2: Pick the C8 01_GIS output folder.")
print("(the one created by STEP 1; ends in ...Wildfire_C8\\01_GIS)")
out_folder = pick_folder(
    "STEP 2 dialog 2: Pick Wildfire_C8\\01_GIS folder",
)
if not out_folder:
    print("CANCELLED."); sys.exit(1)
print(f"  picked: {out_folder}")

out_shp = os.path.join(out_folder, "wildfire_points_1992.shp")
print(f"\nOutput will be: {out_shp}")

# 3. Apply the BC-agency filter and convert polygons to centroids
print("\nFiltering SRC_AGENCY = 'BC' ...")
arcpy.analysis.Select(fires, "in_memory/bc_only", "SRC_AGENCY = 'BC'")
n_bc = int(arcpy.management.GetCount("in_memory/bc_only")[0])
print(f"  BC-agency fires: {n_bc:,}")

print("Converting polygons to centroids (INSIDE) ...")
arcpy.management.FeatureToPoint("in_memory/bc_only", out_shp, "INSIDE")
n_pts = int(arcpy.management.GetCount(out_shp)[0])
print(f"  centroids written: {n_pts:,}")

# 4. Verify
print("\n" + "=" * 60)
if n_pts == 1992:
    print(f"OK: produced exactly 1,992 wildfire centroids.")
else:
    print(f"WARNING: produced {n_pts} points (expected 1,992).")
    print("Check the source file -- it may not include 2024 fires, or it")
    print("may already be a filtered subset.")
print(f"Output: {out_shp}")
print("=" * 60)
print()
print("Next: run STEP 3 inside ArcGIS Pro Python window:")
print(r"  exec(open(r'C:\Users\saadz\Documents\wildfire-bc-bilstm-pso\src\c8_step3_generate_pseudo_absence.py').read())")

try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo(
        "STEP 2 complete",
        f"Wrote {n_pts:,} wildfire centroids to:\n{out_shp}\n\nNext: run STEP 3 in ArcGIS Pro.",
    )
    a.destroy()
except Exception:
    pass
