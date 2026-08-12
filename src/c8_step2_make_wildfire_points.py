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
print("\nDialog 1 of 2: Pick your fires_geq_70ha.shp source file.")
print("(probably under <your-repo>\\data\\processed\\fires_geq_70ha.shp)")
fires = pick_file(
    "STEP 2 dialog 1: Pick fires_geq_70ha.shp",
    [("Shapefile", "*.shp")],
    initialdir=os.path.expandvars(r"%USERPROFILE%\Documents\wildfire-bc-bilstm-pso\data\processed"),
)
if not fires:
    print("CANCELLED."); sys.exit(1)
print(f"  picked: {fires}")
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
print("\nFiltering SRC_AGENCY = 'BC' ...")
arcpy.analysis.Select(fires, "in_memory/bc_only", "SRC_AGENCY = 'BC'")
n_bc = int(arcpy.management.GetCount("in_memory/bc_only")[0])
print(f"  BC-agency fires: {n_bc:,}")
print("Converting polygons to centroids (INSIDE) ...")
arcpy.management.FeatureToPoint("in_memory/bc_only", out_shp, "INSIDE")
n_pts = int(arcpy.management.GetCount(out_shp)[0])
print(f"  centroids written: {n_pts:,}")
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
print(os.path.expandvars(r"  exec(open(r'%USERPROFILE%\Documents\wildfire-bc-bilstm-pso\src\c8_step3_generate_pseudo_absence.py').read())"))
try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo(
        "STEP 2 complete",
        f"Wrote {n_pts:,} wildfire centroids to:\n{out_shp}\n\nNext: run STEP 3 in ArcGIS Pro.",
    )
    a.destroy()
except Exception:
    pass
