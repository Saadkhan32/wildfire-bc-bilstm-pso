# -*- coding: utf-8 -*-
"""
revision_step1c_make_base_points.py
===================================
STEP 1c (optional preview, run in ArcGIS Pro Python window).

Quickly creates ONE wildfire + non-wildfire pair so you can visually
verify the sampling in ArcGIS before committing to the 12-combination
Step 2 run.

Produces in 01_Input_Data/:
    wildfire_points_2000_2024.shp          (filtered wildfire centroids)
    pseudo_absence_seed42.shp              (matching non-fire points)
    base_points_summary.csv                (one-row summary)

Filters applied:
    YEAR in [2000, 2024]
    AREA_HA >= 70
    SRC_AGENCY = 'BC' OR 'PC' (if available)
    20 km wildfire exclusion buffer for pseudo-absences
    20 km minimum spacing among pseudo-absences
    Mersenne Twister seed = 42

Run in ArcGIS Pro -> Analysis -> Python window:
    exec(open(r"C:\\Users\\saadz\\Documents\\wildfire-bc-bilstm-pso\\src\\revision_step1c_make_base_points.py").read())
"""
import os
import sys
import arcpy
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

PROJECTED_CRS = arcpy.SpatialReference(3005)  # BC Albers
WGS84         = arcpy.SpatialReference(4326)
SEED          = 42
THRESHOLD_HA  = 70
START_YEAR, END_YEAR = 2000, 2024
BUFFER  = "20000 Meters"
SPACING = "20000 Meters"

YEAR_C   = ("YEAR", "YEAR_", "FIRE_YEAR", "YEAR_FIRE")
AREA_C   = ("AREA_HA", "SIZE_HA", "POLY_HA", "HECTARES", "AREA")
ID_C     = ("NFDBFIREID", "FIRE_ID", "FIREID", "POLY_ID", "OBJECTID")
AGENCY_C = ("SRC_AGENCY", "AGENCY", "SOURCE")

def pf(t, ft):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=t, filetypes=ft); r.destroy(); return p
def pd_(t):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=t); r.destroy(); return p

print("=" * 60)
print("STEP 1c (preview): make 2000-2024 wildfire + non-wildfire points")
print("=" * 60)

print("\nDialog 1: pick the BC boundary .shp")
bc = pf("STEP 1c: BC boundary .shp", [("Shapefile", "*.shp")])
if not bc: sys.exit(1)

print("\nDialog 2: pick the wildfire layer .shp (NFDB or similar)")
fire = pf("STEP 1c: wildfire .shp", [("Shapefile", "*.shp")])
if not fire: sys.exit(1)

print("\nDialog 3: pick the 01_Input_Data folder (output destination)")
out_dir = pd_("STEP 1c: 01_Input_Data folder")
if not out_dir: sys.exit(1)

# Workspace in-memory + a scratch GDB next to the output
scratch = os.path.join(out_dir, "preview_scratch.gdb")
if not arcpy.Exists(scratch):
    arcpy.management.CreateFileGDB(os.path.dirname(scratch),
                                    os.path.basename(scratch))
arcpy.env.workspace = scratch
arcpy.env.outputCoordinateSystem = PROJECTED_CRS

# Project inputs
bc_proj   = os.path.join(scratch, "bc_proj")
fire_proj = os.path.join(scratch, "fire_proj")
if not arcpy.Exists(bc_proj):   arcpy.management.Project(bc, bc_proj, PROJECTED_CRS)
if not arcpy.Exists(fire_proj): arcpy.management.Project(fire, fire_proj, PROJECTED_CRS)

fields = [f.name for f in arcpy.ListFields(fire_proj)]
YEAR_F   = next((c for c in YEAR_C   if c in fields), None)
AREA_F   = next((c for c in AREA_C   if c in fields), None)
ID_F     = next((c for c in ID_C     if c in fields), None)
AGENCY_F = next((c for c in AGENCY_C if c in fields), None)
print(f"\nDetected: YEAR={YEAR_F}, AREA={AREA_F}, ID={ID_F}, AGENCY={AGENCY_F}")
if YEAR_F is None or AREA_F is None:
    print(f"ERROR: missing YEAR/AREA. Available: {fields}")
    sys.exit(2)

# Filter wildfire
y = arcpy.AddFieldDelimiters(scratch, YEAR_F)
a = arcpy.AddFieldDelimiters(scratch, AREA_F)
parts = [f"{y} >= {START_YEAR}", f"{y} <= {END_YEAR}",
         f"{a} >= {THRESHOLD_HA}"]
if AGENCY_F:
    ag = arcpy.AddFieldDelimiters(scratch, AGENCY_F)
    parts.append(f"({ag} = 'BC' OR {ag} = 'PC')")
sql = " AND ".join(parts)
print(f"\nSQL: {sql}")

sel = os.path.join(scratch, "fire_sel")
cl  = os.path.join(scratch, "fire_cl")
pts = os.path.join(scratch, "fire_pts")
for p in (sel, cl, pts):
    if arcpy.Exists(p): arcpy.management.Delete(p)

lyr = "fire_lyr_tmp"
if arcpy.Exists(lyr): arcpy.management.Delete(lyr)
arcpy.management.MakeFeatureLayer(fire_proj, lyr)
arcpy.management.SelectLayerByAttribute(lyr, "NEW_SELECTION", sql)
arcpy.management.CopyFeatures(lyr, sel)
if ID_F:
    arcpy.management.DeleteIdentical(sel, [ID_F])
arcpy.analysis.Clip(sel, bc_proj, cl)
if arcpy.Describe(cl).shapeType.lower() == "polygon":
    arcpy.management.FeatureToPoint(cl, pts, "INSIDE")
else:
    arcpy.management.CopyFeatures(cl, pts)
if "Status" not in [f.name for f in arcpy.ListFields(pts)]:
    arcpy.management.AddField(pts, "Status", "SHORT")
arcpy.management.CalculateField(pts, "Status", 1, "PYTHON3")
n_fire = int(arcpy.management.GetCount(pts)[0])
print(f"  wildfire points (2000-2024, >={THRESHOLD_HA} ha, BC): n = {n_fire}")

# Pseudo-absence area
buf  = os.path.join(scratch, "fire_buf20km")
elig = os.path.join(scratch, "eligible_area")
edis = os.path.join(scratch, "eligible_diss")
for p in (buf, elig, edis):
    if arcpy.Exists(p): arcpy.management.Delete(p)
arcpy.analysis.Buffer(pts, buf, BUFFER, dissolve_option="ALL")
arcpy.analysis.Erase(bc_proj, buf, elig)
arcpy.management.Dissolve(elig, edis)

# Generate pseudo-absence at seed 42
arcpy.env.randomGenerator = f"{SEED} MERSENNE_TWISTER"
pa = os.path.join(scratch, f"pa_seed{SEED}")
if arcpy.Exists(pa): arcpy.management.Delete(pa)
arcpy.management.CreateRandomPoints(
    out_path=scratch, out_name=f"pa_seed{SEED}",
    constraining_feature_class=edis, constraining_extent="",
    number_of_points_or_field=n_fire,
    minimum_allowed_distance=SPACING,
    create_multipoint_output="POINT",
)
if "Status" not in [f.name for f in arcpy.ListFields(pa)]:
    arcpy.management.AddField(pa, "Status", "SHORT")
arcpy.management.CalculateField(pa, "Status", 0, "PYTHON3")
n_pa = int(arcpy.management.GetCount(pa)[0])
print(f"  pseudo-absence points (seed {SEED}): n = {n_pa}")

# QC distances
nf_t = os.path.join(scratch, "qc_nf"); ns_t = os.path.join(scratch, "qc_ns")
for p in (nf_t, ns_t):
    if arcpy.Exists(p): arcpy.management.Delete(p)
arcpy.analysis.GenerateNearTable(pa, [pts], nf_t,
    location="NO_LOCATION", angle="NO_ANGLE", closest="CLOSEST")
arcpy.analysis.GenerateNearTable(pa, [pa], ns_t,
    location="NO_LOCATION", angle="NO_ANGLE",
    closest="ALL", closest_count=2)
min_to_fire = min(r[0] for r in arcpy.da.SearchCursor(nf_t, ["NEAR_DIST"]))
sd = [d for i, n, d in arcpy.da.SearchCursor(ns_t,
        ["IN_FID", "NEAR_FID", "NEAR_DIST"])
      if i != n and d is not None and d > 0.001]
min_spacing = min(sd) if sd else float("nan")
print(f"\nQC: min dist to fire = {min_to_fire:.0f} m  "
      f"min spacing = {min_spacing:.0f} m")

# Export to shapefiles in 01_Input_Data
out_fire = os.path.join(out_dir, "wildfire_points_2000_2024.shp")
out_pa   = os.path.join(out_dir, f"pseudo_absence_seed{SEED}.shp")
for p in (out_fire, out_pa):
    if arcpy.Exists(p): arcpy.management.Delete(p)
arcpy.management.CopyFeatures(pts, out_fire)
arcpy.management.CopyFeatures(pa,  out_pa)
print(f"\n[saved] {out_fire}")
print(f"[saved] {out_pa}")

# One-row CSV summary
qc_csv = os.path.join(out_dir, "base_points_summary.csv")
pd.DataFrame([{
    "year_range":               f"{START_YEAR}-{END_YEAR}",
    "threshold_ha":             THRESHOLD_HA,
    "seed":                     SEED,
    "wildfire_points":          n_fire,
    "pseudo_absence_points":    n_pa,
    "total_points":             n_fire + n_pa,
    "min_dist_pseudo_to_fire_m": round(float(min_to_fire), 1),
    "min_spacing_among_pseudo_m": round(float(min_spacing), 1),
    "buffer_OK_20km":           bool(min_to_fire >= 20000),
    "spacing_OK_20km":           bool(min_spacing >= 20000),
    "wildfire_shp":              out_fire,
    "pseudo_absence_shp":        out_pa,
}]).to_csv(qc_csv, index=False)
print(f"[saved] {qc_csv}")

print("\n" + "=" * 60)
print("STEP 1c preview DONE.")
print("=" * 60)
print(f"  Wildfire points    : {n_fire:>5}   (2000-2024, >={THRESHOLD_HA} ha, BC/PC)")
print(f"  Pseudo-absence pts : {n_pa:>5}   (seed {SEED}, 20 km buffer, 20 km spacing)")
print(f"  Min dist to fire   : {min_to_fire:>7.0f} m  ({'OK' if min_to_fire>=20000 else 'FAIL'})")
print(f"  Min pseudo spacing : {min_spacing:>7.0f} m  ({'OK' if min_spacing>=20000 else 'FAIL'})")
print()
print("Verify visually: in ArcGIS Pro Catalog open these layers and check")
print("there is no overlap and the spread looks reasonable across BC:")
print(f"  {out_fire}")
print(f"  {out_pa}")
print()
print("If the preview looks good, run the full Step 2 to produce 12 datasets:")
print("  exec(open(r\"C:\\Users\\saadz\\Documents\\wildfire-bc-bilstm-pso\\"
      "src\\revision_step2_arcgis_generate.py\").read())")

try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo("STEP 1c preview done",
        f"wildfire: {n_fire}  pseudo-absence: {n_pa}\n"
        f"min dist to fire: {min_to_fire:.0f} m\n"
        f"min spacing: {min_spacing:.0f} m\n\n"
        "If OK, run STEP 2 for all 12 combinations.")
    a.destroy()
except Exception:
    pass
