# -*- coding: utf-8 -*-
"""
c8_step3_generate_pseudo_absence.py
===================================
STEP 3 of 5 -- Reviewer Comment 8 (random-seed sensitivity).

What this does:
  Generates 10 pseudo-absence datasets (1,992 points each) using 10
  fixed ArcGIS seeds (42, 101, ..., 909). For each seed it:
    1. Creates 1,992 random points inside BC outside a 20 km wildfire buffer
       with 20 km minimum inter-point spacing (Mersenne Twister, fixed seed).
    2. Auto-runs QC: min distance to fire >= 20 km, min spacing >= 20 km.
    3. Merges fire + non-fire points; samples 17 predictor rasters at each.
    4. Exports training_seed_<seed>.csv with columns:
         UniqueID, Longitude, Latitude, Status, plus the 17 predictors.

Why this matters for the reviewer:
  Huettmann asked 'how reliable and reproducible'. Generating the same
  procedure with 10 different random seeds and showing the model performs
  consistently across them is exactly the empirical answer.

How to run:
  Open ArcGIS Pro. Analysis tab -> Python window. Paste:

    exec(open(r"C:\\Users\\saadz\\Documents\\wildfire-bc-bilstm-pso\\src\\c8_step3_generate_pseudo_absence.py").read())
"""
import os
import sys
import arcpy
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

# ---- Constants (do not change unless you know why) ----
N_PSEUDO_ABSENCE = 1992
WILDFIRE_BUFFER  = "20000 Meters"
MIN_SPACING      = "20000 Meters"
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
PROJECTED_CRS = arcpy.SpatialReference(3005)  # BC Albers Equal Area (meters)

# Predictor metadata. The script will look for rasters with these EXACT
# filenames in the rasters folder you pick. raster_kind controls sampling:
#   "continuous"  -> BILINEAR interpolation (smoother)
#   "categorical" -> NONE (preserves class codes; e.g. LULC)
PREDICTORS = [
    ("Aspect.tif",              "Aspect",              "continuous"),
    ("Slope.tif",               "Slope",               "continuous"),
    ("Elevation.tif",           "Elevation",           "continuous"),
    ("TWI.tif",                 "TWI",                 "continuous"),
    ("ProfileCurv.tif",         "Profile_Curvature",   "continuous"),
    ("PlanCurv.tif",            "Plan_Curvature",      "continuous"),
    ("NDVI.tif",                "NDVI",                "continuous"),
    ("LULC.tif",                "LULC",                "categorical"),
    ("MaxTemp.tif",             "Max_Temperature",     "continuous"),
    ("Precipitation.tif",       "Precipitation",       "continuous"),
    ("WindSpeed.tif",           "WS",                  "continuous"),
    ("RelativeHumidity.tif",    "Relative_Humidity",   "continuous"),
    ("AET.tif",                 "AET",                 "continuous"),
    ("DSI.tif",                 "DSI",                 "continuous"),
    ("SoilMoisture.tif",        "Soil_Moisture",       "continuous"),
    ("DistanceRoads.tif",       "Distance_roads",      "continuous"),
    ("DistanceRivers.tif",      "Distance_rivers",     "continuous"),
    ("DistanceHouseholds.tif",  "Distance_households", "continuous"),
]

def pick_file(title, filetypes, initialdir=None):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=title, filetypes=filetypes,
                                    initialdir=initialdir)
    r.destroy(); return p

def pick_folder(title, initialdir=None):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=title, initialdir=initialdir)
    r.destroy(); return p

def confirm(title, msg):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    a = messagebox.askyesno(title, msg)
    r.destroy(); return a

print("=" * 60)
print("STEP 3 / 5: Generate 10 pseudo-absence datasets")
print("=" * 60)
print()

# 1. Pick BC boundary shapefile
print("Dialog 1 of 5: Pick your BC boundary shapefile.")
bc_shp = pick_file(
    "STEP 3 dialog 1: Pick BC boundary .shp",
    [("Shapefile", "*.shp")],
)
if not bc_shp:
    print("CANCELLED."); sys.exit(1)
print(f"  BC boundary: {bc_shp}")

# 2. Pick the 1,992 wildfire centroid file from STEP 2
print("\nDialog 2 of 5: Pick the wildfire_points_1992.shp file from STEP 2.")
fire_shp = pick_file(
    "STEP 3 dialog 2: Pick wildfire_points_1992.shp",
    [("Shapefile", "*.shp")],
)
if not fire_shp:
    print("CANCELLED."); sys.exit(1)
print(f"  Wildfire points: {fire_shp}")

# 3. Pick the rasters folder (containing the 17 predictor .tif files)
print("\nDialog 3 of 5: Pick the rasters folder.")
print("(should contain Aspect.tif, Slope.tif, NDVI.tif, ..., RelativeHumidity.tif)")
rasters_dir = pick_folder("STEP 3 dialog 3: Pick rasters folder")
if not rasters_dir:
    print("CANCELLED."); sys.exit(1)
print(f"  rasters folder: {rasters_dir}")

# Confirm all expected rasters exist
print("\nChecking predictor rasters ...")
predictor_paths = []
missing = []
for fname, fieldname, kind in PREDICTORS:
    p = os.path.join(rasters_dir, fname)
    if os.path.exists(p):
        predictor_paths.append((p, fieldname, kind))
        print(f"  OK   {fname}")
    else:
        missing.append(fname)
        print(f"  MISS {fname}")
if missing:
    print(f"\nERROR: {len(missing)} raster(s) not found:")
    for m in missing:
        print(f"  - {m}")
    print("\nFix: rename your rasters to match the expected names, or edit")
    print("PREDICTORS at the top of this script. Then re-run STEP 3.")
    sys.exit(2)

# 4. Pick output folder for the seed_sensitivity.gdb + tables
print("\nDialog 4 of 5: Pick the output folder Wildfire_C8\\01_GIS.")
out_gis = pick_folder("STEP 3 dialog 4: Pick Wildfire_C8\\01_GIS folder")
if not out_gis:
    print("CANCELLED."); sys.exit(1)
out_gdb = os.path.join(out_gis, "seed_sensitivity.gdb")
print(f"  output gdb: {out_gdb}")

# 5. Pick output folder for the CSV tables
print("\nDialog 5 of 5: Pick the output folder Wildfire_C8\\02_Seed_Tables.")
tables_folder = pick_folder("STEP 3 dialog 5: Pick Wildfire_C8\\02_Seed_Tables folder")
if not tables_folder:
    print("CANCELLED."); sys.exit(1)
print(f"  tables folder: {tables_folder}")

ok = confirm(
    "Ready to run STEP 3?",
    f"Will generate 10 pseudo-absence datasets (1,992 points each).\n\n"
    f"BC boundary: {os.path.basename(bc_shp)}\n"
    f"Wildfires:   {os.path.basename(fire_shp)}\n"
    f"Rasters:     {rasters_dir}\n"
    f"Output gdb:  {out_gdb}\n"
    f"CSVs:        {tables_folder}\n\n"
    f"Total wall-time: ~30 min (10 seeds x ~3 min each).\n\nProceed?",
)
if not ok:
    print("CANCELLED."); sys.exit(0)

# ============================================================
# Set up file geodatabase + projection
# ============================================================
if not arcpy.Exists(out_gdb):
    arcpy.management.CreateFileGDB(os.path.dirname(out_gdb), os.path.basename(out_gdb))

arcpy.env.workspace = out_gdb
arcpy.env.outputCoordinateSystem = PROJECTED_CRS

print("\nProjecting BC boundary and wildfire points to BC Albers (EPSG:3005) ...")
bc_proj   = os.path.join(out_gdb, "bc_boundary_proj")
fire_proj = os.path.join(out_gdb, "wildfire_points_proj")
if not arcpy.Exists(bc_proj):
    arcpy.management.Project(bc_shp, bc_proj, PROJECTED_CRS)
if not arcpy.Exists(fire_proj):
    arcpy.management.Project(fire_shp, fire_proj, PROJECTED_CRS)

# Add Status = 1 to wildfire points (once)
if "Status" not in [f.name for f in arcpy.ListFields(fire_proj)]:
    arcpy.management.AddField(fire_proj, "Status", "SHORT")
arcpy.management.CalculateField(fire_proj, "Status", 1, "PYTHON3")
n_fire = int(arcpy.management.GetCount(fire_proj)[0])
print(f"  wildfire points (Status=1): n = {n_fire}")
if n_fire != 1992:
    if not confirm(
        "Wildfire count mismatch",
        f"Expected 1,992 wildfire points but got {n_fire}.\n\n"
        "If this is intentional (e.g. you accept a different n) click Yes to continue, otherwise No to stop.",
    ):
        sys.exit(3)

# 20 km wildfire buffer
print("\nBuilding 20 km wildfire exclusion buffer ...")
fire_buf = os.path.join(out_gdb, "wildfire_buffer_20km")
if not arcpy.Exists(fire_buf):
    arcpy.analysis.Buffer(fire_proj, fire_buf, WILDFIRE_BUFFER, dissolve_option="ALL")

# Eligible area = BC boundary minus 20 km wildfire buffer (dissolved)
print("Erasing 20 km buffer from BC boundary to get eligible non-fire area ...")
eligible_raw  = os.path.join(out_gdb, "eligible_area_outside_fire_20km")
eligible_diss = os.path.join(out_gdb, "eligible_area_dissolved")
if not arcpy.Exists(eligible_raw):
    arcpy.analysis.Erase(bc_proj, fire_buf, eligible_raw)
if not arcpy.Exists(eligible_diss):
    arcpy.management.Dissolve(eligible_raw, eligible_diss)

# ============================================================
# Per-seed loop
# ============================================================
summary_rows = []
qc_rows      = []

for i, seed in enumerate(SEEDS, start=1):
    print(f"\n----- seed {seed}  ({i} / {len(SEEDS)}) -----")
    arcpy.env.randomGenerator = f"{seed} MERSENNE_TWISTER"

    pseudo_fc = os.path.join(out_gdb, f"pseudo_absence_seed_{seed}")
    merged_fc = os.path.join(out_gdb, f"training_points_seed_{seed}")
    if arcpy.Exists(pseudo_fc):
        arcpy.management.Delete(pseudo_fc)

    # Generate random points
    arcpy.management.CreateRandomPoints(
        out_path=out_gdb,
        out_name=f"pseudo_absence_seed_{seed}",
        constraining_feature_class=eligible_diss,
        constraining_extent="",
        number_of_points_or_field=N_PSEUDO_ABSENCE,
        minimum_allowed_distance=MIN_SPACING,
        create_multipoint_output="POINT",
    )
    if "Status" not in [f.name for f in arcpy.ListFields(pseudo_fc)]:
        arcpy.management.AddField(pseudo_fc, "Status", "SHORT")
    arcpy.management.CalculateField(pseudo_fc, "Status", 0, "PYTHON3")

    pseudo_count = int(arcpy.management.GetCount(pseudo_fc)[0])
    print(f"  generated {pseudo_count} pseudo-absence points")

    # ---- QC: distances ----
    near_fire = os.path.join(out_gdb, f"qc_near_fire_seed_{seed}")
    if arcpy.Exists(near_fire):
        arcpy.management.Delete(near_fire)
    arcpy.analysis.GenerateNearTable(
        pseudo_fc, [fire_proj], near_fire,
        location="NO_LOCATION", angle="NO_ANGLE", closest="CLOSEST",
    )
    near_self = os.path.join(out_gdb, f"qc_near_self_seed_{seed}")
    if arcpy.Exists(near_self):
        arcpy.management.Delete(near_self)
    arcpy.analysis.GenerateNearTable(
        pseudo_fc, [pseudo_fc], near_self,
        location="NO_LOCATION", angle="NO_ANGLE", closest="CLOSEST",
    )
    min_to_fire = min([r[0] for r in arcpy.da.SearchCursor(near_fire, ["NEAR_DIST"])])
    min_spacing = min([r[0] for r in arcpy.da.SearchCursor(near_self, ["NEAR_DIST"])])
    qc_rows.append({
        "seed": seed,
        "pseudo_absence_count": pseudo_count,
        "min_distance_to_fire_m": round(min_to_fire, 1),
        "min_spacing_among_pseudo_m": round(min_spacing, 1),
        "passed_distance_to_fire_20km": "Yes" if min_to_fire >= 20000 else "No",
        "passed_spacing_20km":          "Yes" if min_spacing >= 20000 else "No",
    })
    print(f"  QC: min distance to fire = {min_to_fire:.0f} m, "
          f"min spacing = {min_spacing:.0f} m")

    # ---- Merge fire + non-fire, add UniqueID ----
    if arcpy.Exists(merged_fc):
        arcpy.management.Delete(merged_fc)
    arcpy.management.Merge([fire_proj, pseudo_fc], merged_fc)
    if "UniqueID" not in [f.name for f in arcpy.ListFields(merged_fc)]:
        arcpy.management.AddField(merged_fc, "UniqueID", "LONG")
        arcpy.management.CalculateField(merged_fc, "UniqueID", "!OBJECTID!", "PYTHON3")

    # ---- Sample predictor rasters ----
    cont = [[p, n] for p, n, k in predictor_paths if k == "continuous"]
    cat  = [[p, n] for p, n, k in predictor_paths if k == "categorical"]
    if cont:
        arcpy.sa.ExtractMultiValuesToPoints(merged_fc, cont, "BILINEAR")
    if cat:
        arcpy.sa.ExtractMultiValuesToPoints(merged_fc, cat, "NONE")

    # ---- Projected X / Y for reference ----
    if "Longitude" not in [f.name for f in arcpy.ListFields(merged_fc)]:
        arcpy.management.AddField(merged_fc, "Longitude", "DOUBLE")
    if "Latitude" not in [f.name for f in arcpy.ListFields(merged_fc)]:
        arcpy.management.AddField(merged_fc, "Latitude", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        merged_fc, [["Longitude", "POINT_X"], ["Latitude", "POINT_Y"]],
        coordinate_system=PROJECTED_CRS,
    )

    # ---- Export to CSV ----
    out_csv = os.path.join(tables_folder, f"training_seed_{seed}.csv")
    fields = ["UniqueID", "Longitude", "Latitude", "Status"] + [n for _, n, _ in predictor_paths]
    rows = list(arcpy.da.SearchCursor(merged_fc, fields))
    pd.DataFrame(rows, columns=fields).to_csv(out_csv, index=False)
    print(f"  wrote {out_csv} ({len(rows)} rows)")

    summary_rows.append({
        "seed": seed,
        "wildfire_count": n_fire,
        "pseudo_absence_count": pseudo_count,
        "total_training_points": int(arcpy.management.GetCount(merged_fc)[0]),
        "csv": out_csv,
    })

# ============================================================
# Write summary + QC tables
# ============================================================
summary_csv = os.path.join(tables_folder, "pseudo_absence_generation_summary.csv")
qc_csv      = os.path.join(tables_folder, "pseudo_absence_qc_table.csv")
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
pd.DataFrame(qc_rows).to_csv(qc_csv, index=False)

print("\n" + "=" * 60)
print("STEP 3 DONE.")
print("=" * 60)
print(f"\n10 training CSVs written to: {tables_folder}")
print(f"\nSummary table:  {summary_csv}")
print(f"QC table:       {qc_csv}\n")
print("QC summary:")
for r in qc_rows:
    print(f"  seed {r['seed']:>3}: n={r['pseudo_absence_count']:>4}  "
          f"min_dist_to_fire={r['min_distance_to_fire_m']:>7.0f} m  "
          f"min_spacing={r['min_spacing_among_pseudo_m']:>7.0f} m  "
          f"buffer_OK={r['passed_distance_to_fire_20km']}  spacing_OK={r['passed_spacing_20km']}")
print("\nNext: switch to your wildfire conda env and run STEP 4:")
print("  conda activate wildfire")
print("  python src/c8_step4_train_sensitivity.py")

try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo(
        "STEP 3 complete",
        f"Wrote 10 training_seed_*.csv files.\n\nNext: STEP 4 (Python training).",
    )
    a.destroy()
except Exception:
    pass
