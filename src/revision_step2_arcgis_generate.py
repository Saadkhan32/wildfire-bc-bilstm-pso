import os
import sys
import arcpy
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")
SEEDS_AT_THR70   = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
THRESHOLDS_AT_S42 = [100, 200]
THR_MAIN = 70
SEED_MAIN = 42
START_YEAR, END_YEAR = 2000, 2024
BUFFER_DISTANCE     = "20000 Meters"
MIN_RANDOM_DISTANCE = "20000 Meters"
PROJECTED_CRS = arcpy.SpatialReference(3005)
WGS84         = arcpy.SpatialReference(4326)
PREDICTORS = [
    ("Aspect.tif",              "Aspect",              "continuous"),
    ("Slope.tif",               "Slope",               "continuous"),
    ("Elevation.tif",           "Elevation",           "continuous"),
    ("TWI.tif",                 "TWI",                 "continuous"),
    ("Profile_Curvature.tif",   "Profile_Curvature",   "continuous"),
    ("Plan_Curvature.tif",      "Plan_Curvature",      "continuous"),
    ("NDVI.tif",                "NDVI",                "continuous"),
    ("LULC.tif",                "LULC",                "categorical"),
    ("Max_Temperature.tif",     "Max_Temperature",     "continuous"),
    ("Precipitation.tif",       "Precipitation",       "continuous"),
    ("WS.tif",                  "WS",                  "continuous"),
    ("Relative_Humidity.tif",   "Relative_Humidity",   "continuous"),
    ("AET.tif",                 "AET",                 "continuous"),
    ("DSI.tif",                 "DSI",                 "continuous"),
    ("Soil_Moisture.tif",       "Soil_Moisture",       "continuous"),
    ("Distance_roads.tif",      "Distance_roads",      "continuous"),
    ("Distance_rivers.tif",     "Distance_rivers",     "continuous"),
    ("Distance_households.tif", "Distance_households", "continuous"),
]
YEAR_CANDIDATES   = ("YEAR", "YEAR_", "FIRE_YEAR", "YEAR_FIRE")
AREA_CANDIDATES   = ("AREA_HA", "SIZE_HA", "POLY_HA", "HECTARES", "AREA")
FIREID_CANDIDATES = ("NFDBFIREID", "FIRE_ID", "FIREID", "POLY_ID", "OBJECTID")
AGENCY_CANDIDATES = ("SRC_AGENCY", "AGENCY", "SOURCE")
def pick_file(t, ft, init=None):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=t, filetypes=ft, initialdir=init)
    r.destroy(); return p
def pick_folder(t, init=None):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=t, initialdir=init)
    r.destroy(); return p
def confirm(t, m):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    a = messagebox.askyesno(t, m); r.destroy(); return a
print("=" * 60)
print("STEP 2 / 5: Generate 12 training CSVs + QC table")
print("=" * 60)
print(f"  10 seeds at thr={THR_MAIN} ha (Comment 8 random-seed sensitivity)")
print(f"  3 thresholds at seed={SEED_MAIN} (Comment 11 threshold sensitivity)")
print(f"  Total = {len(SEEDS_AT_THR70) + len(THRESHOLDS_AT_S42)} CSVs")
print("\nDialog 1: pick BC boundary .shp")
bc_shp = pick_file("STEP 2 dialog 1", [("Shapefile", "*.shp")])
if not bc_shp: sys.exit(1)
print("\nDialog 2: pick wildfire layer .shp (with YEAR + AREA_HA)")
fire_shp = pick_file("STEP 2 dialog 2", [("Shapefile", "*.shp")])
if not fire_shp: sys.exit(1)
print("\nDialog 3: pick the rasters folder (01_Input_Data/rasters)")
rasters_dir = pick_folder("STEP 2 dialog 3")
if not rasters_dir: sys.exit(1)
print("\nDialog 4: pick 02_GIS_Output folder")
gis_out = pick_folder("STEP 2 dialog 4")
if not gis_out: sys.exit(1)
print("\nDialog 5: pick 03_Training_Tables folder")
tables_folder = pick_folder("STEP 2 dialog 5")
if not tables_folder: sys.exit(1)
predictor_paths = []
missing = []
for fname, col, kind in PREDICTORS:
    p = os.path.join(rasters_dir, fname)
    if os.path.exists(p): predictor_paths.append((p, col, kind))
    else: missing.append(fname)
if missing:
    print(f"\nERROR: missing rasters: {missing}")
    sys.exit(2)
print(f"  All {len(predictor_paths)} rasters found.")
combos = [(THR_MAIN, s) for s in SEEDS_AT_THR70]
for thr in THRESHOLDS_AT_S42:
    combos.append((thr, SEED_MAIN))
out_gdb = os.path.join(gis_out, "revision_sampling.gdb")
if not arcpy.Exists(out_gdb):
    arcpy.management.CreateFileGDB(os.path.dirname(out_gdb), os.path.basename(out_gdb))
arcpy.env.workspace = out_gdb
arcpy.env.outputCoordinateSystem = PROJECTED_CRS
bc_proj   = os.path.join(out_gdb, "bc_boundary_proj")
fire_proj = os.path.join(out_gdb, "wildfire_input_proj")
if not arcpy.Exists(bc_proj):
    arcpy.management.Project(bc_shp, bc_proj, PROJECTED_CRS)
if not arcpy.Exists(fire_proj):
    arcpy.management.Project(fire_shp, fire_proj, PROJECTED_CRS)
fields = [f.name for f in arcpy.ListFields(fire_proj)]
YEAR_F   = next((c for c in YEAR_CANDIDATES   if c in fields), None)
AREA_F   = next((c for c in AREA_CANDIDATES   if c in fields), None)
ID_F     = next((c for c in FIREID_CANDIDATES if c in fields), None)
AGENCY_F = next((c for c in AGENCY_CANDIDATES if c in fields), None)
print(f"\nDetected: YEAR={YEAR_F}, AREA={AREA_F}, ID={ID_F}, AGENCY={AGENCY_F}")
if YEAR_F is None or AREA_F is None:
    print(f"ERROR: need YEAR + AREA fields. Available: {fields}"); sys.exit(3)
if not confirm("Ready?",
    f"Will generate {len(combos)} training CSVs in:\n{tables_folder}\n\n"
    "Wall-time ~30 min. Proceed?"):
    sys.exit(0)
def delete_if_exists(p):
    if arcpy.Exists(p): arcpy.management.Delete(p)
def build_sql(thr):
    y = arcpy.AddFieldDelimiters(out_gdb, YEAR_F)
    a = arcpy.AddFieldDelimiters(out_gdb, AREA_F)
    parts = [f"{y} >= {START_YEAR}", f"{y} <= {END_YEAR}", f"{a} >= {thr}"]
    if AGENCY_F:
        ag = arcpy.AddFieldDelimiters(out_gdb, AGENCY_F)
        parts.append(f"({ag} = 'BC' OR {ag} = 'PC')")
    return " AND ".join(parts)
fire_layers   = {}
eligible_diss = {}
for thr in sorted(set([THR_MAIN] + THRESHOLDS_AT_S42)):
    print(f"\n===== Threshold {thr} ha =====")
    selected = os.path.join(out_gdb, f"fire_selected_thr{thr}")
    clipped  = os.path.join(out_gdb, f"fire_clipped_thr{thr}")
    pts      = os.path.join(out_gdb, f"wildfire_points_thr{thr}")
    buf      = os.path.join(out_gdb, f"fire_buffer_20km_thr{thr}")
    elig_raw = os.path.join(out_gdb, f"eligible_area_thr{thr}")
    elig_di  = os.path.join(out_gdb, f"eligible_area_diss_thr{thr}")
    for p in (selected, clipped, pts, buf, elig_raw, elig_di):
        delete_if_exists(p)
    lyr = "wfire_lyr_tmp"
    if arcpy.Exists(lyr): arcpy.management.Delete(lyr)
    arcpy.management.MakeFeatureLayer(fire_proj, lyr)
    sql = build_sql(thr)
    print(f"  SQL: {sql}")
    arcpy.management.SelectLayerByAttribute(lyr, "NEW_SELECTION", sql)
    arcpy.management.CopyFeatures(lyr, selected)
    if ID_F:
        arcpy.management.DeleteIdentical(selected, [ID_F])
    arcpy.analysis.Clip(selected, bc_proj, clipped)
    desc = arcpy.Describe(clipped)
    if desc.shapeType.lower() == "polygon":
        arcpy.management.FeatureToPoint(clipped, pts, "INSIDE")
    else:
        arcpy.management.CopyFeatures(clipped, pts)
    if "Status" not in [f.name for f in arcpy.ListFields(pts)]:
        arcpy.management.AddField(pts, "Status", "SHORT")
    arcpy.management.CalculateField(pts, "Status", 1, "PYTHON3")
    n_fire = int(arcpy.management.GetCount(pts)[0])
    print(f"  wildfire points: n = {n_fire}")
    if n_fire == 0:
        continue
    arcpy.analysis.Buffer(pts, buf, BUFFER_DISTANCE, dissolve_option="ALL")
    arcpy.analysis.Erase(bc_proj, buf, elig_raw)
    arcpy.management.Dissolve(elig_raw, elig_di)
    fire_layers[thr]   = (pts, n_fire)
    eligible_diss[thr] = elig_di
qc_rows = []
for i, (thr, seed) in enumerate(combos, 1):
    print(f"\n----- combo {i}/{len(combos)}: thr={thr} seed={seed} -----")
    fire_points, n_fire = fire_layers[thr]
    eligible = eligible_diss[thr]
    arcpy.env.randomGenerator = f"{seed} MERSENNE_TWISTER"
    pseudo = os.path.join(out_gdb, f"pseudo_thr{thr}_seed{seed}")
    merged = os.path.join(out_gdb, f"training_thr{thr}_seed{seed}")
    for p in (pseudo, merged):
        delete_if_exists(p)
    arcpy.management.CreateRandomPoints(
        out_path=out_gdb,
        out_name=f"pseudo_thr{thr}_seed{seed}",
        constraining_feature_class=eligible,
        constraining_extent="",
        number_of_points_or_field=n_fire,
        minimum_allowed_distance=MIN_RANDOM_DISTANCE,
        create_multipoint_output="POINT",
    )
    if "Status" not in [f.name for f in arcpy.ListFields(pseudo)]:
        arcpy.management.AddField(pseudo, "Status", "SHORT")
    arcpy.management.CalculateField(pseudo, "Status", 0, "PYTHON3")
    n_pseudo = int(arcpy.management.GetCount(pseudo)[0])
    nf = os.path.join(out_gdb, f"qc_nf_thr{thr}_seed{seed}")
    ns = os.path.join(out_gdb, f"qc_ns_thr{thr}_seed{seed}")
    for p in (nf, ns): delete_if_exists(p)
    arcpy.analysis.GenerateNearTable(pseudo, [fire_points], nf,
        location="NO_LOCATION", angle="NO_ANGLE", closest="CLOSEST")
    arcpy.analysis.GenerateNearTable(pseudo, [pseudo], ns,
        location="NO_LOCATION", angle="NO_ANGLE",
        closest="ALL", closest_count=2)
    min_to_fire = min(r[0] for r in arcpy.da.SearchCursor(nf, ["NEAR_DIST"]))
    self_d = [d for i_, n_, d in arcpy.da.SearchCursor(ns,
                ["IN_FID", "NEAR_FID", "NEAR_DIST"])
              if i_ != n_ and d is not None and d > 0.001]
    min_spacing = min(self_d) if self_d else float("nan")
    print(f"  n_pseudo={n_pseudo}  min_to_fire={min_to_fire:.0f} m  "
          f"min_spacing={min_spacing:.0f} m")
    arcpy.management.Merge([fire_points, pseudo], merged)
    if "UniqueID" not in [f.name for f in arcpy.ListFields(merged)]:
        arcpy.management.AddField(merged, "UniqueID", "LONG")
        arcpy.management.CalculateField(merged, "UniqueID", "!OBJECTID!", "PYTHON3")
    cont = [[p, c] for p, c, k in predictor_paths if k == "continuous"]
    cat  = [[p, c] for p, c, k in predictor_paths if k == "categorical"]
    if cont:
        arcpy.sa.ExtractMultiValuesToPoints(merged, cont, "BILINEAR")
    if cat:
        arcpy.sa.ExtractMultiValuesToPoints(merged, cat, "NONE")
    if "Longitude" not in [f.name for f in arcpy.ListFields(merged)]:
        arcpy.management.AddField(merged, "Longitude", "DOUBLE")
    if "Latitude" not in [f.name for f in arcpy.ListFields(merged)]:
        arcpy.management.AddField(merged, "Latitude", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(merged,
        [["Longitude", "POINT_X"], ["Latitude", "POINT_Y"]],
        coordinate_system=WGS84)
    out_csv = os.path.join(tables_folder, f"training_thr{thr}_seed{seed}.csv")
    cols = (["UniqueID", "Longitude", "Latitude", "Status"]
             + [c for _, c, _ in predictor_paths])
    rows = list(arcpy.da.SearchCursor(merged, cols))
    pd.DataFrame(rows, columns=cols).dropna(subset=["Status"]).to_csv(
        out_csv, index=False)
    print(f"  wrote {out_csv}")
    qc_rows.append({
        "threshold_ha":                  thr,
        "seed":                          seed,
        "wildfire_points":               n_fire,
        "pseudo_absence_points":         n_pseudo,
        "total_points":                  n_fire + n_pseudo,
        "min_distance_pseudo_to_fire_m": round(float(min_to_fire), 1),
        "min_distance_among_pseudo_m":   round(float(min_spacing), 1),
        "passed_count":                  bool(n_pseudo == n_fire),
        "passed_20km_fire_buffer":       bool(min_to_fire >= 20000),
        "passed_20km_pseudo_spacing":    bool(np.isnan(min_spacing) or min_spacing >= 20000),
        "training_csv":                  out_csv,
    })
qc_csv = os.path.join(tables_folder, "sampling_QC_summary.csv")
pd.DataFrame(qc_rows).to_csv(qc_csv, index=False)
print("\n" + "=" * 60)
print("STEP 2 DONE.")
print("=" * 60)
print(f"  {len(combos)} training CSVs in: {tables_folder}")
print(f"  QC summary:                    {qc_csv}")
print("\nNext: STEP 3 (run BiLSTM-PSO 12 times):")
print("  conda activate wildfire")
print("  python src/revision_step3_run_bilstm_pso.py")
try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo("STEP 2 complete",
        f"Wrote {len(combos)} training CSVs + QC table.")
    a.destroy()
except Exception: pass
