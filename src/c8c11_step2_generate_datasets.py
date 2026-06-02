# -*- coding: utf-8 -*-
"""
c8c11_step2_generate_datasets.py
================================
STEP 2 of 5 -- Reviewer Comments 8 + 11.

What this does:
  Generates 30 training datasets that cover BOTH Comment 8 (random-seed
  sensitivity) and Comment 11 (70 ha threshold sensitivity):

      3 area thresholds (>=70, >=100, >=200 ha)
      x 10 random seeds (42, 101, 202, 303, 404, 505, 606, 707, 808, 909)
      = 30 training CSVs

  For each (threshold, seed):
    1. Select BC wildfire records from 2000-2024 with AREA_HA >= threshold
    2. Convert polygons to centroid points (if not already points)
    3. Buffer wildfire points by 20 km and erase from BC boundary
       -> eligible pseudo-absence area
    4. Generate N pseudo-absence points (N = n_fire) inside eligible area
       with 20 km minimum spacing, using ArcGIS Mersenne-Twister seed
    5. Auto-QC: minimum distance to fire >= 20 km, min spacing >= 20 km
    6. Merge fire + non-fire, sample 17 predictor rasters at each point
    7. Export training_thr<H>_seed<S>.csv with WGS84 lat/lon for PSO

Why both comments share this script:
  Comment 8 = vary the seed at one threshold.
  Comment 11 = vary the threshold at one seed.
  Doing both together (3 x 10 = 30 datasets) lets the PSO sensitivity in
  STEP 3 answer both reviewer comments from one combined results table.

How to run:
  ArcGIS Pro -> Analysis tab -> Python window. Paste:
      exec(open(r"C:\\Users\\saadz\\Documents\\wildfire-bc-bilstm-pso\\src\\c8c11_step2_generate_datasets.py").read())
"""
import os
import sys
import arcpy
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

# ---- Fixed design constants ----
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
THRESHOLDS_HA = [70, 100, 200]
START_YEAR = 2000
END_YEAR   = 2024
BUFFER_DISTANCE      = "20000 Meters"   # 20 km wildfire exclusion buffer
MIN_RANDOM_DISTANCE  = "20000 Meters"   # 20 km min spacing among pseudo-absences
PROJECTED_CRS = arcpy.SpatialReference(3005)  # BC Albers (meters), for buffers/spacing
WGS84         = arcpy.SpatialReference(4326)  # for the Lat/Lon exported to CSV

# ---- Predictor metadata: (filename, csv_field_name, raster_kind) ----
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

# Possible field names for auto-detection
YEAR_CANDIDATES    = ("YEAR", "YEAR_", "FIRE_YEAR", "YEAR_FIRE", "Yr")
AREA_CANDIDATES    = ("AREA_HA", "SIZE_HA", "POLY_HA", "HECTARES", "Area_ha", "AREA")
FIREID_CANDIDATES  = ("NFDBFIREID", "FIRE_ID", "FIREID", "POLY_ID", "OBJECTID")
AGENCY_CANDIDATES  = ("SRC_AGENCY", "AGENCY", "SOURCE", "PROVINCE")

# ---- Tkinter helpers ----
def pick_file(title, ft, initialdir=None):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=title, filetypes=ft, initialdir=initialdir)
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
print("STEP 2 / 5: Generate 30 training datasets")
print("           (3 thresholds x 10 seeds)")
print("=" * 60)

# ============================================================
# Pickers
# ============================================================
print("\nDialog 1 of 5: Pick the BC boundary shapefile.")
bc_shp = pick_file("STEP 2 dialog 1: pick BC boundary .shp",
                    [("Shapefile", "*.shp")])
if not bc_shp:
    print("CANCELLED."); sys.exit(1)
print(f"  BC boundary: {bc_shp}")

print("\nDialog 2 of 5: Pick the wildfire layer (polygons or points).")
print("It must have YEAR + AREA_HA columns (we will autodetect alternatives).")
fire_shp = pick_file("STEP 2 dialog 2: pick wildfire .shp",
                      [("Shapefile", "*.shp")])
if not fire_shp:
    print("CANCELLED."); sys.exit(1)
print(f"  Wildfire layer: {fire_shp}")

print("\nDialog 3 of 5: Pick the rasters folder")
print("(must contain all 18 raster files listed in PREDICTORS).")
rasters_dir = pick_folder("STEP 2 dialog 3: pick rasters folder")
if not rasters_dir:
    print("CANCELLED."); sys.exit(1)
print(f"  Rasters folder: {rasters_dir}")

# Verify all rasters present
predictor_paths = []
missing = []
for fname, csv_name, kind in PREDICTORS:
    p = os.path.join(rasters_dir, fname)
    if os.path.exists(p):
        predictor_paths.append((p, csv_name, kind))
    else:
        missing.append(fname)
if missing:
    print(f"\nERROR: {len(missing)} raster(s) missing:")
    for m in missing:
        print(f"  - {m}")
    print("\nFix: rename your TIFFs to match the expected filenames, or edit")
    print("PREDICTORS at the top of this script. Then re-run STEP 2.")
    sys.exit(2)
print(f"  All {len(predictor_paths)} predictor rasters found.")

print("\nDialog 4 of 5: Pick Wildfire_Reviewer_Response\\02_GIS_Output folder.")
gis_out = pick_folder("STEP 2 dialog 4: pick 02_GIS_Output folder")
if not gis_out:
    print("CANCELLED."); sys.exit(1)
out_gdb = os.path.join(gis_out, "reviewer_response_sampling.gdb")
print(f"  Output gdb: {out_gdb}")

print("\nDialog 5 of 5: Pick Wildfire_Reviewer_Response\\03_Training_Tables folder.")
tables_folder = pick_folder("STEP 2 dialog 5: pick 03_Training_Tables folder")
if not tables_folder:
    print("CANCELLED."); sys.exit(1)
print(f"  Tables folder: {tables_folder}")

# ============================================================
# Project inputs to BC Albers (meters) for distance calculations
# ============================================================
if not arcpy.Exists(out_gdb):
    arcpy.management.CreateFileGDB(os.path.dirname(out_gdb), os.path.basename(out_gdb))
arcpy.env.workspace = out_gdb
arcpy.env.outputCoordinateSystem = PROJECTED_CRS

print("\nProjecting inputs to BC Albers (EPSG:3005) ...")
bc_proj   = os.path.join(out_gdb, "bc_boundary_proj")
fire_proj = os.path.join(out_gdb, "wildfire_input_proj")
if not arcpy.Exists(bc_proj):
    arcpy.management.Project(bc_shp, bc_proj, PROJECTED_CRS)
if not arcpy.Exists(fire_proj):
    arcpy.management.Project(fire_shp, fire_proj, PROJECTED_CRS)

# Auto-detect wildfire layer field names
fire_fields = [f.name for f in arcpy.ListFields(fire_proj)]
def first_match(candidates):
    return next((c for c in candidates if c in fire_fields), None)

YEAR_F   = first_match(YEAR_CANDIDATES)
AREA_F   = first_match(AREA_CANDIDATES)
ID_F     = first_match(FIREID_CANDIDATES)
AGENCY_F = first_match(AGENCY_CANDIDATES)
print(f"\nDetected fields: YEAR={YEAR_F}, AREA={AREA_F}, ID={ID_F}, AGENCY={AGENCY_F}")
if YEAR_F is None or AREA_F is None:
    print("ERROR: could not find YEAR or AREA_HA columns in your wildfire layer.")
    print(f"Available columns: {fire_fields}")
    sys.exit(3)

# If AREA is missing (it shouldn't be since AREA_F was found) or polygons,
# we'd add AREA_HA from geometry. Skipping that here; trust the detected field.

ok = confirm(
    "Ready to run STEP 2?",
    f"Will generate 3 thresholds x 10 seeds = 30 training CSVs.\n\n"
    f"Wildfire field YEAR  = {YEAR_F}\n"
    f"Wildfire field AREA  = {AREA_F}\n"
    f"Wildfire field ID    = {ID_F}\n"
    f"Wildfire field AGENCY= {AGENCY_F}\n\n"
    "Wall-time ~60-90 min total. Proceed?",
)
if not ok:
    sys.exit(0)

# ============================================================
# Build SQL with optional BC-agency filter (BC + PC if available)
# ============================================================
def build_sql(threshold_ha):
    year_d = arcpy.AddFieldDelimiters(out_gdb, YEAR_F)
    area_d = arcpy.AddFieldDelimiters(out_gdb, AREA_F)
    parts = [f"{year_d} >= {START_YEAR}",
             f"{year_d} <= {END_YEAR}",
             f"{area_d} >= {threshold_ha}"]
    if AGENCY_F is not None:
        agency_d = arcpy.AddFieldDelimiters(out_gdb, AGENCY_F)
        parts.append(f"({agency_d} = 'BC' OR {agency_d} = 'PC')")
    return " AND ".join(parts)

def delete_if_exists(p):
    if arcpy.Exists(p):
        arcpy.management.Delete(p)

# ============================================================
# Per-threshold: build fire points + eligible area (done once per threshold)
# ============================================================
fire_layers   = {}   # threshold_ha -> fire_points_fc, n_fire
eligible_diss = {}   # threshold_ha -> eligible_area_fc

for thr in THRESHOLDS_HA:
    print(f"\n===== Threshold >= {thr} ha =====")
    selected_fc  = os.path.join(out_gdb, f"fire_selected_thr{thr}")
    clipped_fc   = os.path.join(out_gdb, f"fire_clipped_thr{thr}")
    fire_points  = os.path.join(out_gdb, f"wildfire_points_thr{thr}")
    fire_buf     = os.path.join(out_gdb, f"fire_buffer_20km_thr{thr}")
    elig_raw     = os.path.join(out_gdb, f"eligible_area_thr{thr}")
    elig_diss    = os.path.join(out_gdb, f"eligible_area_diss_thr{thr}")
    for p in (selected_fc, clipped_fc, fire_points, fire_buf, elig_raw, elig_diss):
        delete_if_exists(p)

    lyr = "wildfire_lyr_temp"
    if arcpy.Exists(lyr):
        arcpy.management.Delete(lyr)
    arcpy.management.MakeFeatureLayer(fire_proj, lyr)
    sql = build_sql(thr)
    print(f"  SQL: {sql}")
    arcpy.management.SelectLayerByAttribute(lyr, "NEW_SELECTION", sql)
    arcpy.management.CopyFeatures(lyr, selected_fc)

    # Deduplicate by ID if available
    if ID_F is not None:
        arcpy.management.DeleteIdentical(selected_fc, [ID_F])

    arcpy.analysis.Clip(selected_fc, bc_proj, clipped_fc)
    desc = arcpy.Describe(clipped_fc)
    if desc.shapeType.lower() == "polygon":
        arcpy.management.FeatureToPoint(clipped_fc, fire_points, "INSIDE")
    else:
        arcpy.management.CopyFeatures(clipped_fc, fire_points)

    # Status = 1
    if "Status" not in [f.name for f in arcpy.ListFields(fire_points)]:
        arcpy.management.AddField(fire_points, "Status", "SHORT")
    arcpy.management.CalculateField(fire_points, "Status", 1, "PYTHON3")
    n_fire = int(arcpy.management.GetCount(fire_points)[0])
    print(f"  wildfire points: n = {n_fire}")

    if n_fire == 0:
        print(f"  SKIP: no fires at threshold {thr}.")
        continue

    # 20 km exclusion buffer + eligible non-fire area
    arcpy.analysis.Buffer(fire_points, fire_buf, BUFFER_DISTANCE, dissolve_option="ALL")
    arcpy.analysis.Erase(bc_proj, fire_buf, elig_raw)
    arcpy.management.Dissolve(elig_raw, elig_diss)

    fire_layers[thr]   = (fire_points, n_fire)
    eligible_diss[thr] = elig_diss

# ============================================================
# Per (threshold, seed): generate pseudo-absences, QC, merge, extract, export
# ============================================================
summary_rows = []

for thr in THRESHOLDS_HA:
    if thr not in fire_layers:
        continue
    fire_points, n_fire = fire_layers[thr]
    eligible_area = eligible_diss[thr]

    for seed in SEEDS:
        tag = f"thr{thr}_seed{seed}"
        print(f"\n----- {tag}  ({SEEDS.index(seed)+1}/{len(SEEDS)} for thr{thr}) -----")
        arcpy.env.randomGenerator = f"{seed} MERSENNE_TWISTER"

        pseudo_fc = os.path.join(out_gdb, f"pseudo_{tag}")
        merged_fc = os.path.join(out_gdb, f"training_points_{tag}")
        for p in (pseudo_fc, merged_fc):
            delete_if_exists(p)

        # Generate pseudo-absence points
        arcpy.management.CreateRandomPoints(
            out_path=out_gdb,
            out_name=f"pseudo_{tag}",
            constraining_feature_class=eligible_area,
            constraining_extent="",
            number_of_points_or_field=n_fire,
            minimum_allowed_distance=MIN_RANDOM_DISTANCE,
            create_multipoint_output="POINT",
        )
        if "Status" not in [f.name for f in arcpy.ListFields(pseudo_fc)]:
            arcpy.management.AddField(pseudo_fc, "Status", "SHORT")
        arcpy.management.CalculateField(pseudo_fc, "Status", 0, "PYTHON3")
        n_pseudo = int(arcpy.management.GetCount(pseudo_fc)[0])
        print(f"  pseudo-absence count: {n_pseudo}")

        # QC distances
        near_fire = os.path.join(out_gdb, f"qc_near_fire_{tag}")
        near_self = os.path.join(out_gdb, f"qc_near_self_{tag}")
        for p in (near_fire, near_self):
            delete_if_exists(p)
        arcpy.analysis.GenerateNearTable(
            pseudo_fc, [fire_points], near_fire,
            location="NO_LOCATION", angle="NO_ANGLE", closest="CLOSEST",
        )
        arcpy.analysis.GenerateNearTable(
            pseudo_fc, [pseudo_fc], near_self,
            location="NO_LOCATION", angle="NO_ANGLE",
            closest="ALL", closest_count=2,
        )
        min_to_fire = min(r[0] for r in arcpy.da.SearchCursor(near_fire, ["NEAR_DIST"]))
        self_dists = [d for in_fid, near_fid, d in arcpy.da.SearchCursor(
            near_self, ["IN_FID", "NEAR_FID", "NEAR_DIST"])
            if in_fid != near_fid and d is not None and d > 0.001]
        min_spacing = min(self_dists) if self_dists else float("nan")
        print(f"  QC: min dist to fire = {min_to_fire:.0f} m, "
              f"min spacing = {min_spacing:.0f} m")

        # Merge fire + non-fire
        arcpy.management.Merge([fire_points, pseudo_fc], merged_fc)
        if "UniqueID" not in [f.name for f in arcpy.ListFields(merged_fc)]:
            arcpy.management.AddField(merged_fc, "UniqueID", "LONG")
            arcpy.management.CalculateField(merged_fc, "UniqueID", "!OBJECTID!", "PYTHON3")

        # Sample predictor rasters (BILINEAR for continuous, NONE for categorical)
        cont = [[p, n] for p, n, k in predictor_paths if k == "continuous"]
        cat  = [[p, n] for p, n, k in predictor_paths if k == "categorical"]
        if cont:
            arcpy.sa.ExtractMultiValuesToPoints(merged_fc, cont, "BILINEAR")
        if cat:
            arcpy.sa.ExtractMultiValuesToPoints(merged_fc, cat, "NONE")

        # WGS84 decimal-degree lat/lon (REQUIRED by your PSO compute_block_ids)
        if "Longitude" not in [f.name for f in arcpy.ListFields(merged_fc)]:
            arcpy.management.AddField(merged_fc, "Longitude", "DOUBLE")
        if "Latitude" not in [f.name for f in arcpy.ListFields(merged_fc)]:
            arcpy.management.AddField(merged_fc, "Latitude", "DOUBLE")
        arcpy.management.CalculateGeometryAttributes(
            merged_fc, [["Longitude", "POINT_X"], ["Latitude", "POINT_Y"]],
            coordinate_system=WGS84,
        )

        # Export CSV
        out_csv = os.path.join(tables_folder, f"training_thr{thr}_seed{seed}.csv")
        fields_export = (["UniqueID", "Longitude", "Latitude", "Status"]
                          + [n for _, n, _ in predictor_paths])
        rows = list(arcpy.da.SearchCursor(merged_fc, fields_export))
        df = pd.DataFrame(rows, columns=fields_export).dropna(subset=["Status"])
        df.to_csv(out_csv, index=False)
        print(f"  wrote {out_csv} ({len(df)} rows)")

        summary_rows.append({
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

# ============================================================
# Write QC summary
# ============================================================
qc_csv = os.path.join(tables_folder, "sampling_QC_summary.csv")
pd.DataFrame(summary_rows).to_csv(qc_csv, index=False)

print("\n" + "=" * 60)
print("STEP 2 DONE.")
print("=" * 60)
print(f"\nTraining CSVs: {len(summary_rows)} files in {tables_folder}")
print(f"QC summary:    {qc_csv}")
print("\nQC overview:")
for r in summary_rows:
    print(f"  thr={r['threshold_ha']:>3}, seed={r['seed']:>3}: "
          f"n_fire={r['wildfire_points']:>4}, n_pseudo={r['pseudo_absence_points']:>4}, "
          f"min_to_fire={r['min_distance_pseudo_to_fire_m']:>7.0f} m, "
          f"min_spacing={r['min_distance_among_pseudo_m']:>7.0f} m, "
          f"counts_OK={r['passed_count']}, "
          f"buffer_OK={r['passed_20km_fire_buffer']}, "
          f"spacing_OK={r['passed_20km_pseudo_spacing']}")
print("\nNext: switch to your wildfire conda env and run STEP 3:")
print("  python src/c8c11_step3_run_pso_sensitivity.py")

try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo("STEP 2 complete",
        f"Wrote {len(summary_rows)} training CSVs.\nNext: STEP 3.")
    a.destroy()
except Exception:
    pass
