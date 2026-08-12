import os
import arcpy
import pandas as pd
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")
PROJECT_FOLDER = r"G:\Wildfire_RandomSeed_Sensitivity"
OUT_GDB = os.path.join(PROJECT_FOLDER, "01_GIS", "seed_sensitivity.gdb")
BC_BOUNDARY = r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\bc_boundary.shp"
WILDFIRE_POINTS = r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\wildfire_points_1992.shp"
PROJECTED_CRS = arcpy.SpatialReference(3005)
N_PSEUDO_ABSENCE = 1992
WILDFIRE_BUFFER = "20000 Meters"
MIN_SPACING     = "20000 Meters"
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
PREDICTOR_RASTERS = [
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\Aspect.tif",              "Aspect",              "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\Slope.tif",               "Slope",               "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\Elevation.tif",           "Elevation",           "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\TWI.tif",                 "TWI",                 "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\ProfileCurv.tif",         "Profile_Curvature",   "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\PlanCurv.tif",            "Plan_Curvature",      "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\NDVI.tif",                "NDVI",                "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\LULC.tif",                "LULC",                "categorical"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\MaxTemp.tif",             "Max_Temperature",     "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\Precipitation.tif",       "Precipitation",       "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\WindSpeed.tif",           "WS",                  "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\RelativeHumidity.tif",    "Relative_Humidity",   "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\AET.tif",                 "AET",                 "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\DSI.tif",                 "DSI",                 "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\SoilMoisture.tif",        "Soil_Moisture",       "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\DistanceRoads.tif",       "Distance_roads",      "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\DistanceRivers.tif",      "Distance_rivers",     "continuous"],
    [r"G:\Wildfire_RandomSeed_Sensitivity\01_GIS\rasters\DistanceHouseholds.tif",  "Distance_households", "continuous"],
]
if not arcpy.Exists(OUT_GDB):
    arcpy.management.CreateFileGDB(os.path.dirname(OUT_GDB), os.path.basename(OUT_GDB))
TABLES_FOLDER = os.path.join(PROJECT_FOLDER, "02_Seed_Tables")
os.makedirs(TABLES_FOLDER, exist_ok=True)
arcpy.env.workspace = OUT_GDB
arcpy.env.outputCoordinateSystem = PROJECTED_CRS
bc_proj   = os.path.join(OUT_GDB, "bc_boundary_proj")
fire_proj = os.path.join(OUT_GDB, "wildfire_points_proj")
if not arcpy.Exists(bc_proj):
    arcpy.management.Project(BC_BOUNDARY, bc_proj, PROJECTED_CRS)
if not arcpy.Exists(fire_proj):
    arcpy.management.Project(WILDFIRE_POINTS, fire_proj, PROJECTED_CRS)
if "Status" not in [f.name for f in arcpy.ListFields(fire_proj)]:
    arcpy.management.AddField(fire_proj, "Status", "SHORT")
arcpy.management.CalculateField(fire_proj, "Status", 1, "PYTHON3")
n_fire = int(arcpy.management.GetCount(fire_proj)[0])
print(f"[INFO] Wildfire points (Status=1): n = {n_fire}")
if n_fire != 1992:
    print(f"[WARNING] Expected n_fire = 1992 (BC-agency, 2000-2024). Got {n_fire}.")
    print("[WARNING] Verify WILDFIRE_POINTS is the SRC_AGENCY='BC' subset of fires_geq_70ha.")
fire_buf = os.path.join(OUT_GDB, "wildfire_buffer_20km")
if not arcpy.Exists(fire_buf):
    arcpy.analysis.Buffer(fire_proj, fire_buf, WILDFIRE_BUFFER, dissolve_option="ALL")
eligible_raw  = os.path.join(OUT_GDB, "eligible_area_outside_fire_20km")
eligible_diss = os.path.join(OUT_GDB, "eligible_area_dissolved")
if not arcpy.Exists(eligible_raw):
    arcpy.analysis.Erase(bc_proj, fire_buf, eligible_raw)
if not arcpy.Exists(eligible_diss):
    arcpy.management.Dissolve(eligible_raw, eligible_diss)
summary_rows = []
qc_rows      = []
for seed in SEEDS:
    print(f"\n========== Processing seed {seed} ==========")
    arcpy.env.randomGenerator = f"{seed} MERSENNE_TWISTER"
    pseudo_fc = os.path.join(OUT_GDB, f"pseudo_absence_seed_{seed}")
    merged_fc = os.path.join(OUT_GDB, f"training_points_seed_{seed}")
    if arcpy.Exists(pseudo_fc):
        arcpy.management.Delete(pseudo_fc)
    arcpy.management.CreateRandomPoints(
        out_path=OUT_GDB,
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
    print(f"[seed {seed}] pseudo-absence count = {pseudo_count}")
    if pseudo_count < N_PSEUDO_ABSENCE:
        print(f"[WARNING] seed {seed} produced only {pseudo_count} / {N_PSEUDO_ABSENCE} points "
              f"(20 km spacing too tight in available area).")
    near_fire = os.path.join(OUT_GDB, f"qc_near_fire_seed_{seed}")
    if arcpy.Exists(near_fire):
        arcpy.management.Delete(near_fire)
    arcpy.analysis.GenerateNearTable(
        in_features=pseudo_fc, near_features=[fire_proj],
        out_table=near_fire, search_radius="", location="NO_LOCATION",
        angle="NO_ANGLE", closest="CLOSEST",
    )
    near_self = os.path.join(OUT_GDB, f"qc_near_self_seed_{seed}")
    if arcpy.Exists(near_self):
        arcpy.management.Delete(near_self)
    arcpy.analysis.GenerateNearTable(
        in_features=pseudo_fc, near_features=[pseudo_fc],
        out_table=near_self, search_radius="", location="NO_LOCATION",
        angle="NO_ANGLE", closest="CLOSEST",
    )
    min_dist_to_fire = min([r[0] for r in arcpy.da.SearchCursor(near_fire, ["NEAR_DIST"])])
    min_self_spacing = min([r[0] for r in arcpy.da.SearchCursor(near_self, ["NEAR_DIST"])])
    qc_rows.append({
        "seed": seed,
        "pseudo_absence_count": pseudo_count,
        "min_distance_to_fire_m": round(min_dist_to_fire, 1),
        "min_spacing_among_pseudo_m": round(min_self_spacing, 1),
        "passed_distance_to_fire_20km": "Yes" if min_dist_to_fire >= 20000 else "No",
        "passed_spacing_20km":          "Yes" if min_self_spacing >= 20000 else "No",
    })
    if arcpy.Exists(merged_fc):
        arcpy.management.Delete(merged_fc)
    arcpy.management.Merge([fire_proj, pseudo_fc], merged_fc)
    if "UniqueID" not in [f.name for f in arcpy.ListFields(merged_fc)]:
        arcpy.management.AddField(merged_fc, "UniqueID", "LONG")
        arcpy.management.CalculateField(merged_fc, "UniqueID", "!OBJECTID!", "PYTHON3")
    cont_rasters = [[p, n] for p, n, k in PREDICTOR_RASTERS if k == "continuous"]
    cat_rasters  = [[p, n] for p, n, k in PREDICTOR_RASTERS if k == "categorical"]
    if cont_rasters:
        arcpy.sa.ExtractMultiValuesToPoints(merged_fc, cont_rasters, bilinear_interpolate_values="BILINEAR")
    if cat_rasters:
        arcpy.sa.ExtractMultiValuesToPoints(merged_fc, cat_rasters,  bilinear_interpolate_values="NONE")
    if "Longitude" not in [f.name for f in arcpy.ListFields(merged_fc)]:
        arcpy.management.AddField(merged_fc, "Longitude", "DOUBLE")
    if "Latitude" not in [f.name for f in arcpy.ListFields(merged_fc)]:
        arcpy.management.AddField(merged_fc, "Latitude", "DOUBLE")
    arcpy.management.CalculateGeometryAttributes(
        merged_fc, [["Longitude", "POINT_X"], ["Latitude", "POINT_Y"]],
        coordinate_system=PROJECTED_CRS,
    )
    out_csv = os.path.join(TABLES_FOLDER, f"training_seed_{seed}.csv")
    fields_to_export = ["UniqueID", "Longitude", "Latitude", "Status"] + [n for _, n, _ in PREDICTOR_RASTERS]
    rows = list(arcpy.da.SearchCursor(merged_fc, fields_to_export))
    pd.DataFrame(rows, columns=fields_to_export).to_csv(out_csv, index=False)
    print(f"[seed {seed}] wrote {out_csv} ({len(rows)} rows)")
    summary_rows.append({
        "seed": seed,
        "pseudo_absence_count": pseudo_count,
        "wildfire_count": n_fire,
        "total_training_points": int(arcpy.management.GetCount(merged_fc)[0]),
        "csv": out_csv,
    })
summary_csv = os.path.join(TABLES_FOLDER, "pseudo_absence_generation_summary.csv")
qc_csv      = os.path.join(TABLES_FOLDER, "pseudo_absence_qc_table.csv")
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
pd.DataFrame(qc_rows).to_csv(qc_csv, index=False)
print("\n" + "=" * 60)
print("DONE.")
print(f"Generated 10 seed datasets in: {TABLES_FOLDER}")
print(f"Summary table: {summary_csv}")
print(f"QC table:      {qc_csv}")
print("=" * 60)
print("\nQC table contents:")
for r in qc_rows:
    print(f"  seed {r['seed']:>3}: n={r['pseudo_absence_count']:>4} | "
          f"min_dist_to_fire={r['min_distance_to_fire_m']:>7.0f} m | "
          f"min_spacing={r['min_spacing_among_pseudo_m']:>7.0f} m | "
          f"buffer_OK={r['passed_distance_to_fire_20km']} | "
          f"spacing_OK={r['passed_spacing_20km']}")
