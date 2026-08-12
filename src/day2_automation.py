from __future__ import annotations
import csv
import os
import sys
import time
import traceback
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False
def pick_folder(title: str, initialdir: str | None = None) -> str:
    if not HAS_TK:
        return input(f"{title} (paste path): ").strip().strip('"')
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title=title, initialdir=initialdir)
    root.destroy()
    return path
def pick_file(title: str, filetypes: list[tuple[str, str]],
              initialdir: str | None = None) -> str:
    if not HAS_TK:
        return input(f"{title} (paste path): ").strip().strip('"')
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes,
                                      initialdir=initialdir)
    root.destroy()
    return path
def confirm(title: str, msg: str) -> bool:
    if not HAS_TK:
        return input(f"{msg} [y/N]: ").strip().lower().startswith("y")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ans = messagebox.askyesno(title, msg)
    root.destroy()
    return ans
DEFAULT_SRC_RASTERS = r"G:\Deep learning for wildfire susceptibility mapping\BC Rasampled 1500m Rasters"
DEFAULT_PROJ_ROOT   = os.path.expandvars(r"%USERPROFILE%\Documents\wildfire-bc-bilstm-pso")
DEFAULT_NFDB        = r"G:\Deep learning for wildfire susceptibility mapping"
CATEGORICAL_KEYWORDS = ("lulc", "aspect")
THRESHOLDS_HA = (50, 70, 100)
print("=" * 70)
print("DAY 2 AUTOMATION - Wildfire BC BiLSTM-PSO revision (ECOINF-D-26-01275 R1)")
print("=" * 70)
print()
print("This script will open three file-picker dialogs:")
print("  1. The folder containing your 1500 m source rasters (G: drive)")
print("  2. The project root folder (C:\\...\\wildfire-bc-bilstm-pso)")
print("  3. The CWFIS NFDB perimeter shapefile (.shp)")
print()
print("If a dialog appears behind ArcGIS Pro, click the tkinter icon in the taskbar.")
print()
SRC_RASTERS = pick_folder(
    "Step 1 of 3: Select the folder with your 1500 m source rasters",
    initialdir=DEFAULT_SRC_RASTERS if os.path.isdir(DEFAULT_SRC_RASTERS) else None,
)
if not SRC_RASTERS:
    print("CANCELLED at Step 1. Re-run when ready.")
    sys.exit(1)
print(f"  Source rasters folder: {SRC_RASTERS}")
PROJ_ROOT = pick_folder(
    "Step 2 of 3: Select your project root (wildfire-bc-bilstm-pso)",
    initialdir=DEFAULT_PROJ_ROOT if os.path.isdir(DEFAULT_PROJ_ROOT) else None,
)
if not PROJ_ROOT:
    print("CANCELLED at Step 2. Re-run when ready.")
    sys.exit(1)
print(f"  Project root:          {PROJ_ROOT}")
NFDB_SHP = pick_file(
    "Step 3 of 3: Select the CWFIS NFDB perimeter shapefile (.shp)",
    filetypes=[("Shapefile", "*.shp"), ("GeoPackage", "*.gpkg"), ("All files", "*.*")],
    initialdir=DEFAULT_NFDB if os.path.isdir(DEFAULT_NFDB) else None,
)
if not NFDB_SHP:
    print("CANCELLED at Step 3. Re-run when ready.")
    sys.exit(1)
print(f"  NFDB perimeters:       {NFDB_SHP}")
if not confirm("Confirm settings",
               f"Source rasters:\n  {SRC_RASTERS}\n\n"
               f"Project root:\n  {PROJ_ROOT}\n\n"
               f"NFDB perimeters:\n  {NFDB_SHP}\n\n"
               f"Proceed with reprojection + threshold sensitivity + autocorrelation?"):
    print("CANCELLED by user.")
    sys.exit(1)
RASTERS_BC = os.path.join(PROJ_ROOT, "data", "rasters_bcalbers")
VECTORS_BC = os.path.join(PROJ_ROOT, "data", "vectors_bcalbers")
PROCESSED  = os.path.join(PROJ_ROOT, "data", "processed")
TABLES     = os.path.join(PROJ_ROOT, "tables")
FIGS       = os.path.join(PROJ_ROOT, "figs")
LOG_PATH   = os.path.join(TABLES, "T_day2_automation_log.txt")
for d in (RASTERS_BC, VECTORS_BC, PROCESSED, TABLES, FIGS):
    os.makedirs(d, exist_ok=True)
open(LOG_PATH, "w", encoding="utf-8").close()
def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
def clean_name(name: str) -> str:
    out = name
    for suf in ("_resamp_1000.0_resamp_1500.0", "_BC_500m", "_BC", "_500m"):
        out = out.replace(suf, "")
    out = out.replace(".tif", "").lower()
    return out + ".tif"
try:
    import arcpy
except ImportError:
    log("ERROR: arcpy is not available. Run this script from ArcGIS Pro's Python window.")
    sys.exit(2)
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")
TARGET_CRS = arcpy.SpatialReference(3005)
log(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Source rasters: {SRC_RASTERS}")
log(f"Project root:   {PROJ_ROOT}")
log(f"NFDB shapefile: {NFDB_SHP}")
log("")
def phase_c_rasters() -> int:
    log("=" * 70)
    log("PHASE C - Reproject rasters to EPSG:3005 BC Albers (Comment 9)")
    log("=" * 70)
    tifs = sorted(
        f for f in os.listdir(SRC_RASTERS)
        if f.lower().endswith(".tif")
           and "resamp_1500" in f.lower()
           and not any(s in f.lower() for s in (".aux", ".ovr", ".vat"))
    )
    log(f"  Found {len(tifs)} 1500 m raster(s) matching '*resamp_1500*.tif'")
    rows = []
    for i, tif in enumerate(tifs, 1):
        src = os.path.join(SRC_RASTERS, tif)
        out_name = clean_name(tif)
        dst = os.path.join(RASTERS_BC, out_name)
        is_cat = any(k in tif.lower() for k in CATEGORICAL_KEYWORDS)
        resamp = "NEAREST" if is_cat else "BILINEAR"
        try:
            t0 = time.time()
            arcpy.management.ProjectRaster(
                in_raster=src,
                out_raster=dst,
                out_coor_system=TARGET_CRS,
                resampling_type=resamp,
                cell_size="1500 1500",
            )
            dt = time.time() - t0
            log(f"  [{i:>2}/{len(tifs)}] {tif:55s} -> {out_name:25s}  ({resamp}, {dt:.1f}s)")
            rows.append((tif, out_name, resamp, "EPSG:26912 -> EPSG:3005", round(dt, 1)))
        except Exception as e:
            log(f"  FAILED: {tif}: {e}")
            rows.append((tif, out_name, resamp, "FAILED", str(e)[:80]))
    csv_path = os.path.join(TABLES, "T_reprojection_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_name", "output_name", "resampling", "crs_change", "seconds_or_error"])
        w.writerows(rows)
    log(f"  Wrote: {csv_path}")
    return len(rows)
def phase_c1_nfdb() -> str | None:
    log("")
    log("=" * 70)
    log("PHASE C.1 - Reproject CWFIS NFDB perimeters")
    log("=" * 70)
    out_shp = os.path.join(VECTORS_BC, "NFDB_poly_large.shp")
    try:
        arcpy.management.Project(
            in_dataset=NFDB_SHP,
            out_dataset=out_shp,
            out_coor_system=TARGET_CRS,
        )
        n = int(arcpy.management.GetCount(out_shp)[0])
        log(f"  {n:,} fires reprojected -> {out_shp}")
        return out_shp
    except Exception as e:
        log(f"  FAILED: {e}")
        return None
def phase_d_threshold(nfdb_bc: str) -> dict:
    log("")
    log("=" * 70)
    log("PHASE D - 70 ha threshold sensitivity (Comment 11)")
    log("=" * 70)
    fields = [f.name for f in arcpy.ListFields(nfdb_bc)]
    if "AREA_HA" not in fields:
        arcpy.management.AddField(nfdb_bc, "AREA_HA", "DOUBLE")
        arcpy.management.CalculateGeometryAttributes(
            in_features=nfdb_bc,
            geometry_property=[["AREA_HA", "AREA"]],
            area_unit="HECTARES",
        )
        log("  Added and calculated AREA_HA")
        fields.append("AREA_HA")
    year_col = next((c for c in ("YEAR", "YEAR_", "FIRE_YEAR", "YEAR_FIRE") if c in fields), None)
    parts = []
    if year_col:
        parts.append(f"\"{year_col}\" >= 2000 AND \"{year_col}\" <= 2024")
    if "LATITUDE" in fields and "LONGITUDE" in fields:
        parts.append("\"LATITUDE\" >= 48.30 AND \"LATITUDE\" <= 60.00")
        parts.append("\"LONGITUDE\" >= -139.06 AND \"LONGITUDE\" <= -114.03")
        log("  BC filter: lat 48.30-60.00, lon -139.06 to -114.03 (catches BC + PC)")
    elif "SRC_AGENCY" in fields:
        parts.append("(\"SRC_AGENCY\" = 'BC' OR \"SRC_AGENCY\" = 'PC')")
        log("  BC filter: SRC_AGENCY IN ('BC','PC') -- no LAT/LON fields available")
    base = " AND ".join(parts)
    log(f"  Base filter: {base if base else '(none)'}")
    counts = {}
    for thr in THRESHOLDS_HA:
        where = f"({base}) AND \"AREA_HA\" >= {thr}" if base else f"\"AREA_HA\" >= {thr}"
        out = os.path.join(PROCESSED, f"fires_geq_{thr}ha.shp")
        try:
            arcpy.analysis.Select(nfdb_bc, out, where)
            n = int(arcpy.management.GetCount(out)[0])
            counts[thr] = n
            log(f"  >= {thr:>3} ha -> {n:>6,} fires  ({out})")
        except Exception as e:
            log(f"  FAILED at {thr} ha: {e}")
            counts[thr] = 0
    csv_path = os.path.join(TABLES, "T_sensitivity_threshold.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold_ha", "n_fires", "is_baseline"])
        for thr in THRESHOLDS_HA:
            w.writerow([thr, counts.get(thr, 0), "TRUE" if thr == 70 else "FALSE"])
    log(f"  Wrote: {csv_path}")
    return counts
def _parse_moran_messages(msgs: str) -> tuple[float | None, float | None, float | None]:
    I = z = p = None
    for line in msgs.splitlines():
        s = line.strip()
        if "Moran's Index" in s:
            try: I = float(s.split(":")[-1].strip())
            except: pass
        elif s.lower().startswith("z-score"):
            try: z = float(s.split(":")[-1].strip())
            except: pass
        elif s.lower().startswith("p-value"):
            try: p = float(s.split(":")[-1].strip())
            except: pass
    return I, z, p
def phase_e_autocorr() -> list:
    log("")
    log("=" * 70)
    log("PHASE E - Spatial autocorrelation: Moran's I + LISA (Comment 10)")
    log("=" * 70)
    fires_70 = os.path.join(PROCESSED, "fires_geq_70ha.shp")
    if not os.path.exists(fires_70):
        log(f"  ABORT: {fires_70} not found. Run Phase D first.")
        return []
    centroids = os.path.join(PROCESSED, "fire_centroids_70ha.shp")
    arcpy.management.FeatureToPoint(fires_70, centroids, "INSIDE")
    n = int(arcpy.management.GetCount(centroids)[0])
    log(f"  Fire centroids: {n:,} points")
    rasters = sorted(
        os.path.join(RASTERS_BC, f) for f in os.listdir(RASTERS_BC)
        if f.lower().endswith(".tif") and "mask" not in f.lower()
    )
    extract_list = [[r, os.path.splitext(os.path.basename(r))[0]] for r in rasters]
    log(f"  Sampling {len(extract_list)} predictor rasters at centroids ...")
    arcpy.sa.ExtractMultiValuesToPoints(centroids, extract_list, "NONE")
    skip = {"FID", "Shape", "ORIG_FID", "Id", "AREA_HA", "SIZE_HA", "POLY_HA", "OBJECTID"}
    sampled = [
        f.name for f in arcpy.ListFields(centroids)
        if f.type in ("Double", "Single", "Integer", "SmallInteger")
           and f.name not in skip
    ]
    csv_path = os.path.join(TABLES, "T_autocorrelation_global.csv")
    log(f"  Computing Moran's I for {len(sampled)} predictor field(s) ...")
    moran_rows = []
    for fld in sampled:
        try:
            res = arcpy.stats.SpatialAutocorrelation(
                Input_Feature_Class=centroids,
                Input_Field=fld,
                Generate_Report="NO_REPORT",
                Conceptualization_of_Spatial_Relationships="K_NEAREST_NEIGHBORS",
                Distance_Method="EUCLIDEAN_DISTANCE",
                Standardization="ROW",
                Number_of_Neighbors=8,
            )
            I, z, p = _parse_moran_messages(res.getMessages())
            interp = ""
            if I is not None:
                interp = ("strong cluster" if I > 0.4 else
                          "moderate cluster" if I > 0.15 else
                          "weak cluster" if I > 0.05 else
                          "random" if abs(I) <= 0.05 else
                          "dispersed")
            moran_rows.append({"variable": fld, "moran_I": I, "moran_z": z,
                               "moran_p": p, "interpretation": interp})
            log(f"    {fld:30s}  I={I}  z={z}  p={p}  [{interp}]")
        except Exception as e:
            log(f"    {fld:30s}  FAILED: {str(e)[:80]}")
            moran_rows.append({"variable": fld, "moran_I": None, "moran_z": None,
                               "moran_p": None, "interpretation": f"ERROR: {e}"})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variable", "moran_I", "moran_z", "moran_p", "interpretation"])
        w.writeheader()
        w.writerows(moran_rows)
    log(f"  Wrote: {csv_path}")
    valid = [r for r in moran_rows if isinstance(r["moran_I"], (int, float))]
    valid.sort(key=lambda r: abs(r["moran_I"]), reverse=True)
    top4 = [r["variable"] for r in valid[:4]]
    log(f"  Top 4 predictors by |Moran's I|: {top4}")
    for fld in top4:
        out_lisa = os.path.join(PROCESSED, f"lisa_{fld}.shp")
        try:
            arcpy.stats.ClustersOutliers(
                Input_Feature_Class=centroids,
                Input_Field=fld,
                Output_Feature_Class=out_lisa,
                Conceptualization_of_Spatial_Relationships="K_NEAREST_NEIGHBORS",
                Distance_Method="EUCLIDEAN_DISTANCE",
                Standardization="ROW",
                Number_of_Neighbors=8,
                Number_of_Permutations=999,
            )
            log(f"    LISA {fld:30s} -> {out_lisa}")
        except Exception as e:
            log(f"    LISA {fld:30s} FAILED: {str(e)[:80]}")
    return top4
def main():
    try:
        n_rasters = phase_c_rasters()
        nfdb_bc = phase_c1_nfdb()
        if nfdb_bc:
            counts = phase_d_threshold(nfdb_bc)
        top4 = phase_e_autocorr()
        log("")
        log("=" * 70)
        log("DAY 2 ARCPY AUTOMATION COMPLETE")
        log("=" * 70)
        log("Deliverables for git commit:")
        log(f"  data/rasters_bcalbers/*.tif       ({n_rasters} rasters)")
        log(f"  data/vectors_bcalbers/NFDB_poly_large.shp")
        log(f"  data/processed/fires_geq_{{50,70,100}}ha.shp")
        log(f"  data/processed/fire_centroids_70ha.shp")
        log(f"  data/processed/lisa_*.shp")
        log(f"  tables/T_reprojection_log.csv")
        log(f"  tables/T_sensitivity_threshold.csv")
        log(f"  tables/T_autocorrelation_global.csv")
        log(f"  tables/T_day2_automation_log.txt")
        log("")
        log("PHASE F (manual) - ENSO/PDO ingestion (Comment 2):")
        log("  In Anaconda Prompt:")
        log("    conda activate wildfire")
        log(f"    cd {PROJ_ROOT}")
        log('    Invoke-WebRequest -Uri "https://psl.noaa.gov/enso/mei/data/meiv2.data" -OutFile data\\raw\\meiv2.txt')
        log('    Invoke-WebRequest -Uri "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat" -OutFile data\\raw\\pdo.csv')
        log("    jupyter lab   # then run notebooks/03_enso_pdo_ingest.ipynb")
        log("")
        log("Then commit and push (PowerShell):")
        log(f"  cd {PROJ_ROOT}")
        log("  git add data\\processed\\*.shp data\\processed\\*.dbf data\\processed\\*.shx data\\processed\\*.prj")
        log("  git add tables\\T_*.csv")
        log('  git commit -m "Day 2 ArcPy automation: reprojection + threshold sensitivity + spatial autocorrelation (Comments 7, 9, 10, 11)"')
        log("  git push")
        if HAS_TK:
            messagebox.showinfo(
                "Day 2 Automation Complete",
                f"All phases finished.\n\n"
                f"Rasters reprojected: {n_rasters}\n"
                f"Thresholds processed: {len(THRESHOLDS_HA)}\n"
                f"Top 4 LISA predictors: {', '.join(top4) if top4 else '(none)'}\n\n"
                f"Log: {LOG_PATH}"
            )
    except Exception as e:
        log(f"\nUNHANDLED ERROR: {e}")
        log(traceback.format_exc())
        if HAS_TK:
            messagebox.showerror("Day 2 Automation - Error",
                                 f"An error occurred:\n\n{e}\n\nSee log:\n{LOG_PATH}")
main()
