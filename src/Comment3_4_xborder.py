import os
import sys
import subprocess
import arcpy
REPO   = r"C:\Users\saadz\Documents\wildfire-bc-bilstm-pso\revision_c8c11"
INPUT  = os.path.join(REPO, "01_Input_Data")
OUTPUT = os.path.join(REPO, "02_GIS_Output")
BC_BOUNDARY = os.path.join(INPUT, "bc_boundary.shp")
US_STATES   = os.path.join(INPUT, "borders", "US_states", "cb_2023_us_state_5m.shp")
BEC_ZONES   = os.path.join(INPUT, "BEC_zones", "BEC_BIOGEOCLIMATIC_POLY",
                            "BEC_POLY_polygon.shp")
US_COMBINED  = os.path.join(OUTPUT, "us_combined_WA_ID_MT.shp")
LINES_ALL    = os.path.join(OUTPUT, "bc_us_lines_all.shp")
BORDER_OUT   = os.path.join(OUTPUT, "bc_us_border.shp")
BUFFER_50KM  = os.path.join(OUTPUT, "border_buffer_50km.shp")
BC_STRIP     = os.path.join(OUTPUT, "bc_strip.shp")
US_STRIP     = os.path.join(OUTPUT, "us_strip.shp")
BC_PROJ = os.path.join(OUTPUT, "_bc_boundary_3005.shp")
US_PROJ = os.path.join(OUTPUT, "_us_states_3005.shp")
BC_ALBERS    = arcpy.SpatialReference(3005)
STATE_QUERY  = "STUSPS IN ('WA', 'ID', 'MT')"
SEARCH_DIST  = "100 Meters"
BUFFER_DIST  = "50 Kilometers"
arcpy.env.overwriteOutput = True
arcpy.env.outputCoordinateSystem = BC_ALBERS
def _try_install_tqdm():
    try:
        print("tqdm not found -- attempting silent install ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "tqdm",
             "--quiet", "--disable-pip-version-check"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    if _try_install_tqdm():
        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False
    else:
        HAS_TQDM = False
if not HAS_TQDM:
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total       = kwargs.get("total", 0)
            self.n           = 0
            self.desc        = kwargs.get("desc", "")
            self.unit        = kwargs.get("unit", "")
            self.last_postfix = ""
            print("[{0}] starting (total={1} {2})".format(self.desc, self.total, self.unit))
        def update(self, n=1):
            self.n += n
            pct = (100.0 * self.n / self.total) if self.total else 0
            print("[{0}] {1}/{2} ({3:.0f}%)   {4}".format(
                self.desc, self.n, self.total, pct, self.last_postfix))
        def set_postfix_str(self, s):
            self.last_postfix = s
        @staticmethod
        def write(msg):
            print(msg)
        def close(self):
            print("[{0}] finished.".format(self.desc))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
def msg(s):
    if HAS_TQDM:
        tqdm.write(s)
    else:
        print(s)
def require(path, label):
    if not arcpy.Exists(path):
        raise FileNotFoundError(
            "{0} not found:\n    {1}\n"
            "Edit the CONFIG section at the top of the script.".format(label, path))
def project_to_albers(src, dst, label):
    require(src, label)
    msg("  Projecting {0} -> BC Albers ...".format(label))
    arcpy.management.Project(src, dst, BC_ALBERS)
    return dst
def area_km2(fc):
    n = int(arcpy.management.GetCount(fc)[0])
    total = 0.0
    inner = tqdm(total=n, desc="  area " + os.path.basename(fc),
                 unit="poly", leave=False)
    for r in arcpy.da.SearchCursor(fc, ["SHAPE@AREA"]):
        total += r[0]
        inner.update(1)
    inner.close()
    return total / 1e6
def line_length_km(fc):
    n = int(arcpy.management.GetCount(fc)[0])
    total = 0.0
    inner = tqdm(total=n, desc="  length " + os.path.basename(fc),
                 unit="seg", leave=False)
    for r in arcpy.da.SearchCursor(fc, ["SHAPE@LENGTH"]):
        total += r[0]
        inner.update(1)
    inner.close()
    return total / 1000.0
def stage_add_layers(pbar):
    pbar.set_postfix_str("B2: add layers")
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
    except OSError:
        msg("Not running inside ArcGIS Pro -- skipping map/layer step (B2).")
        pbar.update(1)
        return None
    m = aprx.activeMap or aprx.listMaps()[0]
    m.spatialReference = BC_ALBERS
    for fc, label in [(BC_BOUNDARY, "BC boundary"),
                      (US_STATES,   "US states"),
                      (BEC_ZONES,   "BEC zones")]:
        if arcpy.Exists(fc):
            m.addDataFromPath(fc)
            msg("  Added layer: {0}".format(label))
        else:
            msg("  (skip) {0} not found: {1}".format(label, fc))
    aprx.save()
    pbar.update(1)
    return aprx
def stage_project_inputs(pbar):
    os.makedirs(OUTPUT, exist_ok=True)
    pbar.set_postfix_str("B3.0: project BC boundary")
    project_to_albers(BC_BOUNDARY, BC_PROJ, "BC boundary")
    pbar.update(1)
    pbar.set_postfix_str("B3.0: project US states")
    project_to_albers(US_STATES, US_PROJ, "US states")
    pbar.update(1)
def stage_filter_dissolve(pbar):
    pbar.set_postfix_str("B3.1: filter WA/ID/MT + dissolve")
    msg("\nSelecting WA / ID / MT and dissolving to a single polygon ...")
    arcpy.management.MakeFeatureLayer(US_PROJ, "us_sel", STATE_QUERY)
    count = int(arcpy.management.GetCount("us_sel")[0])
    if count == 0:
        raise RuntimeError(
            "Definition query returned 0 features. "
            "Check the state field/code (expected 'STUSPS').")
    msg("  {0} state features matched.".format(count))
    arcpy.management.Dissolve("us_sel", US_COMBINED)
    pbar.update(1)
def stage_feature_to_line(pbar):
    pbar.set_postfix_str("B3.3: feature to line")
    msg("Running Feature To Line on BC boundary + combined US polygon ...")
    arcpy.management.FeatureToLine(
        [BC_PROJ, US_COMBINED], LINES_ALL,
        cluster_tolerance="", attributes="ATTRIBUTES")
    pbar.update(1)
def stage_select_border(pbar):
    pbar.set_postfix_str("B3.4: dual proximity selection")
    msg("Selecting only the shared BC-US border segments ...")
    arcpy.management.MakeFeatureLayer(LINES_ALL, "lines_lyr")
    arcpy.management.SelectLayerByLocation(
        "lines_lyr", "WITHIN_A_DISTANCE", US_COMBINED,
        SEARCH_DIST, "NEW_SELECTION")
    arcpy.management.SelectLayerByLocation(
        "lines_lyr", "WITHIN_A_DISTANCE", BC_PROJ,
        SEARCH_DIST, "SUBSET_SELECTION")
    n_sel = int(arcpy.management.GetCount("lines_lyr")[0])
    if n_sel == 0:
        raise RuntimeError(
            "No coincident border segments found. "
            "Try a larger SEARCH_DIST (e.g. '250 Meters').")
    msg("  {0} border segment(s) selected.".format(n_sel))
    arcpy.conversion.ExportFeatures("lines_lyr", BORDER_OUT)
    msg("  Exported -> {0}".format(BORDER_OUT))
    pbar.update(1)
def stage_buffer(pbar):
    pbar.set_postfix_str("B4: 50-km symmetric buffer")
    msg("\nBuffering bc_us_border by {0} (symmetric / FULL) ...".format(BUFFER_DIST))
    arcpy.analysis.Buffer(
        BORDER_OUT, BUFFER_50KM,
        buffer_distance_or_field=BUFFER_DIST,
        line_side="FULL", line_end_type="FLAT", dissolve_option="ALL")
    pbar.update(1)
def stage_intersect_bc(pbar):
    pbar.set_postfix_str("B4: intersect -> bc_strip")
    msg("Intersecting buffer with BC boundary -> bc_strip ...")
    arcpy.analysis.Intersect([BUFFER_50KM, BC_PROJ], BC_STRIP, "ONLY_FID")
    pbar.update(1)
def stage_intersect_us(pbar):
    pbar.set_postfix_str("B4: intersect -> us_strip")
    msg("Intersecting buffer with us_combined -> us_strip ...")
    arcpy.analysis.Intersect([BUFFER_50KM, US_COMBINED], US_STRIP, "ONLY_FID")
    pbar.update(1)
def stage_metrics(pbar):
    pbar.set_postfix_str("B5: compute metrics")
    km   = line_length_km(BORDER_OUT)
    bc_a = area_km2(BC_STRIP)
    us_a = area_km2(US_STRIP)
    msg("\n" + "=" * 60)
    msg("  RESULTS")
    msg("=" * 60)
    msg("  bc_us_border length: {0:>10,.1f} km   (expect ~1,060 km)".format(km))
    msg("  bc_strip area      : {0:>10,.0f} km^2 (expect ~45,000 km^2)".format(bc_a))
    msg("  us_strip area      : {0:>10,.0f} km^2 (expect ~45,000 km^2)".format(us_a))
    msg("=" * 60)
    if not (800 <= km <= 1300):
        msg("  WARN: border length out of expected range.")
    if not (35000 <= bc_a <= 55000):
        msg("  WARN: BC strip area out of expected range.")
    if not (35000 <= us_a <= 55000):
        msg("  WARN: US strip area out of expected range.")
    pbar.update(1)
def stage_cleanup(pbar):
    pbar.set_postfix_str("cleanup")
    for tmp in (BC_PROJ, US_PROJ):
        if arcpy.Exists(tmp):
            arcpy.management.Delete(tmp)
            msg("  Deleted intermediate: {0}".format(os.path.basename(tmp)))
    pbar.update(1)
def stage_add_results_to_map(pbar):
    pbar.set_postfix_str("add outputs to map")
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        active_map = aprx.activeMap or aprx.listMaps()[0]
        for fc, label in [(BORDER_OUT, "bc_us_border"),
                          (BC_STRIP,   "bc_strip"),
                          (US_STRIP,   "us_strip")]:
            active_map.addDataFromPath(fc)
            msg("  Added to active map: {0}".format(label))
        aprx.save()
    except OSError:
        pass
    pbar.update(1)
TOTAL_STAGES = 11
def main():
    print("=" * 60)
    print("Comment3_4_xborder: BC-CONUS border + 50-km strips")
    print("tqdm available: {0}".format("YES" if HAS_TQDM else "NO (fallback mode)"))
    print("=" * 60)
    bar_fmt = ("{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
               " {postfix}")
    pbar = tqdm(total=TOTAL_STAGES, desc="Phase B", unit="stage",
                bar_format=bar_fmt if HAS_TQDM else None)
    try:
        stage_add_layers(pbar)
        stage_project_inputs(pbar)
        stage_filter_dissolve(pbar)
        stage_feature_to_line(pbar)
        stage_select_border(pbar)
        stage_buffer(pbar)
        stage_intersect_bc(pbar)
        stage_intersect_us(pbar)
        stage_metrics(pbar)
        stage_cleanup(pbar)
        stage_add_results_to_map(pbar)
    finally:
        pbar.close()
    print("\nPhase B (B2+B3+B4+B5) complete.\n"
          "Next: build Fig_S5 in the GUI (BEC fire-ecology context map), "
          "then export PDF + PNG @ 300 dpi to "
          "figs/Fig_S5_BC_fireecology_context.pdf")
if __name__ == "__main__":
    main()
