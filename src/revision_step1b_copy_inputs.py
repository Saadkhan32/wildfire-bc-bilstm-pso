import os
import sys
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
REQUIRED_RASTERS = [
    "Aspect.tif", "Slope.tif", "Elevation.tif", "TWI.tif",
    "Profile_Curvature.tif", "Plan_Curvature.tif", "NDVI.tif", "LULC.tif",
    "Max_Temperature.tif", "Precipitation.tif", "WS.tif",
    "Relative_Humidity.tif", "AET.tif", "DSI.tif", "Soil_Moisture.tif",
    "Distance_roads.tif", "Distance_rivers.tif", "Distance_households.tif",
]
SHP_SIBLINGS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn",
                ".sbx", ".qix", ".aih", ".ain", ".shp.xml")
def pick_file(t, ft, init=None):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title=t, filetypes=ft, initialdir=init)
    r.destroy(); return p
def pick_folder(t, init=None):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=t, initialdir=init)
    r.destroy(); return p
def copy_shapefile(src_shp, dst_folder, dst_name=None):
    base = os.path.splitext(src_shp)[0]
    src_dir = os.path.dirname(src_shp)
    src_stem = os.path.basename(base)
    dst_stem = os.path.splitext(dst_name)[0] if dst_name else src_stem
    n_copied = 0
    for ext in SHP_SIBLINGS:
        s = base + ext
        if os.path.exists(s):
            d = os.path.join(dst_folder, dst_stem + ext)
            shutil.copy2(s, d)
            n_copied += 1
    sxml = base + ".shp.xml"
    if os.path.exists(sxml):
        d = os.path.join(dst_folder, dst_stem + ".shp.xml")
        shutil.copy2(sxml, d)
        n_copied += 1
    return n_copied
print("=" * 60)
print("STEP 1b: Copy inputs into revision_c8c11 via Tkinter")
print("=" * 60)
print("\nDialog 1: pick your repo root (wildfire-bc-bilstm-pso)")
repo = pick_folder(
    "STEP 1b dialog 1: pick repo root",
    init=r"C:\Users\saadz\Documents\wildfire-bc-bilstm-pso",
)
if not repo:
    print("CANCELLED."); sys.exit(1)
base = os.path.join(repo, "revision_c8c11")
if not os.path.isdir(base):
    print(f"ERROR: {base} not found. Run STEP 1 first.")
    sys.exit(2)
INPUT_DIR    = os.path.join(base, "01_Input_Data")
RASTER_DIR   = os.path.join(INPUT_DIR, "rasters")
SCRIPT_DIR   = os.path.join(base, "04_Model_Scripts")
for d in (INPUT_DIR, RASTER_DIR, SCRIPT_DIR):
    os.makedirs(d, exist_ok=True)
print("\nDialog 2: pick your BC boundary .shp file")
bc_shp = pick_file("STEP 1b dialog 2: BC boundary .shp",
                    [("Shapefile", "*.shp")])
if not bc_shp:
    print("CANCELLED."); sys.exit(1)
n = copy_shapefile(bc_shp, INPUT_DIR, "bc_boundary.shp")
print(f"  Copied {n} files for bc_boundary")
print("\nDialog 3: pick your wildfire layer .shp")
print("  (NFDB polygons or centroids; must have YEAR + AREA_HA columns)")
fire_shp = pick_file("STEP 1b dialog 3: wildfire layer .shp",
                      [("Shapefile", "*.shp")])
if not fire_shp:
    print("CANCELLED."); sys.exit(1)
n = copy_shapefile(fire_shp, INPUT_DIR, "wildfire_layer.shp")
print(f"  Copied {n} files for wildfire_layer")
print("\nDialog 4: pick the folder containing your 18 predictor TIFFs")
src_rasters_dir = pick_folder("STEP 1b dialog 4: rasters source folder")
if not src_rasters_dir:
    print("CANCELLED."); sys.exit(1)
src_files_lc = {}
for fn in os.listdir(src_rasters_dir):
    if fn.lower().endswith(".tif"):
        src_files_lc[fn.lower()] = os.path.join(src_rasters_dir, fn)
matched = {}
to_pick = []
print("\nAuto-matching required rasters:")
for req in REQUIRED_RASTERS:
    key = req.lower()
    if key in src_files_lc:
        matched[req] = src_files_lc[key]
        print(f"  FOUND   {req}")
    else:
        to_pick.append(req)
        print(f"  MISSING {req}  (will ask)")
for req in to_pick:
    print(f"\nDialog: pick the raster that should become '{req}'")
    p = pick_file(
        f"Pick the .tif to use as '{req}'",
        [("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")],
        init=src_rasters_dir,
    )
    if not p:
        print(f"  SKIPPED {req}"); continue
    matched[req] = p
print("\nCopying rasters:")
copied = 0
for req, src in matched.items():
    dst = os.path.join(RASTER_DIR, req)
    shutil.copy2(src, dst)
    src_base = os.path.splitext(src)[0]
    for ext in (".tfw", ".tif.aux.xml", ".tif.ovr", ".tif.xml"):
        side = src_base + ext if not ext.startswith(".tif") else src + ext.replace(".tif", "")
    for side_ext in (".tfw", ".tif.aux.xml", ".tif.ovr"):
        side_src = (src[:-4] + side_ext) if side_ext.startswith(".") and not side_ext.startswith(".tif") \
                    else src + side_ext.replace(".tif", "")
    for ext_pair in [(".tfw", ".tfw"), (".tif.aux.xml", ".tif.aux.xml")]:
        s_ext, d_ext = ext_pair
        s_path = src[:-4] + s_ext if s_ext.endswith(".tfw") else src + ".aux.xml"
        d_path = dst[:-4] + d_ext if d_ext.endswith(".tfw") else dst + ".aux.xml"
        if os.path.exists(s_path):
            shutil.copy2(s_path, d_path)
    print(f"  OK   {os.path.basename(src)}  ->  {req}")
    copied += 1
print(f"\nTotal rasters copied: {copied} / {len(REQUIRED_RASTERS)}")
print("\nDialog 5: pick BiLSTM PSO FE.py")
pso_script = pick_file(
    "STEP 1b dialog 5: pick BiLSTM PSO FE.py",
    [("Python", "*.py")],
)
if pso_script:
    dst = os.path.join(SCRIPT_DIR, os.path.basename(pso_script))
    shutil.copy2(pso_script, dst)
    print(f"  Copied PSO script -> {dst}")
else:
    print("  SKIPPED PSO script.")
print("\n" + "=" * 60)
print("Inventory:")
print("=" * 60)
print(f"  BC boundary       : {'OK' if os.path.exists(os.path.join(INPUT_DIR, 'bc_boundary.shp')) else 'MISSING'}")
print(f"  Wildfire layer    : {'OK' if os.path.exists(os.path.join(INPUT_DIR, 'wildfire_layer.shp')) else 'MISSING'}")
n_ok = sum(1 for r in REQUIRED_RASTERS
            if os.path.exists(os.path.join(RASTER_DIR, r)))
print(f"  Predictor rasters : {n_ok} / {len(REQUIRED_RASTERS)}")
for r in REQUIRED_RASTERS:
    p = os.path.join(RASTER_DIR, r)
    print(f"    {'OK ' if os.path.exists(p) else 'MISS'} {r}")
print(f"  PSO script        : {'OK' if os.listdir(SCRIPT_DIR) else 'MISSING'}")
print("\n" + "=" * 60)
all_ok = (os.path.exists(os.path.join(INPUT_DIR, "bc_boundary.shp"))
          and os.path.exists(os.path.join(INPUT_DIR, "wildfire_layer.shp"))
          and n_ok == len(REQUIRED_RASTERS)
          and len(os.listdir(SCRIPT_DIR)) > 0)
if all_ok:
    print("STEP 1b complete. All inputs in place. Next: STEP 2 in ArcGIS Pro.")
else:
    print("STEP 1b finished with MISSING items. Re-run to add them, or copy manually.")
try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    msg = (f"BC boundary: {'OK' if os.path.exists(os.path.join(INPUT_DIR, 'bc_boundary.shp')) else 'MISSING'}\n"
           f"Wildfire layer: {'OK' if os.path.exists(os.path.join(INPUT_DIR, 'wildfire_layer.shp')) else 'MISSING'}\n"
           f"Rasters: {n_ok} / {len(REQUIRED_RASTERS)}\n"
           f"PSO script: {'OK' if len(os.listdir(SCRIPT_DIR)) > 0 else 'MISSING'}\n\n"
           f"{'Run STEP 2 next.' if all_ok else 'Re-run to add missing items.'}")
    messagebox.showinfo("STEP 1b complete", msg)
    a.destroy()
except Exception:
    pass
