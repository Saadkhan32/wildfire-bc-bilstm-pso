import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
def pick_folder(title, initialdir=None):
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory(title=title, initialdir=initialdir)
    r.destroy(); return p
print("=" * 60)
print("STEP 1 / 5: Set up folder structure inside your GitHub repo")
print("=" * 60)
print()
print("A folder picker will open. Pick your GitHub repo root, i.e.")
print(os.path.expandvars("  %USERPROFILE%\\Documents\\wildfire-bc-bilstm-pso"))
print()
repo = pick_folder(
    "STEP 1: pick your GitHub repo root",
    initialdir=os.path.expandvars(r"%USERPROFILE%\Documents\wildfire-bc-bilstm-pso"),
)
if not repo:
    print("CANCELLED."); sys.exit(1)
base = os.path.join(repo, "revision_c8c11")
subdirs = [
    "01_Input_Data",
    "01_Input_Data/rasters",
    "02_GIS_Output",
    "03_Training_Tables",
    "04_Model_Scripts",
    "05_Model_Results",
    "06_Final_Tables",
]
print(f"\nCreating folder tree under: {base}")
for s in subdirs:
    d = os.path.join(base, s)
    os.makedirs(d, exist_ok=True)
    print(f"  OK   {d}")
cfg = os.path.join(base, "revision_config.txt")
with open(cfg, "w", encoding="utf-8") as f:
    f.write(base + "\n")
print(f"\n[saved] {cfg}")
print()
print("=" * 60)
print("DONE.")
print("=" * 60)
print()
print("Manual copy-in before STEP 2:")
print(f"  - BC boundary shapefile         -> {base}\\01_Input_Data\\")
print(f"  - Your wildfire layer           -> {base}\\01_Input_Data\\")
print(f"    (must have YEAR + AREA_HA columns)")
print(f"  - 17 predictor rasters          -> {base}\\01_Input_Data\\rasters\\")
print(f"  - BiLSTM PSO FE.py              -> {base}\\04_Model_Scripts\\")
print()
print("Required raster filenames (rename to match these exactly):")
print("  Aspect.tif, Slope.tif, Elevation.tif, TWI.tif,")
print("  Profile_Curvature.tif, Plan_Curvature.tif, NDVI.tif, LULC.tif,")
print("  Max_Temperature.tif, Precipitation.tif, WS.tif,")
print("  Relative_Humidity.tif,   <- NOT Specific_Humidity")
print("  AET.tif, DSI.tif, Soil_Moisture.tif,")
print("  Distance_roads.tif, Distance_rivers.tif, Distance_households.tif")
print()
print("Then run STEP 2 in ArcGIS Pro Python window.")
try:
    a = tk.Tk(); a.withdraw(); a.attributes("-topmost", True)
    messagebox.showinfo(
        "STEP 1 complete",
        f"Folder tree:\n{base}\n\nPlace input files, then run STEP 2.",
    )
    a.destroy()
except Exception:
    pass
