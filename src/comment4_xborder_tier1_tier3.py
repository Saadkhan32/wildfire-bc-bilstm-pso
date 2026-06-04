# -*- coding: utf-8 -*-
"""
comment4_xborder_tier1_tier3.py
================================
Cross-border consistency (Tier 1) + no-discontinuity (Tier 3) test.

Tier 2 (occurrence-density validation) is intentionally NOT included here -
the 50 km border band contains too few large fires (131 BC + 152 US, 2000-2023)
to power a Spearman ranking on a sparse zero-heavy density surface. The
manuscript covers the U.S. side performance question by citing FSim/WRC's
published validation literature (Scott et al. 2013; Finney et al. 2011;
Short et al. 2020) and discloses the sample-size rationale in the limitations.

Pipeline:
  1. Build a 1.5 km template grid in EPSG:3005 covering bc_strip + us_strip.
  2. Reproject both rasters onto the template (identical snap, no off-by-one).
     - BC_susceptibility_BiLSTM_PSO.tif (continuous probability)
     - BP_WA_ID_MT.tif (FSim burn probability)
  3. Mask each raster by its strip polygon.
  4. Hex-tile aggregation (50 km diameter, pointy-top) over the union of strips;
     keep only hexes that straddle the border (have BOTH strips' cells inside).
  5. Tier 1: Spearman rho (pooled + per US state) + Cohen's kappa on 5-class
            manual binning (0.2/0.4/0.6/0.8).
  6. Tier 3: Mann-Whitney U on the 5 km near-border band (BC near-band cells
            vs US near-band cells).
  7. Write Fig_S6b (2-panel BC | US maps on shared color scale), JSON stats,
     and per-hex CSV.

Outputs:
  revision_c8c11/03_Figures/Fig_S6b_xborder_consistency.pdf (+ .png)
  revision_c8c11/06_Final_Tables/T_C4_xborder_tier1_tier3.json
  revision_c8c11/06_Final_Tables/T_C4_hex_tile_values.csv

No TensorFlow / no GPU - pure rasterio + geopandas + scipy + matplotlib.
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# tqdm with auto-install
try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm",
                            "--quiet", "--disable-pip-version-check"])
    from tqdm import tqdm

import rasterio
from rasterio.transform import from_bounds, rowcol
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask

import geopandas as gpd
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import cohen_kappa_score

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------------- Paths ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO       = os.path.dirname(SCRIPT_DIR)

BC_SUS_TIF = os.path.join(REPO, "revision_c8c11", "02_GIS_Output",
                          "BC_susceptibility_BiLSTM_PSO.tif")
BP_TIF     = os.path.join(REPO, "revision_c8c11", "01_Input_Data",
                          "borders", "US_burn_probability", "BP_WA_ID_MT.tif")
BC_STRIP   = os.path.join(REPO, "revision_c8c11", "02_GIS_Output", "bc_strip.shp")
US_STRIP   = os.path.join(REPO, "revision_c8c11", "02_GIS_Output", "us_strip.shp")
BORDER     = os.path.join(REPO, "revision_c8c11", "02_GIS_Output", "bc_us_border.shp")
US_STATES  = os.path.join(REPO, "revision_c8c11", "01_Input_Data",
                          "borders", "US_states", "cb_2023_us_state_5m.shp")

OUT_FIG_DIR = os.path.join(REPO, "revision_c8c11", "03_Figures")
OUT_TBL_DIR = os.path.join(REPO, "revision_c8c11", "06_Final_Tables")
os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_TBL_DIR, exist_ok=True)

FIG_PDF   = os.path.join(OUT_FIG_DIR, "Fig_S6b_xborder_consistency.pdf")
FIG_PNG   = os.path.join(OUT_FIG_DIR, "Fig_S6b_xborder_consistency.png")
STATS_JSON = os.path.join(OUT_TBL_DIR, "T_C4_xborder_tier1_tier3.json")
HEX_CSV   = os.path.join(OUT_TBL_DIR, "T_C4_hex_tile_values.csv")

# ---------------- Constants ----------------
CRS_OUT       = "EPSG:3005"      # NAD83 BC Albers - area-preserving for BC
RES_M         = 1500.0           # 1.5 km template grid
HEX_RADIUS_M  = 25000.0          # 25 km circumradius -> 50 km diameter hex
NEARBAND_M    = 5000.0           # 5 km near-border band for Tier 3
CLASS_BREAKS  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # manual 5-class breaks
US_STATE_FIPS = {"53": "WA", "16": "ID", "30": "MT"}  # for per-side analysis

print("=" * 72)
print(" Cross-border Tier 1 (consistency) + Tier 3 (no discontinuity)")
print(" Tier 2 omitted - addressed in manuscript via citation (see limitations)")
print("=" * 72)

# ---------------- Step 1: Load strips, border, US states; build template grid ----------------
print("\n[1/7] Loading vector inputs and building template grid")

bc_strip_gdf = gpd.read_file(BC_STRIP).to_crs(CRS_OUT)
us_strip_gdf = gpd.read_file(US_STRIP).to_crs(CRS_OUT)
border_gdf   = gpd.read_file(BORDER).to_crs(CRS_OUT)
us_states_gdf = gpd.read_file(US_STATES).to_crs(CRS_OUT)
us_states_gdf = us_states_gdf[us_states_gdf["STATEFP"].isin(US_STATE_FIPS.keys())].copy()
us_states_gdf["state_abbr"] = us_states_gdf["STATEFP"].map(US_STATE_FIPS)

bc_strip_geom = unary_union(bc_strip_gdf.geometry.values)
us_strip_geom = unary_union(us_strip_gdf.geometry.values)
border_geom   = unary_union(border_gdf.geometry.values)

union_bounds = unary_union([bc_strip_geom, us_strip_geom]).bounds
# Pad by one cell so we never clip a hex on the boundary
pad = RES_M
minx, miny, maxx, maxy = union_bounds
minx -= pad; miny -= pad; maxx += pad; maxy += pad
# Snap to RES_M multiples for clean template origin
minx = np.floor(minx / RES_M) * RES_M
miny = np.floor(miny / RES_M) * RES_M
maxx = np.ceil (maxx / RES_M) * RES_M
maxy = np.ceil (maxy / RES_M) * RES_M

W = int(round((maxx - minx) / RES_M))
H = int(round((maxy - miny) / RES_M))
template_transform = from_bounds(minx, miny, maxx, maxy, W, H)
print(f"  Template grid: {H} rows x {W} cols @ {RES_M:.0f} m, CRS={CRS_OUT}")
print(f"  Bounds (m):    minx={minx:.0f} miny={miny:.0f} maxx={maxx:.0f} maxy={maxy:.0f}")

# ---------------- Step 2: Reproject both rasters onto the template ----------------
def reproject_to_template(src_path, label, resampling=Resampling.bilinear):
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            src_arr = np.where(src_arr == src.nodata, np.nan, src_arr)
        dst = np.full((H, W), np.nan, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_transform,
            dst_crs=CRS_OUT,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        print(f"  reprojected {label}: src {src.shape} {src.crs} -> "
              f"({H}, {W}) {CRS_OUT}   "
              f"valid={np.isfinite(dst).sum():,}  "
              f"range=[{np.nanmin(dst):.3f}, {np.nanmax(dst):.3f}]")
        return dst

print("\n[2/7] Reprojecting rasters onto template grid")
bc_grid = reproject_to_template(BC_SUS_TIF, "BC_susceptibility")
us_grid = reproject_to_template(BP_TIF,     "US_FSim_BP")

# ---------------- Step 3: Mask each raster by its strip ----------------
def mask_by_polygon(arr, geom, transform):
    """Return a copy of arr with NaN outside geom."""
    inside = ~geometry_mask([geom], out_shape=arr.shape,
                            transform=transform, invert=False)
    out = arr.copy()
    out[~inside] = np.nan
    return out

print("\n[3/7] Masking rasters by strip polygons")
bc_in_strip = mask_by_polygon(bc_grid, bc_strip_geom, template_transform)
us_in_strip = mask_by_polygon(us_grid, us_strip_geom, template_transform)
print(f"  BC strip valid cells: {np.isfinite(bc_in_strip).sum():,}")
print(f"  US strip valid cells: {np.isfinite(us_in_strip).sum():,}")

# ---------------- Step 4: Hex tile grid (50 km diameter, pointy-top) ----------------
def hex_grid(bounds, R, flat_top=False):
    """Pointy-top hex grid over bounds, circumradius R (m).
    Returns list of shapely Polygons."""
    minx, miny, maxx, maxy = bounds
    if flat_top:
        dx = np.sqrt(3) * R
        dy = 1.5 * R
        angle_offset = 0.0
    else:
        dx = 1.5 * R
        dy = np.sqrt(3) * R
        angle_offset = np.deg2rad(30)
    nx = int((maxx - minx) / dx) + 2
    ny = int((maxy - miny) / dy) + 2
    hexes = []
    for r in range(ny):
        for c in range(nx):
            if flat_top:
                cx = minx + c * dx + ((dx / 2) if (r % 2) else 0)
                cy = miny + r * dy
            else:
                cx = minx + c * dx
                cy = miny + r * dy + ((dy / 2) if (c % 2) else 0)
            verts = [(cx + R * np.cos(angle_offset + np.deg2rad(60 * i)),
                       cy + R * np.sin(angle_offset + np.deg2rad(60 * i)))
                      for i in range(6)]
            hexes.append(Polygon(verts))
    return hexes

print(f"\n[4/7] Generating hex tiles (R={HEX_RADIUS_M/1000:.0f} km circumradius, "
      f"D={HEX_RADIUS_M*2/1000:.0f} km diameter)")
hexes = hex_grid(union_bounds, HEX_RADIUS_M, flat_top=False)
# Keep only hexes that intersect BOTH strips (straddle the border)
both_strips = unary_union([bc_strip_geom, us_strip_geom])
straddle_hexes = []
for h in hexes:
    if (h.intersects(bc_strip_geom) and h.intersects(us_strip_geom)):
        straddle_hexes.append(h)
print(f"  Hexes total: {len(hexes):,}  straddle (BC AND US): {len(straddle_hexes):,}")

# Spatial assignment of each hex to a US state (for per-side rho)
def assign_state(h):
    cent = h.centroid
    for _, row in us_states_gdf.iterrows():
        if row.geometry.contains(cent):
            return row["state_abbr"]
    # If centroid is in Canada, take the US state whose geometry the
    # hex's US-side intersection lies in.
    us_part = h.intersection(us_strip_geom)
    for _, row in us_states_gdf.iterrows():
        if row.geometry.intersects(us_part):
            return row["state_abbr"]
    return "Unknown"

print("  Assigning hexes to US states (WA / ID / MT) ...")
hex_states = [assign_state(h) for h in tqdm(straddle_hexes, desc="  state assign", unit="hex")]

# ---------------- Step 5: Per-hex aggregation ----------------
def cells_in_polygon(arr, geom, transform):
    """Return finite values of arr that fall inside geom."""
    inside = ~geometry_mask([geom], out_shape=arr.shape,
                            transform=transform, invert=False)
    vals = arr[inside]
    return vals[np.isfinite(vals)]

print("\n[5/7] Per-hex aggregation")
rows = []
for hi, h in enumerate(tqdm(straddle_hexes, desc="  aggregating hexes", unit="hex")):
    bc_part = h.intersection(bc_strip_geom)
    us_part = h.intersection(us_strip_geom)
    bc_vals = cells_in_polygon(bc_in_strip, bc_part, template_transform) if not bc_part.is_empty else np.array([])
    us_vals = cells_in_polygon(us_in_strip, us_part, template_transform) if not us_part.is_empty else np.array([])
    if len(bc_vals) >= 3 and len(us_vals) >= 3:
        rows.append({
            "hex_id":   hi,
            "x":        h.centroid.x,
            "y":        h.centroid.y,
            "us_state": hex_states[hi],
            "bc_mean":  float(np.mean(bc_vals)),
            "bc_n":     int(len(bc_vals)),
            "us_mean":  float(np.mean(us_vals)),
            "us_n":     int(len(us_vals)),
        })
hex_df = pd.DataFrame(rows)
print(f"  Hexes with >=3 cells on both sides: {len(hex_df)}")
hex_df.to_csv(HEX_CSV, index=False)
print(f"  wrote {HEX_CSV}")

# ---------------- Step 6: Tier 1 (Spearman rho + Cohen's kappa) ----------------
print("\n[6/7] Tier 1: rank consistency between BC and US")

def quintile_classify(values, breaks=CLASS_BREAKS):
    """Classify values into 1..5 using breaks (5 classes => 4 internal cuts).
    breaks of length 6 -> upper bounds 0.2/0.4/0.6/0.8/1.0."""
    cuts = breaks[1:-1]
    cls = np.digitize(values, cuts) + 1
    return np.clip(cls, 1, 5)

results = {
    "tier1": {},
    "tier3": {},
    "meta": {
        "bc_susceptibility": os.path.relpath(BC_SUS_TIF, REPO),
        "us_burn_probability": os.path.relpath(BP_TIF, REPO),
        "bc_strip": os.path.relpath(BC_STRIP, REPO),
        "us_strip": os.path.relpath(US_STRIP, REPO),
        "border":   os.path.relpath(BORDER, REPO),
        "template_grid_m": RES_M,
        "template_shape": [H, W],
        "template_crs": CRS_OUT,
        "hex_diameter_km": HEX_RADIUS_M * 2 / 1000,
        "nearband_km": NEARBAND_M / 1000,
        "class_breaks": CLASS_BREAKS,
    }
}

if len(hex_df) >= 5:
    bc_means = hex_df["bc_mean"].values
    us_means = hex_df["us_mean"].values
    # Pooled Spearman
    rho_p, p_p = spearmanr(bc_means, us_means)
    # Cohen's kappa on 5-class binning
    bc_cls = quintile_classify(bc_means)
    us_cls = quintile_classify(us_means)
    kappa  = cohen_kappa_score(bc_cls, us_cls, weights="quadratic")
    results["tier1"]["pooled"] = {
        "n_hexes":     int(len(hex_df)),
        "spearman_rho": float(rho_p),
        "spearman_p":   float(p_p),
        "cohen_kappa_quadratic": float(kappa),
    }
    print(f"  Pooled (n={len(hex_df)}): rho={rho_p:+.3f} (p={p_p:.2e})   "
          f"kappa_quadratic={kappa:+.3f}")

    # Per-state Spearman
    for st in ("WA", "ID", "MT"):
        sub = hex_df[hex_df["us_state"] == st]
        if len(sub) >= 4:
            r_s, p_s = spearmanr(sub["bc_mean"].values, sub["us_mean"].values)
            results["tier1"][st] = {
                "n_hexes": int(len(sub)),
                "spearman_rho": float(r_s),
                "spearman_p":   float(p_s),
            }
            print(f"  {st} (n={len(sub)}): rho={r_s:+.3f} (p={p_s:.2e})")
        else:
            results["tier1"][st] = {"n_hexes": int(len(sub)), "note": "n<4, rho not reported"}
            print(f"  {st} (n={len(sub)}): not enough hexes for stable rho")
else:
    results["tier1"]["pooled"] = {"n_hexes": int(len(hex_df)),
                                   "note": "n<5, rho/kappa not reported"}
    print(f"  Pooled (n={len(hex_df)}): too few hexes for stable statistics")

# ---------------- Step 7: Tier 3 (Mann-Whitney across near-border band) ----------------
print("\n[7/7] Tier 3: no-discontinuity test across the near-border band")
near_band = border_geom.buffer(NEARBAND_M)
bc_near_geom = near_band.intersection(bc_strip_geom)
us_near_geom = near_band.intersection(us_strip_geom)
bc_near_vals = cells_in_polygon(bc_in_strip, bc_near_geom, template_transform)
us_near_vals = cells_in_polygon(us_in_strip, us_near_geom, template_transform)
print(f"  Near-band (<= {NEARBAND_M/1000:.0f} km from border): "
      f"BC cells={len(bc_near_vals):,}, US cells={len(us_near_vals):,}")

if len(bc_near_vals) >= 30 and len(us_near_vals) >= 30:
    U, p_u = mannwhitneyu(bc_near_vals, us_near_vals, alternative="two-sided")
    # Common-language effect size (rank-biserial)
    n1, n2 = len(bc_near_vals), len(us_near_vals)
    cles = U / (n1 * n2)
    rb   = 2 * cles - 1
    results["tier3"] = {
        "n_bc_near": int(n1),
        "n_us_near": int(n2),
        "U":         float(U),
        "p_value":   float(p_u),
        "cles":      float(cles),
        "rank_biserial": float(rb),
        "bc_near_mean": float(np.mean(bc_near_vals)),
        "us_near_mean": float(np.mean(us_near_vals)),
        "bc_near_median": float(np.median(bc_near_vals)),
        "us_near_median": float(np.median(us_near_vals)),
    }
    print(f"  Mann-Whitney U = {U:.0f}   p = {p_u:.2e}   "
          f"CLES = {cles:.3f}   rank-biserial = {rb:+.3f}")
    print(f"  BC near mean={np.mean(bc_near_vals):.3f} median={np.median(bc_near_vals):.3f}")
    print(f"  US near mean={np.mean(us_near_vals):.3f} median={np.median(us_near_vals):.3f}")
    if p_u >= 0.05:
        print(f"  -> p >= 0.05: no statistically detectable discontinuity at the border")
    else:
        print(f"  -> p <  0.05: distributional difference present (interpret carefully:")
        print(f"     the two layers are different products and a small offset is expected)")
else:
    results["tier3"] = {"n_bc_near": int(len(bc_near_vals)),
                          "n_us_near": int(len(us_near_vals)),
                          "note": "n<30 on one side; test skipped"}
    print(f"  Skipped: too few cells in near-band")

# ---------------- Save stats JSON ----------------
with open(STATS_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {STATS_JSON}")

# ---------------- Figure S6b ----------------
print("\nBuilding Fig_S6b (2-panel BC | US on shared colour scale)")
fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharex=True, sharey=True)
# Shared color scale on 0-1 (since both rasters are probabilities)
cmap = mpl.colormaps["YlOrRd"]
norm = mpl.colors.Normalize(vmin=0, vmax=1)

# BC panel
ax = axes[0]
ax.set_title("BC BiLSTM-PSO susceptibility (strip)", fontsize=12)
extent = [minx, maxx, miny, maxy]
ax.imshow(bc_in_strip, cmap=cmap, norm=norm, extent=extent, origin="upper")
border_gdf.plot(ax=ax, color="black", linewidth=1.2, zorder=10)
bc_strip_gdf.boundary.plot(ax=ax, color="cyan", linewidth=0.5, zorder=9)
us_strip_gdf.boundary.plot(ax=ax, color="orange", linewidth=0.5, zorder=9)
ax.set_aspect("equal"); ax.set_xlabel("x (m, EPSG:3005)"); ax.set_ylabel("y (m, EPSG:3005)")

# US panel
ax = axes[1]
ax.set_title("U.S. FSim burn probability (strip)", fontsize=12)
ax.imshow(us_in_strip, cmap=cmap, norm=norm, extent=extent, origin="upper")
border_gdf.plot(ax=ax, color="black", linewidth=1.2, zorder=10)
bc_strip_gdf.boundary.plot(ax=ax, color="cyan", linewidth=0.5, zorder=9)
us_strip_gdf.boundary.plot(ax=ax, color="orange", linewidth=0.5, zorder=9)
ax.set_aspect("equal"); ax.set_xlabel("x (m, EPSG:3005)")

cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.65])
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax,
              label="Probability (BC: BiLSTM-PSO susc., US: FSim BP)")

# Annotation block with key stats
ann = []
if "pooled" in results["tier1"] and "spearman_rho" in results["tier1"]["pooled"]:
    ann.append(f"Tier 1 pooled: rho={results['tier1']['pooled']['spearman_rho']:+.3f}, "
                f"kappa={results['tier1']['pooled']['cohen_kappa_quadratic']:+.3f} "
                f"(n={results['tier1']['pooled']['n_hexes']} hexes)")
for st in ("WA", "ID", "MT"):
    if "spearman_rho" in results["tier1"].get(st, {}):
        ann.append(f"  {st}: rho={results['tier1'][st]['spearman_rho']:+.3f} "
                    f"(n={results['tier1'][st]['n_hexes']})")
if "p_value" in results["tier3"]:
    ann.append(f"Tier 3 (Mann-Whitney, {NEARBAND_M/1000:.0f} km near-band): "
                f"p={results['tier3']['p_value']:.2e}, "
                f"CLES={results['tier3']['cles']:.3f}")
fig.text(0.02, 0.02, "\n".join(ann), fontsize=9, va="bottom", family="monospace")

plt.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.10, wspace=0.12)
plt.savefig(FIG_PDF, dpi=300, bbox_inches="tight")
plt.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
print(f"  wrote {FIG_PDF}")
print(f"  wrote {FIG_PNG}")

print("\nDone.")
