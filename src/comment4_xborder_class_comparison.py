# -*- coding: utf-8 -*-
"""
D2: comment4_xborder_class_comparison.py
========================================
Cross-border class-distribution + hex-tile Spearman comparison
between BC BiLSTM-PSO susceptibility and U.S. WHP.

Uses PROVINCE-WIDE quintile breaks (not strip-only).
Uses FULL BC-CONUS border (BC strip + US strip from Phase B).
Uses symmetric buffer + intersect (already done in ArcGIS Pro).
"""
import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry import Polygon, Point
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO     = r"C:\Users\saadz\Documents\wildfire-bc-bilstm-pso"
BC_SUS   = os.path.join(REPO, "data", "BC_susceptibility_500m.tif")
WHP      = os.path.join(REPO, "revision_c8c11", "01_Input_Data",
                        "US_fire_products", "WHP", "WHP2023_classified_conus.tif")
BC_STRIP = os.path.join(REPO, "revision_c8c11", "02_GIS_Output", "bc_strip.shp")
US_STRIP = os.path.join(REPO, "revision_c8c11", "02_GIS_Output", "us_strip.shp")
OUT_TBL  = os.path.join(REPO, "revision_c8c11", "06_Final_Tables")
OUT_FIG  = os.path.join(REPO, "figs")
os.makedirs(OUT_TBL, exist_ok=True); os.makedirs(OUT_FIG, exist_ok=True)

CRS_3005 = "EPSG:3005"
TILE_KM = 50.0  # hex tile size

print("=" * 70)
print("D2: Cross-border BC vs WHP class comparison (full BC-CONUS border)")
print("=" * 70)

# ----- (1) Compute province-wide quintile breaks on BC susceptibility -----
print("\n[1/6] Computing province-wide quintile breaks on BC raster ...")
with rasterio.open(BC_SUS) as src:
    bc_full = src.read(1)
    bc_meta = src.meta.copy()
    bc_transform = src.transform
    bc_crs = src.crs
    valid_full = np.isfinite(bc_full)
    province_breaks = np.percentile(bc_full[valid_full], [20, 40, 60, 80])
    print(f"  Province-wide quintile breaks: {province_breaks}")

def classify_5(arr, breaks):
    """Classify continuous arr into 1-5 using breaks [q20, q40, q60, q80]."""
    out = np.full_like(arr, np.nan, dtype=np.float32)
    valid = np.isfinite(arr)
    cls = np.digitize(arr[valid], breaks) + 1   # gives 1..5
    out[valid] = cls
    return out

bc_classified = classify_5(bc_full, province_breaks)

# ----- (2) Mask classified BC raster to BC strip -----
print("\n[2/6] Masking BC classified raster to BC strip ...")
bc_strip_gdf = gpd.read_file(BC_STRIP).to_crs(CRS_3005)
tmp_bc_cls = os.path.join(OUT_TBL, "_bc_classified_full.tif")
with rasterio.open(tmp_bc_cls, "w", driver="GTiff",
        height=bc_classified.shape[0], width=bc_classified.shape[1],
        count=1, dtype="float32", crs=bc_crs, transform=bc_transform,
        nodata=np.nan, compress="lzw") as dst:
    dst.write(bc_classified, 1)

with rasterio.open(tmp_bc_cls) as src:
    bc_strip_arr, bc_strip_tx = mask(src, bc_strip_gdf.geometry, crop=True, nodata=np.nan)
    bc_strip_arr = bc_strip_arr[0]

print(f"  BC strip valid pixels: {int(np.isfinite(bc_strip_arr).sum()):,}")

# ----- (3) Reproject WHP to EPSG:3005 @ 500 m and mask to US strip -----
print("\n[3/6] Reprojecting WHP to EPSG:3005 @ 500m and masking to US strip ...")
us_strip_gdf = gpd.read_file(US_STRIP).to_crs(CRS_3005)
tmp_whp_reproj = os.path.join(OUT_TBL, "_whp_reproj.tif")
with rasterio.open(WHP) as src:
    tr, w, h = calculate_default_transform(src.crs, CRS_3005,
                                            src.width, src.height, *src.bounds,
                                            resolution=500.0)
    arr = np.full((h, w), np.nan, dtype=np.float32)
    reproject(rasterio.band(src,1), arr,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=tr, dst_crs=CRS_3005,
              resampling=Resampling.nearest)  # nearest for class data
    with rasterio.open(tmp_whp_reproj, "w", driver="GTiff",
            height=h, width=w, count=1, dtype="float32",
            crs=CRS_3005, transform=tr, nodata=np.nan, compress="lzw") as dst:
        dst.write(arr, 1)

with rasterio.open(tmp_whp_reproj) as src:
    us_strip_arr, us_strip_tx = mask(src, us_strip_gdf.geometry, crop=True, nodata=np.nan)
    us_strip_arr = us_strip_arr[0]

# WHP native: 1=VeryLow, 2=Low, 3=Mod, 4=High, 5=VeryHigh; non-burnable=NaN
print(f"  US strip valid WHP pixels: {int(np.isfinite(us_strip_arr).sum()):,}")

# ----- (4) Class-distribution comparison -----
print("\n[4/6] Computing class distributions ...")
CLASS_NAMES = ["Very Low","Low","Moderate","High","Very High"]
rows = []
for k, name in enumerate(CLASS_NAMES, start=1):
    bc_pct = 100*np.sum(bc_strip_arr == k) / max(np.isfinite(bc_strip_arr).sum(), 1)
    us_pct = 100*np.sum(us_strip_arr == k) / max(np.isfinite(us_strip_arr).sum(), 1)
    rows.append({"class": name, "class_value": k,
                 "BC_strip_pct": round(bc_pct, 2),
                 "US_strip_pct": round(us_pct, 2)})
df_class = pd.DataFrame(rows)
bc_h_vh = df_class.query("class in ['High','Very High']")["BC_strip_pct"].sum()
us_h_vh = df_class.query("class in ['High','Very High']")["US_strip_pct"].sum()
df_class.loc[len(df_class)] = ["High+Very High combined", None,
                                round(bc_h_vh, 2), round(us_h_vh, 2)]
df_class.to_csv(os.path.join(OUT_TBL, "T_C4_class_distribution.csv"), index=False)
print(df_class.to_string(index=False))

# ----- (5) Hex-tile Spearman on matched border tiles -----
print("\n[5/6] Tessellating into 50-km hex tiles + matched-tile Spearman ...")

def hex_grid(bounds, tile_km=50.0):
    """Build a hex grid covering bounds. Tiles in EPSG:3005 (metres)."""
    minx, miny, maxx, maxy = bounds
    s = tile_km * 1000.0    # tile centre spacing
    w_step = s * np.sqrt(3)/2  # horizontal step
    h_step = s * 3/4           # vertical step
    polys = []
    y = miny
    row = 0
    while y < maxy + s:
        x_offset = 0 if row % 2 == 0 else w_step/2
        x = minx + x_offset
        while x < maxx + s:
            # Hex with flat top
            cx, cy = x, y
            verts = [(cx + s/2 * np.cos(np.radians(60*i + 30)),
                      cy + s/2 * np.sin(np.radians(60*i + 30))) for i in range(6)]
            polys.append(Polygon(verts))
            x += w_step
        y += h_step
        row += 1
    return gpd.GeoDataFrame(geometry=polys, crs=CRS_3005)

# Build hex grids for BC and US strips (limit to their bounding boxes)
bc_hex = hex_grid(bc_strip_gdf.total_bounds, tile_km=TILE_KM)
bc_hex = gpd.overlay(bc_hex, bc_strip_gdf, how="intersection", keep_geom_type=False)
bc_hex = bc_hex[bc_hex.geometry.area > (TILE_KM*1000)**2 * 0.3]  # drop tiny edge tiles
bc_hex["centroid"] = bc_hex.geometry.centroid

us_hex = hex_grid(us_strip_gdf.total_bounds, tile_km=TILE_KM)
us_hex = gpd.overlay(us_hex, us_strip_gdf, how="intersection", keep_geom_type=False)
us_hex = us_hex[us_hex.geometry.area > (TILE_KM*1000)**2 * 0.3]
us_hex["centroid"] = us_hex.geometry.centroid

print(f"  BC hex tiles: {len(bc_hex)}")
print(f"  US hex tiles: {len(us_hex)}")

# Compute mean BC class per BC tile (zonal stats — sample pixels under tile)
def tile_mean_class(tiles, raster_path):
    means = []
    with rasterio.open(raster_path) as src:
        for geom in tiles.geometry:
            try:
                arr, _ = mask(src, [geom], crop=True, nodata=np.nan)
                v = arr[0]; v = v[np.isfinite(v)]
                means.append(float(np.mean(v)) if len(v) else np.nan)
            except Exception:
                means.append(np.nan)
    return np.array(means)

def tile_mode_class(tiles, raster_path):
    modes = []
    with rasterio.open(raster_path) as src:
        for geom in tiles.geometry:
            try:
                arr, _ = mask(src, [geom], crop=True, nodata=np.nan)
                v = arr[0]; v = v[np.isfinite(v)].astype(int)
                if len(v):
                    vals, cnt = np.unique(v, return_counts=True)
                    modes.append(int(vals[np.argmax(cnt)]))
                else:
                    modes.append(np.nan)
            except Exception:
                modes.append(np.nan)
    return np.array(modes, dtype=float)

bc_hex["bc_mean_class"] = tile_mean_class(bc_hex, tmp_bc_cls)
us_hex["us_mode_class"] = tile_mode_class(us_hex, tmp_whp_reproj)

# Pair each BC tile with nearest US tile (one-to-one nearest neighbour)
bc_cent = np.array([[p.x, p.y] for p in bc_hex.centroid])
us_cent = np.array([[p.x, p.y] for p in us_hex.centroid])
from scipy.spatial import cKDTree
tree = cKDTree(us_cent)
dist, idx = tree.query(bc_cent, k=1)

pairs = []
used_us = set()
order = np.argsort(dist)  # match closest first to preserve 1:1
for bi in order:
    ui = idx[bi]
    if ui in used_us: continue
    used_us.add(ui)
    pairs.append({"bc_idx": int(bi), "us_idx": int(ui),
                  "dist_m": float(dist[bi]),
                  "bc_mean_class": float(bc_hex["bc_mean_class"].iloc[bi]),
                  "us_mode_class": float(us_hex["us_mode_class"].iloc[ui])})
df_pairs = pd.DataFrame(pairs)
df_pairs = df_pairs.dropna(subset=["bc_mean_class", "us_mode_class"])
print(f"  Paired hex tiles (after NaN drop): {len(df_pairs)}")

# Spearman + bootstrap CI
rho, p = spearmanr(df_pairs["bc_mean_class"], df_pairs["us_mode_class"])
rng = np.random.default_rng(42)
n_boot = 1000
boot = np.empty(n_boot)
n = len(df_pairs)
for i in range(n_boot):
    idx_b = rng.integers(0, n, size=n)
    boot[i] = spearmanr(df_pairs["bc_mean_class"].iloc[idx_b],
                         df_pairs["us_mode_class"].iloc[idx_b])[0]
ci = np.percentile(boot[~np.isnan(boot)], [2.5, 97.5])
print(f"  Spearman rho = {rho:.3f} (p={p:.2e})")
print(f"  Bootstrap 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

# ----- (6) Save outputs + Fig S6 -----
print("\n[6/6] Saving summary JSON + Figure S6 ...")
summary = {
    "bc_strip_area_km2": float(bc_strip_gdf.area.sum() / 1e6),
    "us_strip_area_km2": float(us_strip_gdf.area.sum() / 1e6),
    "bc_strip_valid_pixels": int(np.isfinite(bc_strip_arr).sum()),
    "us_strip_valid_pixels": int(np.isfinite(us_strip_arr).sum()),
    "province_wide_quintile_breaks": [float(x) for x in province_breaks],
    "bc_high_vh_pct": float(bc_h_vh),
    "us_high_vh_pct": float(us_h_vh),
    "paired_hex_tiles": int(len(df_pairs)),
    "spearman_rho": float(rho),
    "spearman_p_value": float(p),
    "spearman_95CI": [float(ci[0]), float(ci[1])],
    "border_length_km_approximate": 1060,
    "tile_size_km": TILE_KM,
    "interpretation": "external consistency check, not validation",
}
with open(os.path.join(OUT_TBL, "T_C4_xborder_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# Figure S6: 3-panel
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
cmap_cls = mcolors.ListedColormap(['#fef0d9','#fdcc8a','#fc8d59','#e34a33','#b30000'])
bounds_5 = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = mcolors.BoundaryNorm(bounds_5, cmap_cls.N)

im0 = axes[0].imshow(bc_strip_arr, cmap=cmap_cls, norm=norm)
axes[0].set_title("(a) BC strip (50 km north of border)\nBC susceptibility, 5-class\n[province-wide quintile breaks]")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], ticks=[1,2,3,4,5], label="Class")

im1 = axes[1].imshow(us_strip_arr, cmap=cmap_cls, norm=norm)
axes[1].set_title("(b) US strip (50 km south of border)\nWHP 2023, 5-class native")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], ticks=[1,2,3,4,5], label="Class")

axes[2].scatter(df_pairs["bc_mean_class"], df_pairs["us_mode_class"],
                s=40, alpha=0.7, c="darkred", edgecolors="black", linewidth=0.5)
axes[2].plot([0.5, 5.5], [0.5, 5.5], "k--", lw=0.7, label="1:1 line")
axes[2].set_xlabel("BC mean class per 50-km hex tile")
axes[2].set_ylabel("US modal WHP class per matched tile")
axes[2].set_title(f"(c) Matched hex tiles (N={len(df_pairs)})\nSpearman ρ = {rho:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
axes[2].set_xlim(0.5, 5.5); axes[2].set_ylim(0.5, 5.5)
axes[2].grid(alpha=0.3); axes[2].legend()

plt.tight_layout()
out_pdf = os.path.join(OUT_FIG, "Fig_S6_xborder_class_comparison.pdf")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"\nSaved figure: {out_pdf}")
print("DONE.")
