import os, sys, json, csv, glob, argparse
try:
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    from rasterio.features import geometry_mask, rasterize
    import geopandas as gpd
    from scipy.stats import spearmanr, mannwhitneyu
    from scipy.ndimage import distance_transform_edt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
except ImportError as e:
    sys.exit("Missing package: {0}\nInstall into the env you ran your last script "
             "from, e.g.:  conda install -c conda-forge rasterio geopandas scipy "
             "matplotlib".format(e))
REPO = r"."
DEF_US_BP    = os.path.join(REPO, "01_Input_Data", "borders", "US_burn_probability", "BP_WA_ID_MT.tif")
DEF_BC_STRIP = os.path.join(REPO, "02_GIS_Output", "bc_strip.shp")
DEF_US_STRIP = os.path.join(REPO, "02_GIS_Output", "us_strip.shp")
DEF_BORDER   = os.path.join(REPO, "02_GIS_Output", "bc_us_border.shp")
FIG_DIR = os.path.join(REPO, "03_Figures")
TAB_DIR = os.path.join(REPO, "06_Final_Tables")
TARGET_EPSG = 3005
CELL        = 1500
N_BINS      = 5
MIN_CELLS   = 3
NEAR_M      = 5000
TILE_PRIMARY_KM = 20
TILE_SWEEP_KM   = [15, 20, 25, 30]
def log(s): print(s, flush=True)
def build_grid(strip_paths):
    xs, ys = [], []
    for p in strip_paths:
        g = gpd.read_file(p).to_crs(epsg=TARGET_EPSG)
        minx, miny, maxx, maxy = g.total_bounds
        xs += [minx, maxx]; ys += [miny, maxy]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minx = np.floor(minx / CELL) * CELL
    miny = np.floor(miny / CELL) * CELL
    maxx = np.ceil(maxx / CELL) * CELL
    maxy = np.ceil(maxy / CELL) * CELL
    width  = int(round((maxx - minx) / CELL))
    height = int(round((maxy - miny) / CELL))
    transform = from_origin(minx, maxy, CELL, CELL)
    log("  Common grid: {0} rows x {1} cols @ {2} m, EPSG:{3}".format(
        height, width, CELL, TARGET_EPSG))
    return transform, width, height, (minx, miny, maxx, maxy)
def warp_to_grid(path, transform, width, height):
    with rasterio.open(path) as src:
        dst = np.full((height, width), np.nan, dtype="float64")
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs="EPSG:{0}".format(TARGET_EPSG),
            src_nodata=src.nodata, dst_nodata=np.nan,
            resampling=Resampling.bilinear)
        log("    warped {0}: src {1} {2} -> grid; finite={3:,} range=[{4:.3f},{5:.3f}]".format(
            os.path.basename(path), src.shape, src.crs,
            int(np.isfinite(dst).sum()),
            float(np.nanmin(dst)) if np.isfinite(dst).any() else float("nan"),
            float(np.nanmax(dst)) if np.isfinite(dst).any() else float("nan")))
    return dst
def strip_mask(path, transform, width, height):
    g = gpd.read_file(path).to_crs(epsg=TARGET_EPSG)
    geoms = [geom for geom in g.geometry if geom is not None]
    m = geometry_mask(geoms, out_shape=(height, width), transform=transform, invert=True)
    return m
def border_distance(path, transform, width, height):
    g = gpd.read_file(path).to_crs(epsg=TARGET_EPSG)
    shapes = [(geom, 1) for geom in g.geometry if geom is not None]
    bmask = rasterize(shapes, out_shape=(height, width), transform=transform,
                      fill=0, all_touched=True).astype(bool)
    if not bmask.any():
        log("  [WARN] border rasterized to 0 cells; distance band may be empty.")
    dist = distance_transform_edt(~bmask) * CELL
    return dist
def pct_rank(a, mask):
    out = np.full(a.shape, np.nan)
    v = a[mask]
    if len(v) == 0: return out
    order = v.argsort(); r = np.empty(len(v)); r[order] = np.arange(len(v))
    out[mask] = 100.0 * r / max(len(v) - 1, 1)
    return out
def quantile_class(vals, k):
    qs = np.quantile(vals, np.linspace(0, 1, k + 1)[1:-1])
    return np.digitize(vals, qs)
def weighted_kappa(a, b, k, quadratic=True):
    a = np.asarray(a, int); b = np.asarray(b, int)
    O = np.zeros((k, k))
    for x, y in zip(a, b): O[x, y] += 1
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            W[i, j] = (i - j) ** 2 if quadratic else abs(i - j)
    W /= (k - 1) ** 2
    n = O.sum()
    if n == 0: return float("nan")
    E = np.outer(O.sum(1), O.sum(0)) / n
    denom = (W * E).sum()
    return 1.0 - (W * O).sum() / denom if denom > 0 else float("nan")
def tile_pairs(bc, us, bc_m, us_m, transform, tile_km):
    tile = max(int(round(tile_km * 1000.0 / CELL)), 1)
    h, w = bc.shape
    rows, cols = np.indices((h, w))
    ncols_t = (w // tile) + 2
    tid = ((rows // tile) * ncols_t + (cols // tile)).ravel()
    fin_bc = (bc_m & np.isfinite(bc)).ravel()
    fin_us = (us_m & np.isfinite(us)).ravel()
    bcv = np.where(fin_bc, bc.ravel(), 0.0)
    usv = np.where(fin_us, us.ravel(), 0.0)
    mlen = tid.max() + 1
    sum_bc = np.bincount(tid, weights=bcv, minlength=mlen)
    cnt_bc = np.bincount(tid, weights=fin_bc.astype(float), minlength=mlen)
    sum_us = np.bincount(tid, weights=usv, minlength=mlen)
    cnt_us = np.bincount(tid, weights=fin_us.astype(float), minlength=mlen)
    ok = (cnt_bc >= MIN_CELLS) & (cnt_us >= MIN_CELLS)
    bc_mean = sum_bc[ok] / cnt_bc[ok]
    us_mean = sum_us[ok] / cnt_us[ok]
    return bc_mean, us_mean
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-susc", default=None, help="continuous BC susceptibility raster (0-1)")
    ap.add_argument("--us-bp", default=DEF_US_BP)
    ap.add_argument("--bc-strip", default=DEF_BC_STRIP)
    ap.add_argument("--us-strip", default=DEF_US_STRIP)
    ap.add_argument("--border", default=DEF_BORDER)
    args = ap.parse_args()
    log("=" * 64)
    log("  Cross-border comment #4 -- Tier 1 (consistency) + Tier 3 (edge)")
    log("=" * 64)
    bc_path = args.bc_susc
    if not bc_path or not os.path.exists(bc_path):
        log("BC susceptibility raster not given/found.")
        cands = []
        for d in (os.path.join(REPO, "02_GIS_Output"),
                  os.path.join(REPO, "05_Model_Output"),
                  REPO):
            cands += glob.glob(os.path.join(d, "**", "*.tif"), recursive=True)
        cands = sorted(set(cands))
        log("Candidate .tif files in the project (pass one with --bc-susc):")
        for c in cands:
            log("   " + c)
        sys.exit("\nRe-run e.g.:  python cross_border_c4.py --bc-susc \"<one of the above>\"")
    for p, lab in [(bc_path, "BC susceptibility"), (args.us_bp, "US burn probability"),
                   (args.bc_strip, "bc_strip"), (args.us_strip, "us_strip"),
                   (args.border, "border")]:
        if not os.path.exists(p):
            sys.exit("Missing {0}: {1}".format(lab, p))
    os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)
    log("[1/5] Building common grid from strips")
    transform, W, H, bounds = build_grid([args.bc_strip, args.us_strip])
    log("[2/5] Warping rasters onto the grid")
    bc = warp_to_grid(bc_path, transform, W, H)
    us = warp_to_grid(args.us_bp, transform, W, H)
    log("[3/5] Masking by strips + computing border distance")
    bc_m = strip_mask(args.bc_strip, transform, W, H)
    us_m = strip_mask(args.us_strip, transform, W, H)
    dist = border_distance(args.border, transform, W, H)
    log("    BC strip cells: {0:,}   US strip cells: {1:,}".format(
        int((bc_m & np.isfinite(bc)).sum()), int((us_m & np.isfinite(us)).sum())))
    results = {"grid": {"rows": H, "cols": W, "cell_m": CELL, "epsg": TARGET_EPSG},
               "tier1": {}, "tier3": {}}
    log("[4/5] TIER 1 - cross-border rank consistency (straddle tiles)")
    sweep = {}
    primary_pairs = None
    for tkm in TILE_SWEEP_KM:
        bcm, usm = tile_pairs(bc, us, bc_m, us_m, transform, tkm)
        n = len(bcm)
        if n >= 4:
            rho, p = spearmanr(bcm, usm)
        else:
            rho, p = float("nan"), float("nan")
        sweep[tkm] = {"n_tiles": int(n), "spearman_rho": float(rho), "p": float(p)}
        log("    tile={0:>2d} km : n={1:<4d} rho={2:+.3f}  p={3:.2e}".format(
            tkm, n, rho, p))
        if tkm == TILE_PRIMARY_KM:
            primary_pairs = (bcm, usm)
    kap = float("nan")
    if primary_pairs is not None and len(primary_pairs[0]) >= N_BINS:
        bcm, usm = primary_pairs
        ca = quantile_class(bcm, N_BINS); cb = quantile_class(usm, N_BINS)
        kap = weighted_kappa(ca, cb, N_BINS, quadratic=True)
    log("    primary tile = {0} km : quadratic-weighted kappa = {1:+.3f}".format(
        TILE_PRIMARY_KM, kap))
    results["tier1"] = {"primary_tile_km": TILE_PRIMARY_KM,
                        "primary_kappa_quadratic": kap,
                        "sweep": sweep,
                        "min_cells_per_side": MIN_CELLS}
    if primary_pairs is not None:
        csv_path = os.path.join(TAB_DIR, "T_C4_tile_pairs_v2.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f); wcsv.writerow(["bc_susc_mean", "us_bp_mean"])
            for x, y in zip(*primary_pairs):
                wcsv.writerow(["{0:.6f}".format(x), "{0:.6f}".format(y)])
        log("    wrote " + csv_path)
    log("[5/5] TIER 3 - artificial-edge test WITHIN each product (near vs far)")
    def edge_test(arr, m, name):
        near = arr[m & np.isfinite(arr) & (dist <= NEAR_M)]
        far  = arr[m & np.isfinite(arr) & (dist >  NEAR_M)]
        if len(near) < 30 or len(far) < 30:
            log("    {0}: insufficient cells (near={1}, far={2}) -- skipped.".format(
                name, len(near), len(far)))
            return None
        u, p = mannwhitneyu(near, far, alternative="two-sided")
        out = {"near_n": int(len(near)), "far_n": int(len(far)),
               "near_median": float(np.median(near)), "far_median": float(np.median(far)),
               "near_mean": float(near.mean()), "far_mean": float(far.mean()),
               "mannwhitney_p": float(p),
               "verdict": "no artificial edge (near ~ far)" if p > 0.05
                          else "near-border differs from interior"}
        log("    {0}: near median={1:.4f} far median={2:.4f}  MW p={3:.2e} -> {4}".format(
            name, out["near_median"], out["far_median"], p, out["verdict"]))
        return out
    results["tier3"] = {
        "near_band_m": NEAR_M,
        "BC_within_product": edge_test(bc, bc_m, "BC susceptibility"),
        "US_within_product": edge_test(us, us_m, "US burn probability"),
        "note": ("Within-product test on a common scale. The earlier cross-product "
                 "raw-value Mann-Whitney was invalid (BC ~0-1 vs BP ~0-0.06 never "
                 "overlap -> spurious perfect separation).")}
    log("Building Fig_S6b (combined percentile map + tile scatter)")
    bc_pct = pct_rank(bc, bc_m & np.isfinite(bc))
    us_pct = pct_rank(us, us_m & np.isfinite(us))
    combined = np.where(np.isfinite(bc_pct), bc_pct,
                        np.where(np.isfinite(us_pct), us_pct, np.nan))
    minx, miny, maxx, maxy = bounds
    border_g = gpd.read_file(args.border).to_crs(epsg=TARGET_EPSG)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    im = ax[0].imshow(np.ma.masked_invalid(combined), cmap="YlOrRd", vmin=0, vmax=100,
                      extent=(minx, maxx, miny, maxy), origin="upper")
    border_g.plot(ax=ax[0], color="black", linewidth=0.8)
    ax[0].set_title("Within-strip percentile rank\n(BC susceptibility + US burn prob, shared scale)",
                    fontsize=11)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label="percentile")
    if primary_pairs is not None and len(primary_pairs[0]) >= 2:
        bcm, usm = primary_pairs
        ax[1].scatter(bcm, usm, s=28, alpha=0.7, edgecolor="k", linewidth=0.4)
        rho, p = spearmanr(bcm, usm)
        ax[1].set_xlabel("mean BC susceptibility (tile)")
        ax[1].set_ylabel("mean US burn probability (tile)")
        ax[1].set_title("Straddle-tile pairing ({0} km)\nSpearman rho={1:+.2f}, p={2:.2g}, n={3}".format(
            TILE_PRIMARY_KM, rho, p, len(bcm)), fontsize=11)
        ax[1].grid(alpha=0.3)
    fig.suptitle("Fig. S6b. Cross-border consistency of BC susceptibility vs US FSim "
                 "burn probability (50 km strips)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        fp = os.path.join(FIG_DIR, "Fig_S6b_xborder_consistency_v2." + ext)
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        log("  wrote " + fp)
    plt.close(fig)
    json_path = os.path.join(TAB_DIR, "T_C4_tier1_tier3_v2.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log("  wrote " + json_path)
    log("\nDONE.")
    log("  Read Tier 1 as: rho stable & positive across tile sizes => consistent")
    log("  cross-border ranking. Report the SWEEP, and note that few independent")
    log("  straddle units + spatial autocorrelation limit formal significance.")
    log("  Read Tier 3 as: near ~ far within each product => no artificial edge.")
if __name__ == "__main__":
    main()
