"""Compose the journal figure from the ArcGIS Pro map exports.

Reads figs/arcgis_exports/*.png (produced by arcgis_make_maps.py with native
ArcGIS Pro symbology) and assembles a single-page portrait figure with:
  - panel letters and titles (auto-wrapped, never clipped)
  - a coordinate graticule frame on every panel (ticks all round; labels on
    the outer panels), computed from the raster projection
  - a colour bar under every panel whose ramp is sampled directly from the
    ArcGIS rendering, so the legend matches Pro's symbology exactly
  - the LULC class legend, and ArcGIS Pro's own north arrow and scale bar
Sized for a 16 cm placement in Word: all text remains legible at 120% zoom.

Output: figs/Fig_predictors_final.{png,tif}  (600 dpi)
Run:    python arcgis_compose_figure.py
"""
import os
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "figs", "arcgis_exports")
RAS = os.path.join(HERE, "revision_c8c11", "01_Input_Data", "rasters")
OUT = os.path.join(HERE, "figs", "Fig_predictors_final")

# Times New Roman when present (Windows); Liberation Serif is the
# metric-compatible substitute used elsewhere.
_have = {f.name for f in fm.fontManager.ttflist}
SERIF = ("Times New Roman" if "Times New Roman" in _have else
         "Liberation Serif" if "Liberation Serif" in _have else "Nimbus Roman")
plt.rcParams.update({
    "font.family": "serif", "font.serif": [SERIF],
    "mathtext.fontset": "custom", "mathtext.rm": SERIF,
    "mathtext.it": f"{SERIF}:italic", "mathtext.bf": f"{SERIF}:bold",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

PANELS = [
    ("Elevation",           "Elevation (m)",                  "val"),
    ("Slope",               "Slope (°)",                 "val"),
    ("Aspect",              "Aspect (°)",                "asp"),
    ("TWI",                 "TWI",                            "val"),
    ("Profile_Curvature",   "Profile Curvature",              "val"),
    ("Plan_Curvature",      "Plan Curvature",                 "val"),
    ("NDVI",                "NDVI",                           "val"),
    ("Distance_rivers",     "Distance to Rivers\n(km)",       "km"),
    ("Distance_roads",      "Distance to Roads\n(km)",        "km"),
    ("Distance_households", "Distance to\nHouseholds (km)",   "km"),
    ("Max_Temperature",     "Max Temperature (°C)",      "val"),
    ("Precipitation",       "Precipitation (mm)",             "val"),
    ("Soil_Moisture",       "Soil Moisture (mm)",             "val"),
    ("WS",                  "Wind Speed (m/s)",               "val"),
    ("Relative_Humidity",   "Specific Humidity\n(kg/kg)",     "sh"),
    ("DSI",                 "PDSI",                           "val"),
    ("AET",                 "AET (mm)",                       "val"),
    ("LULC",                "Land Use / Land Cover",          "lulc"),
]

LULC_DEF = [(1, "Water", "#419BDF"), (2, "Trees", "#397D49"),
            (4, "Flooded Vegetation", "#7A87C6"), (5, "Crops", "#E49635"),
            (7, "Built Area", "#C4281B"), (8, "Bare Ground", "#A59B8F"),
            (9, "Snow/Ice", "#D6EFFF"), (11, "Rangeland", "#C6B98A")]

# --- render geometry of the ArcGIS export pages (verified to 1 px) ---
PAGE_W, PAGE_H, MARG = 18.0, 22.0, 0.4          # cm, as used for the layouts


def render_rect(img_w, img_h, grid_w, grid_h):
    """Where the full raster extent sits inside an exported page."""
    fw = (PAGE_W - 2 * MARG) / PAGE_W * img_w
    fh = (PAGE_H - 2 * MARG) / PAGE_H * img_h
    fl = MARG / PAGE_W * img_w
    ft = MARG / PAGE_H * img_h
    asp = grid_w / grid_h
    if asp < fw / fh:                 # fits by height
        dh = fh
        dw = fh * asp
    else:
        dw = fw
        dh = fw / asp
    return fl + (fw - dw) / 2, ft + (fh - dh) / 2, dw, dh


# --- common target extent: the BC data window of the Elevation grid ---
with rasterio.open(os.path.join(RAS, "Elevation.tif")) as r:
    m0 = r.read(1, masked=True)
    TR, CRS = r.transform, r.crs
rows = np.where(~m0.mask.all(axis=1))[0]
cols = np.where(~m0.mask.all(axis=0))[0]
R0, R1, C0, C1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
X0 = TR.c + C0 * TR.a
X1 = TR.c + C1 * TR.a
Y1 = TR.f + R0 * TR.e
Y0 = TR.f + R1 * TR.e
ASPECT = (X1 - X0) / (Y1 - Y0)

# every raster has its own extent; register each export to the common window
_META = {}


def meta(name):
    if name not in _META:
        with rasterio.open(os.path.join(RAS, name + ".tif")) as r:
            _META[name] = (r.width, r.height, r.bounds, r.transform)
    return _META[name]


def crop_panel(name):
    """Crop this panel's export so it represents EXACTLY the common window
    (X0..X1, Y0..Y1), padding with white where the raster does not reach."""
    im = Image.open(os.path.join(EXP, name + ".png")).convert("RGB")
    rw, rh, B, _ = meta(name)
    dl, dt, dw, dh = render_rect(im.size[0], im.size[1], rw, rh)
    sx = dw / (B.right - B.left)          # px per metre in this export
    sy = dh / (B.top - B.bottom)
    px0 = dl + (X0 - B.left) * sx
    px1 = dl + (X1 - B.left) * sx
    py0 = dt + (B.top - Y1) * sy
    py1 = dt + (B.top - Y0) * sy
    out_w, out_h = int(round(px1 - px0)), int(round(py1 - py0))
    canvas = Image.new("RGB", (out_w, out_h), "white")
    ix0, iy0 = max(0, int(round(px0))), max(0, int(round(py0)))
    ix1 = min(im.size[0], int(round(px1)))
    iy1 = min(im.size[1], int(round(py1)))
    if ix1 > ix0 and iy1 > iy0:
        part = im.crop((ix0, iy0, ix1, iy1))
        canvas.paste(part, (int(round(ix0 - px0)), int(round(iy0 - py0))))
    return canvas


def common_window(name):
    """Read this raster over the common window (boundless, masked)."""
    from rasterio.windows import from_bounds
    with rasterio.open(os.path.join(RAS, name + ".tif")) as r:
        win = from_bounds(X0, Y0, X1, Y1, r.transform)
        a = r.read(1, window=win, boundless=True, masked=True,
                   fill_value=r.nodata if r.nodata is not None else 0)
    return a


def ramp_from_render(name, crop):
    """Sample the ArcGIS ramp: pair raster values with rendered colours."""
    a = common_window(name)
    small = np.asarray(crop.resize((a.shape[1], a.shape[0]), Image.NEAREST))
    from scipy.ndimage import binary_erosion
    ok = ~np.ma.getmaskarray(a)
    ok = binary_erosion(ok, iterations=1)   # drop edge pixels (colour blending)
    v = np.asarray(a.filled(0)[ok], dtype=float)
    c = small[ok].astype(float) / 255.0
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    sel = (v >= lo) & (v <= hi)
    v, c = v[sel], c[sel]
    n = 256
    idx = np.clip(((v - lo) / (hi - lo) * n).astype(int), 0, n - 1)
    cols_ = np.full((n, 3), np.nan)
    for b in range(n):
        m = idx == b
        if m.any():
            cols_[b] = np.median(c[m], axis=0)
    good = ~np.isnan(cols_[:, 0])
    if not good.all():                       # interpolate any empty value bins
        xi = np.arange(n)
        for ch in range(3):
            cols_[~good, ch] = np.interp(xi[~good], xi[good], cols_[good, ch])
    return ListedColormap(np.clip(cols_, 0, 1)), lo, hi


def full_ramp(name, crop):
    """The ArcGIS ramp itself: colour as a function of value over the full
    data range (independent of the stretch Pro applied)."""
    a = common_window(name)
    small = np.asarray(crop.resize((a.shape[1], a.shape[0]), Image.NEAREST))
    from scipy.ndimage import binary_erosion
    ok = ~np.ma.getmaskarray(a)
    ok = binary_erosion(ok, iterations=1)   # drop edge pixels (colour blending)
    v = np.asarray(a.filled(0)[ok], dtype=float)
    c = small[ok].astype(float) / 255.0
    vmin, vmax = np.percentile(v, 0.05), np.percentile(v, 99.95)
    n = 256
    idx = np.clip(((v - vmin) / (vmax - vmin) * n).astype(int), 0, n - 1)
    cols_ = np.full((n, 3), np.nan)
    for b in range(n):
        m = idx == b
        if m.sum() > 3:
            cols_[b] = np.median(c[m], axis=0)
    good = ~np.isnan(cols_[:, 0])
    if good.sum() < 8:
        return None, None
    xi = np.arange(n)
    for ch in range(3):
        cols_[~good, ch] = np.interp(xi[~good], xi[good], cols_[good, ch])
    w = 17                                     # smooth the sampled ramp
    k = np.ones(w) / w
    for ch in range(3):
        cols_[:, ch] = np.convolve(np.pad(cols_[:, ch], w // 2, mode="edge"),
                                   k, mode="valid")
    return ListedColormap(np.clip(cols_, 0, 1)), (vmin, vmax)


def contrast(cmap, lo, hi, rng):
    """Colour distance spanned between lo and hi within the full ramp."""
    f0 = (lo - rng[0]) / (rng[1] - rng[0])
    f1 = (hi - rng[0]) / (rng[1] - rng[0])
    c0 = np.array(cmap(np.clip(f0, 0, 1))[:3])
    c1 = np.array(cmap(np.clip(f1, 0, 1))[:3])
    mid = np.array(cmap(np.clip((f0 + f1) / 2, 0, 1))[:3])
    return max(np.abs(c0 - c1).sum(), np.abs(c0 - mid).sum(),
               np.abs(mid - c1).sum())


def rerender(name, cmap):
    """Re-draw a panel from the raster with Pro's ramp over the 2-98% range."""
    with rasterio.open(os.path.join(RAS, name + ".tif")) as r:
        a = r.read(1, masked=True)[R0:R1, C0:C1]
    lo, hi = np.percentile(a.compressed(), 2), np.percentile(a.compressed(), 98)
    f = np.clip((np.asarray(a.filled(lo), float) - lo) / (hi - lo), 0, 1)
    rgba = cmap(f)
    rgba[..., 3] = (~a.mask).astype(float)
    return (rgba * 255).astype(np.uint8), lo, hi


def fmt(v, kind):
    if kind == "sh":
        return f"{v:.4f}"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


# --- graticule crossings along the cropped extent ---
tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
NS = 400
FR = np.linspace(0, 1, NS + 1)


def crossings(vals, targets):
    out = []
    for t in targets:
        for i in range(len(vals) - 1):
            a, b = vals[i], vals[i + 1]
            if (a - t) * (b - t) <= 0 and a != b:
                f = (t - a) / (b - a)
                out.append((FR[i] + f * (FR[i + 1] - FR[i]), t))
                break
    return out


lon_b = [tr.transform(X0 + f * (X1 - X0), Y0)[0] for f in FR]
lon_t = [tr.transform(X0 + f * (X1 - X0), Y1)[0] for f in FR]
lat_l = [tr.transform(X0, Y1 - f * (Y1 - Y0))[1] for f in FR]
lat_r = [tr.transform(X1, Y1 - f * (Y1 - Y0))[1] for f in FR]
def _line(pairs):
    pts = []
    for lo_, la_ in pairs:
        x, y = inv.transform(lo_, la_)
        fx = (x - X0) / (X1 - X0)
        fy = (y - Y0) / (Y1 - Y0)
        pts.append((fx, fy))
    return np.array(pts)


inv = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
GRAT_LINES = []
for _lon in range(-140, -108, 5):
    GRAT_LINES.append(_line([(_lon, la) for la in np.linspace(42, 68, 120)]))
for _lat in range(45, 66, 5):
    GRAT_LINES.append(_line([(lo_, _lat) for lo_ in np.linspace(-146, -108, 120)]))

LONS = list(range(-140, -108, 5))
LATS = list(range(45, 66, 5))
CB = crossings(lon_b, LONS)
CT = crossings(lon_t, LONS)
CL = crossings(lat_l, LATS)
CR = crossings(lat_r, LATS)
print(f"graticule: {len(CB)} meridians (bottom), {len(CL)} parallels (left)")

# ----------------------------- layout (cm) -----------------------------
FIG_W = 16.5
COLS, ROWS = 5, 4
L_MARG, R_MARG, T_MARG, B_MARG = 0.78, 0.22, 0.10, 0.10
GAP_X, GAP_Y = 0.16, 0.16
TITLE_H, LONLAB_H, CBAR_H, CBLAB_H = 0.72, 0.34, 0.20, 0.36

PW = (FIG_W - L_MARG - R_MARG - (COLS - 1) * GAP_X) / COLS
PH = PW / ASPECT
ROW_H = TITLE_H + PH + LONLAB_H + CBAR_H + CBLAB_H
FIG_H = T_MARG + ROWS * ROW_H + (ROWS - 1) * GAP_Y + B_MARG
print(f"panel {PW:.2f} x {PH:.2f} cm | figure {FIG_W:.1f} x {FIG_H:.1f} cm")

FS_TITLE, FS_TICK, FS_GRAT, FS_LEG, FS_NOTE = 7.6, 7.0, 7.0, 8.0, 7.4
fig = plt.figure(figsize=(FIG_W / 2.54, FIG_H / 2.54))


def ax_at(x_cm, y_cm, w_cm, h_cm):
    """Axes from top-left in cm."""
    return fig.add_axes([x_cm / FIG_W, 1 - (y_cm + h_cm) / FIG_H,
                         w_cm / FIG_W, h_cm / FIG_H])


def wrap_title(txt, maxw_cm, fs):
    """Split into at most two lines that fit maxw_cm."""
    approx = maxw_cm / (fs * 0.0353 * 0.47)      # chars that fit
    if len(txt) <= approx:
        return txt
    words = txt.split()
    line, out = "", []
    for w in words:
        t = (line + " " + w).strip()
        if len(t) <= approx or not line:
            line = t
        else:
            out.append(line)
            line = w
    out.append(line)
    return "\n".join(out[:2])


last_map_in_col = {}
for i in range(len(PANELS)):
    last_map_in_col[i % COLS] = i

for i, (name, title, kind) in enumerate(PANELS):
    r, c = divmod(i, COLS)
    x = L_MARG + c * (PW + GAP_X)
    y = T_MARG + r * (ROW_H + GAP_Y)
    crop = crop_panel(name)
    img = np.asarray(crop)

    axm = ax_at(x, y + TITLE_H, PW, PH)
    axm.imshow(img, interpolation="lanczos", aspect="auto")
    axm.set_xticks([]); axm.set_yticks([])
    for s in axm.spines.values():
        s.set_linewidth(0.7); s.set_color("0.15")
    ttl = f"({chr(97 + i)}) {title}"
    axm.set_title(wrap_title(ttl, PW + GAP_X, FS_TITLE), fontsize=FS_TITLE,
                  fontweight="bold", pad=3, loc="left", linespacing=1.25)

    # graticule lines inside the panel
    for gl in GRAT_LINES:
        axm.plot(gl[:, 0], gl[:, 1], transform=axm.transAxes, color="0.45",
                 lw=0.25, alpha=0.55, zorder=3, clip_on=True)

    # graticule: ticks all round, labels on outer panels
    tk = 0.055 / PW                                   # tick length (axes frac)
    show_lon = (i == last_map_in_col[c])
    show_lat = (c == 0)
    for f, v in CB:
        axm.plot([f, f], [0, -tk * ASPECT], transform=axm.transAxes,
                 color="0.15", lw=0.7, clip_on=False)
        if show_lon and 0.12 < f < 0.88:
            axm.text(f, -tk * ASPECT - 0.02, f"{abs(v):.0f}°W",
                     transform=axm.transAxes, ha="center", va="top",
                     fontsize=FS_GRAT)
    for f, v in CT:
        axm.plot([f, f], [1, 1 + tk * ASPECT], transform=axm.transAxes,
                 color="0.15", lw=0.7, clip_on=False)
    for f, v in CL:
        axm.plot([0, -tk], [1 - f, 1 - f], transform=axm.transAxes,
                 color="0.15", lw=0.7, clip_on=False)
        if show_lat and 0.06 < f < 0.94:
            axm.text(-tk - 0.03, 1 - f, f"{v:.0f}°N",
                     transform=axm.transAxes, ha="right", va="center",
                     fontsize=FS_GRAT)
    for f, v in CR:
        axm.plot([1, 1 + tk], [1 - f, 1 - f], transform=axm.transAxes,
                 color="0.15", lw=0.7, clip_on=False)

    # colour bar / class legend
    if kind == "lulc":
        continue
    cmap, rng = full_ramp(name, crop)
    lo, hi = rng
    if kind == "km":
        lo, hi = lo / 1000.0, hi / 1000.0
    axc = ax_at(x, y + TITLE_H + PH + LONLAB_H, PW, CBAR_H)
    cb = ColorbarBase(axc, cmap=cmap, norm=Normalize(lo, hi),
                      orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    if kind == "asp":
        cb = ColorbarBase(axc, cmap=cmap, norm=Normalize(0, 360),
                          orientation="horizontal")
        cb.outline.set_linewidth(0.6)
        cb.set_ticks([0, 90, 180, 270, 360])
        cb.set_ticklabels(["N", "E", "S", "W", "N"])
    else:
        ticks = [lo, (lo + hi) / 2, hi]
        cb.set_ticks(ticks)
        cb.set_ticklabels([fmt(t, kind) for t in ticks])
    axc.tick_params(labelsize=FS_TICK, length=2, width=0.6, pad=1.5)
    lab = axc.get_xticklabels()
    lab[0].set_ha("left"); lab[-1].set_ha("right")
    print(f"  {name}: legend {fmt(lo, kind)} .. {fmt(hi, kind)}")

# ---- spare cell 19: LULC legend ----
r, c = divmod(18, COLS)
x = L_MARG + c * (PW + GAP_X)
y = T_MARG + r * (ROW_H + GAP_Y)
axl = ax_at(x, y + TITLE_H, PW + GAP_X, PH + LONLAB_H)
axl.axis("off")
axl.text(0, 1.0, "LULC classes", fontsize=FS_LEG, fontweight="bold",
         transform=axl.transAxes, va="top")
for k, (v, lab_, col_) in enumerate(LULC_DEF):
    yy = 0.86 - k * 0.108
    axl.add_patch(Rectangle((0.02, yy - 0.055), 0.13, 0.075, facecolor=col_,
                            edgecolor="0.2", lw=0.6,
                            transform=axl.transAxes, clip_on=False))
    axl.text(0.20, yy - 0.018, lab_, fontsize=FS_LEG - 0.6,
             transform=axl.transAxes, va="center")

# ---- spare cell 20: ArcGIS north arrow + scale bar, then notes ----
r, c = divmod(19, COLS)
x = L_MARG + c * (PW + GAP_X)
y = T_MARG + r * (ROW_H + GAP_Y)
axf = ax_at(x + PW * 0.14, y + TITLE_H, PW * 0.40, PH * 0.40)
axf.set_xlim(0, 1); axf.set_ylim(0, 1); axf.axis("off")
axf.text(0.5, 0.99, "N", ha="center", va="top", fontsize=FS_LEG + 2,
         fontweight="bold")
axf.add_patch(plt.Polygon([[0.5, 0.72], [0.30, 0.06], [0.5, 0.20]],
                          closed=True, facecolor="black", edgecolor="black",
                          lw=0.4))
axf.add_patch(plt.Polygon([[0.5, 0.72], [0.70, 0.06], [0.5, 0.20]],
                          closed=True, facecolor="white", edgecolor="black",
                          lw=0.6))
# exact scale bar: panel width represents (C1-C0) cells x 1.5 km
KM_PER_PANEL = (C1 - C0) * abs(TR.a) / 1000.0
axs = ax_at(x + PW * 0.02, y + TITLE_H + PH * 0.44, PW * 0.96, PH * 0.22)
axs.set_xlim(-150, 830); axs.set_ylim(0, 1); axs.axis("off")
BAR = 500.0
axs.add_patch(Rectangle((0, 0.52), BAR / 2, 0.20, facecolor="black",
                        edgecolor="black", lw=0.6))
axs.add_patch(Rectangle((BAR / 2, 0.52), BAR / 2, 0.20, facecolor="white",
                        edgecolor="black", lw=0.6))
axs.text(0, 0.40, "0", ha="center", va="top", fontsize=FS_NOTE - 0.6)
axs.text(BAR / 2, 0.40, "250", ha="center", va="top", fontsize=FS_NOTE - 0.6)
axs.text(BAR + 55, 0.40, "500 km", ha="center", va="top",
         fontsize=FS_NOTE - 0.6)
axn = ax_at(x, y + TITLE_H + PH * 0.70, PW + GAP_X, PH * 0.30 + LONLAB_H + CBAR_H + CBLAB_H)
axn.axis("off")
axn.text(0, 1.0, "Grid: 1.5 km cells\nGraticule: WGS 84",
         fontsize=FS_NOTE, transform=axn.transAxes, va="top", linespacing=1.5)

os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(OUT + ".png", dpi=600, facecolor="white")
fig.savefig(OUT + ".tif", dpi=600, facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
im = Image.open(OUT + ".png")
print(f"\nWrote {OUT}.png / .tif  -> {im.size[0]}x{im.size[1]} px, "
      f"{FIG_W:.1f} x {FIG_H:.1f} cm at 600 dpi (font: {SERIF})")
