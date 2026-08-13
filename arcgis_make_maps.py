# ArcGIS Pro driver v3: 18 conditioning-factor maps (native Pro symbology),
# ArcGIS-native north arrow + scale bar, exact labelled graticule frames,
# Times New Roman, 600 dpi. Pure arcpy + Pillow (no pyproj/rasterio needed).
# Skips maps already exported, so reruns jump straight to compositing.
#
# RUN (ArcGIS Pro -> View -> Python window):
#   exec(open(r"C:\Users\saadz\Documents\wildfire-bc-bilstm-pso\arcgis_make_maps.py").read())
#
# The project is NEVER saved. Outputs:
#   figs\arcgis_exports\<name>.png        (18 maps + furniture.png, 600 dpi)
#   figs\Fig_predictors_arcgis.png/.tif   (composite, 600 dpi)
import os
import arcpy

REPO = r"C:\Users\saadz\Documents\wildfire-bc-bilstm-pso"
SRC = os.path.join(REPO, "revision_c8c11", "01_Input_Data", "rasters")
OUTD = os.path.join(REPO, "figs", "arcgis_exports")
os.makedirs(OUTD, exist_ok=True)
SKIP_EXISTING = True

PANELS = [
    ("Elevation",           "Elevation (m)",              ["Elevation #1"],                                    "seq"),
    ("Slope",               "Slope (\u00b0)",             ["Viridis"],                                         "seq"),
    ("Aspect",              "Aspect (\u00b0)",            ["Aspect"],                                          "seq"),
    ("TWI",                 "TWI (\u2013)",               ["Cividis", "Viridis"],                              "seq"),
    ("Profile_Curvature",   "Profile Curvature (\u2013)", ["Red-Blue (Continuous)", "*Red*Blue*"],             "div"),
    ("Plan_Curvature",      "Plan Curvature (\u2013)",    ["Red-Blue (Continuous)", "*Red*Blue*"],             "div"),
    ("NDVI",                "NDVI (\u2013)",              ["*Brown to Blue Green*", "*Vegetation*", "Yellow-Green (Continuous)", "*Yellow*Green*"], "seq"),
    ("Distance_rivers",     "Distance to Rivers (km)",    ["Magma"],                                           "seq"),
    ("Distance_roads",      "Distance to Roads (km)",     ["Magma"],                                           "seq"),
    ("Distance_households", "Distance to Households (km)", ["Magma"],                                          "seq"),
    ("Max_Temperature",     "Max Temperature (\u00b0C)",  ["Plasma"],                                          "seq"),
    ("Precipitation",       "Precipitation (mm)",         ["Yellow-Green-Blue (Continuous)", "*Yellow*Blue*", "Bathymetry #2"], "seq"),
    ("Soil_Moisture",       "Soil Moisture (mm)",         ["Blues (Continuous)", "*Blues*"],                   "seq"),
    ("WS",                  "Wind Speed (m s\u207b\u00b9)", ["Viridis"],                                       "seq"),
    ("Relative_Humidity",   "Specific Humidity (kg kg\u207b\u00b9)", ["Yellow-Green-Blue (Continuous)", "*Yellow*Blue*", "Bathymetry #2"], "seq"),
    ("DSI",                 "PDSI (\u2013)",              ["Red-Blue (Continuous)", "*Red*Blue*"],             "div"),
    ("AET",                 "AET (mm)",                   ["Yellow-Green (Continuous)", "*Yellow*Green*"],     "seq"),
    ("LULC",                "Land Use / Land Cover",      None,                                                "lulc"),
]

LULC_CLASSES = {1: ("Water", (65, 155, 223)), 2: ("Trees", (57, 125, 73)),
                4: ("Flooded Vegetation", (122, 135, 198)), 5: ("Crops", (228, 150, 53)),
                7: ("Built Area", (196, 40, 27)), 8: ("Bare Ground", (165, 155, 143)),
                9: ("Snow/Ice", (214, 239, 255)), 11: ("Rangeland", (198, 185, 138))}

PAGE_W, PAGE_H, M = 18.0, 22.0, 0.4     # layout page and margin, cm
DPI = 600

aprx = arcpy.mp.ArcGISProject("CURRENT")
print("ArcGIS Pro", arcpy.GetInstallInfo()["Version"], "- v3 run")

for mp_ in list(aprx.listMaps("m_*")):
    try:
        aprx.deleteItem(mp_)
    except Exception:
        pass
for ly_ in list(aprx.listLayouts("L_*")):
    try:
        aprx.deleteItem(ly_)
    except Exception:
        pass


def find_ramp(cands):
    for c in cands or []:
        r = aprx.listColorRamps(c)
        if r:
            return r[0]
    return None


def build_map(mname, rname, ramps, kind):
    m = aprx.createMap(mname, "Map")
    for bl in list(m.listLayers()):
        m.removeLayer(bl)
    lyr = m.addDataFromPath(os.path.join(SRC, rname + ".tif"))
    sym = lyr.symbology
    if kind == "lulc":
        sym.updateColorizer("RasterUniqueValueColorizer")
        try:
            for grp in sym.colorizer.groups:
                for itm in grp.items:
                    v = int(float(itm.values[0]))
                    if v in LULC_CLASSES:
                        lab, rgb = LULC_CLASSES[v]
                        itm.label = lab
                        itm.color = {"RGB": [rgb[0], rgb[1], rgb[2], 100]}
        except Exception as e:
            print("  LULC colouring note:", e)
    else:
        sym.updateColorizer("RasterStretchColorizer")
        sym.colorizer.stretchType = "PercentClip"
        try:
            sym.colorizer.minPercent = 2.0
            sym.colorizer.maxPercent = 2.0
        except Exception:
            pass
        ramp = find_ramp(ramps)
        if ramp is not None:
            sym.colorizer.colorRamp = ramp
        else:
            print("  ramp not found, default used:", ramps)
    lyr.symbology = sym
    return m, lyr


def frame_env():
    return arcpy.Polygon(arcpy.Array([
        arcpy.Point(M, M), arcpy.Point(M, PAGE_H - M),
        arcpy.Point(PAGE_W - M, PAGE_H - M), arcpy.Point(PAGE_W - M, M)]))


exported = []
for name, title, ramps, kind in PANELS:
    tif = os.path.join(SRC, name + ".tif")
    png = os.path.join(OUTD, name + ".png")
    if not os.path.exists(tif):
        print("MISSING:", tif)
        continue
    if SKIP_EXISTING and os.path.exists(png):
        exported.append((name, title, kind, png))
        print("  exists, skipping export:", name)
        continue
    try:
        m, lyr = build_map("m_" + name, name, ramps, kind)
        lay = aprx.createLayout(PAGE_W, PAGE_H, "CENTIMETER", "L_" + name)
        mf = lay.createMapFrame(frame_env(), m, "mf_" + name)
        mf.camera.setExtent(mf.getLayerExtent(lyr, False, True))
        lay.exportToPNG(png, resolution=DPI)
        exported.append((name, title, kind, png))
        print("  exported", name)
    except Exception as e:
        print("  FAILED", name, "->", repr(e))

# ---- furniture: Pro-native north arrow + scale bar on a scale-true frame ----
furn_png = os.path.join(OUTD, "furniture.png")
if not (SKIP_EXISTING and os.path.exists(furn_png)):
    try:
        mF, lyrF = build_map("m_FURN", "Elevation", ["Elevation #1"], "seq")
        layF = aprx.createLayout(PAGE_W, PAGE_H, "CENTIMETER", "L_FURNITURE")
        mfF = layF.createMapFrame(frame_env(), mF, "mf_FURN")
        mfF.camera.setExtent(mfF.getLayerExtent(lyrF, False, True))
        na_items = (aprx.listStyleItems("ArcGIS 2D", "North_Arrow", "ArcGIS North 1")
                    or aprx.listStyleItems("ArcGIS 2D", "North_Arrow", "*North 1*")
                    or aprx.listStyleItems("ArcGIS 2D", "North_Arrow", "*"))
        sb_items = (aprx.listStyleItems("ArcGIS 2D", "Scale_bar", "Scale Line 1")
                    or aprx.listStyleItems("ArcGIS 2D", "Scale_bar", "*Scale Line*")
                    or aprx.listStyleItems("ArcGIS 2D", "Scale_bar", "*"))
        na = layF.createMapSurroundElement(arcpy.Point(13.5, 5.4), "NORTH_ARROW", mfF,
                                           na_items[0] if na_items else None, "furn_na")
        na.elementWidth = 1.4
        sb = layF.createMapSurroundElement(arcpy.Point(11.0, 2.3), "SCALE_BAR", mfF,
                                           sb_items[0] if sb_items else None, "furn_sb")
        sb.elementWidth = 6.0
        sb.elementHeight = 1.1
        try:
            cim = sb.getDefinition("V3")
            def set_font(obj, depth=0):
                if depth > 8:
                    return
                if hasattr(obj, "__dict__"):
                    for k, v in vars(obj).items():
                        if k == "fontFamilyName":
                            setattr(obj, k, "Times New Roman")
                        else:
                            set_font(v, depth + 1)
                elif isinstance(obj, list):
                    for it in obj:
                        set_font(it, depth + 1)
            set_font(cim)
            sb.setDefinition(cim)
        except Exception as e:
            print("  scale-bar font note:", e)
        layF.exportToPNG(furn_png, resolution=DPI)
        print("  exported furniture")
    except Exception as e:
        furn_png = None
        print("  FURNITURE FAILED ->", repr(e))
else:
    print("  exists, skipping furniture export")

print(f"{len(exported)}/18 maps ready. Assembling composite...")

# ------------------- composite (Pillow; transforms via arcpy) -------------------
from PIL import Image, ImageDraw, ImageFont

desc = arcpy.Describe(os.path.join(SRC, "Elevation.tif"))
E = desc.extent
SR_IN = desc.spatialReference
SR_WGS = arcpy.SpatialReference(4326)

# displayed extent inside the map frame (fit extent, preserve aspect)
fw_cm, fh_cm = PAGE_W - 2 * M, PAGE_H - 2 * M
ew, eh = E.XMax - E.XMin, E.YMax - E.YMin
if ew / eh < fw_cm / fh_cm:
    disp_h = eh
    disp_w = eh * fw_cm / fh_cm
else:
    disp_w = ew
    disp_h = ew * fh_cm / fw_cm
cx, cy = (E.XMin + E.XMax) / 2, (E.YMin + E.YMax) / 2
DX0, DX1 = cx - disp_w / 2, cx + disp_w / 2
DY0, DY1 = cy - disp_h / 2, cy + disp_h / 2


def to_wgs(x, y):
    p = arcpy.PointGeometry(arcpy.Point(x, y), SR_IN).projectAs(SR_WGS).firstPoint
    return p.X, p.Y


def crossings_1d(fractions, values, targets):
    out = []
    for t in targets:
        for i in range(len(values) - 1):
            a, b = values[i], values[i + 1]
            if (a - t) * (b - t) <= 0 and a != b:
                f = (t - a) / (b - a)
                out.append((fractions[i] + f * (fractions[i + 1] - fractions[i]), t))
                break
    return out


# precompute graticule crossings ONCE (frame-edge fractions, shared by panels)
NS = 160
FR = [i / NS for i in range(NS + 1)]
LONS = list(range(-140, -108, 4))
LATS = list(range(46, 66, 2))
lon_bottom = crossings_1d(FR, [to_wgs(DX0 + f * (DX1 - DX0), DY0)[0] for f in FR], LONS)
lon_top    = crossings_1d(FR, [to_wgs(DX0 + f * (DX1 - DX0), DY1)[0] for f in FR], LONS)
lat_left   = crossings_1d(FR, [to_wgs(DX0, DY1 - f * (DY1 - DY0))[1] for f in FR], LATS)
lat_right  = crossings_1d(FR, [to_wgs(DX1, DY1 - f * (DY1 - DY0))[1] for f in FR], LATS)
print(f"graticule: {len(lon_bottom)} meridians, {len(lat_left)} parallels")

CELL_W = 2000
GRAT = 130
TITLE_H = 150
PAD = 40
COLS, ROWS = 5, 4

TNR = r"C:\Windows\Fonts\times.ttf"
TNRB = r"C:\Windows\Fonts\timesbd.ttf"
F_LAB = ImageFont.truetype(TNR, 44)
F_NOTE = ImageFont.truetype(TNR, 52)
F_HDR = ImageFont.truetype(TNRB, 58)


def shrink(dr, text, start, maxw):
    size = start
    f = ImageFont.truetype(TNRB, size)
    while dr.textlength(text, font=f) > maxw and size > 30:
        size -= 2
        f = ImageFont.truetype(TNRB, size)
    return f


cells = []
for name, title, kind, png in exported:
    im = Image.open(png).convert("RGB")
    sc = CELL_W / im.size[0]
    im = im.resize((CELL_W, int(im.size[1] * sc)), Image.LANCZOS)
    cells.append((name, title, im))
cell_h = max(im.size[1] for _, _, im in cells)

unit_w = CELL_W + GRAT * 2
unit_h = TITLE_H + cell_h + GRAT
page_w = COLS * unit_w + (COLS + 1) * PAD
page_h = ROWS * unit_h + (ROWS + 1) * PAD
page = Image.new("RGB", (page_w, page_h), "white")
d = ImageDraw.Draw(page)

mx_f, my_f = M / PAGE_W, M / PAGE_H      # frame inset as fraction of the page


def draw_graticule(x, y, im_w, im_h):
    fx0 = x + im_w * mx_f
    fx1 = x + im_w * (1 - mx_f)
    fy0 = y + im_h * my_f
    fy1 = y + im_h * (1 - my_f)
    d.rectangle([fx0, fy0, fx1, fy1], outline=(50, 50, 50), width=4)
    for f, lonv in lon_bottom:
        X = fx0 + f * (fx1 - fx0)
        d.line([(X, fy1), (X, fy1 + 16)], fill="black", width=4)
        lab = f"{abs(lonv):.0f}\u00b0W"
        w = d.textlength(lab, font=F_LAB)
        d.text((X - w / 2, fy1 + 20), lab, font=F_LAB, fill="black")
    for f, lonv in lon_top:
        X = fx0 + f * (fx1 - fx0)
        d.line([(X, fy0 - 16), (X, fy0)], fill="black", width=4)
    for f, latv in lat_left:
        Y = fy0 + f * (fy1 - fy0)
        d.line([(fx0 - 16, Y), (fx0, Y)], fill="black", width=4)
        lab = f"{latv:.0f}\u00b0N"
        w = d.textlength(lab, font=F_LAB)
        d.text((fx0 - 22 - w, Y - 24), lab, font=F_LAB, fill="black")
    for f, latv in lat_right:
        Y = fy0 + f * (fy1 - fy0)
        d.line([(fx1, Y), (fx1 + 16, Y)], fill="black", width=4)


for i, (name, title, im) in enumerate(cells):
    r, c = divmod(i, COLS)
    ux = PAD + c * (unit_w + PAD)
    uy = PAD + r * (unit_h + PAD)
    x = ux + GRAT
    y = uy + TITLE_H
    ttl = f"({chr(97 + i)}) {title}"
    f = shrink(d, ttl, 64, unit_w - 20)
    d.text((ux + GRAT, uy + 26), ttl, font=f, fill="black")
    page.paste(im, (x, y))
    draw_graticule(x, y, im.size[0], im.size[1])

# spare cell 19: LULC legend
sx = PAD + 3 * (unit_w + PAD) + GRAT
sy = PAD + 3 * (unit_h + PAD) + TITLE_H + 40
d.text((sx, sy), "LULC classes", font=F_HDR, fill="black")
sw = 64
for k, (v, (lab, rgb)) in enumerate(sorted(LULC_CLASSES.items())):
    yy = sy + 110 + k * 105
    d.rectangle([sx, yy, sx + sw, yy + sw], fill=rgb, outline="black", width=3)
    d.text((sx + sw + 34, yy + 4), lab, font=F_NOTE, fill="black")

# spare cell 20: ArcGIS-native north arrow + scale bar (cropped from furniture)
nx = PAD + 4 * (unit_w + PAD) + GRAT
ny = sy
if furn_png and os.path.exists(furn_png):
    fu = Image.open(furn_png).convert("RGB")
    x0 = int(round(10.2 / PAGE_W * fu.size[0]))
    x1 = int(round(17.9 / PAGE_W * fu.size[0]))
    y0 = int(round((1 - 6.6 / PAGE_H) * fu.size[1]))
    y1 = int(round((1 - 1.0 / PAGE_H) * fu.size[1]))
    crop = fu.crop((x0, y0, x1, y1))
    tw = unit_w - GRAT
    crop = crop.resize((tw, int(crop.size[1] * tw / crop.size[0])), Image.LANCZOS)
    page.paste(crop, (nx, ny))
    ny = ny + crop.size[1] + 40
d.text((nx, ny),
       "1.5 km analysis grid\nGraticule: WGS 84\nStretch: 2\u201398% percent clip",
       font=F_NOTE, fill="black")

comp = os.path.join(REPO, "figs", "Fig_predictors_arcgis.png")
page.save(comp, dpi=(DPI, DPI))
page.save(comp.replace(".png", ".tif"), dpi=(DPI, DPI), compression="tiff_lzw")
print("DONE.")
print("  composite:", comp, f"({page.size[0]}x{page.size[1]} px)")
