"""Journal-standard conditioning-factor figure: the eighteen 1.5 km predictor
rasters in a 5 x 4 portrait grid with panel letters, rounded legends,
perceptually uniform colour ramps, LULC class legend, north arrow and scale
bar.

Rasters are read from data/rasters/ (shipped in data.zip); where absent, the
staged training workspace (revision_c8c11/01_Input_Data/rasters/) is used.

Output: figs/Fig_predictors_grid.{png,pdf,tif}   (600 dpi PNG and TIFF)
Run from the repository root:  python make_fig_predictors.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, FancyArrow
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
SRC1 = os.path.join(HERE, "data", "rasters")
SRC2 = os.path.join(HERE, "revision_c8c11", "01_Input_Data", "rasters")
ALIAS = {"Specific_Humidity": "Relative_Humidity"}   # legacy local file name
OUT = os.path.join(HERE, "figs", "Fig_predictors_grid")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "font.weight": "bold",
    "axes.titlesize": 7.6, "axes.titleweight": "bold",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

#          name                 title                       unit        cmap        kind
PANELS = [("Elevation",         "Elevation",                "m",        "terrain",  "seq"),
          ("Slope",             "Slope",                    "°",        "viridis",  "seq"),
          ("Aspect",            "Aspect",                   "°",        "twilight", "aspect"),
          ("TWI",               "TWI",                      "–",        "cividis",  "seq"),
          ("Profile_Curvature", "Profile Curvature",        "–",        "RdBu_r",   "div"),
          ("Plan_Curvature",    "Plan Curvature",           "–",        "RdBu_r",   "div"),
          ("NDVI",              "NDVI",                     "–",        "BrBG",     "div"),
          ("Distance_rivers",   "Distance to Rivers",       "km",       "magma",    "km"),
          ("Distance_roads",    "Distance to Roads",        "km",       "magma",    "km"),
          ("Distance_households", "Distance to Households", "km",       "magma",    "km"),
          ("Max_Temperature",   "Max Temperature",          "°C",       "plasma",   "seq"),
          ("Precipitation",     "Precipitation",            "mm",       "YlGnBu",   "seq"),
          ("Soil_Moisture",     "Soil Moisture",            "mm",       "PuBu",     "seq"),
          ("WS",                "Wind Speed",               "m s$^{-1}$", "viridis", "seq"),
          ("Specific_Humidity", "Specific Humidity",        "kg kg$^{-1}$", "YlGnBu", "sh"),
          ("DSI",               "PDSI",                     "–",        "RdBu",     "div"),
          ("AET",               "AET",                      "mm",       "YlGn",     "seq"),
          ("LULC",              "Land Use / Land Cover",    "",         None,       "lulc")]

LULC_DEF = [(1, "Water", "#419BDF"), (2, "Trees", "#397D49"),
            (4, "Flooded Vegetation", "#7A87C6"), (5, "Crops", "#E49635"),
            (7, "Built Area", "#C4281B"), (8, "Bare Ground", "#A59B8F"),
            (9, "Snow/Ice", "#D6EFFF"), (11, "Rangeland", "#C6B98A")]


def load(name):
    for base, nm in [(SRC1, name), (SRC2, name), (SRC2, ALIAS.get(name, name))]:
        p = os.path.join(base, nm + ".tif")
        if os.path.exists(p):
            with rasterio.open(p) as r:
                return r.read(1, masked=True), r.transform
    raise FileNotFoundError(name)


def fmt(v, kind):
    if kind == "sh":
        return f"{v:.4f}"
    a = abs(v)
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def main():
    fig = plt.figure(figsize=(9.6, 12.9))
    outer = gridspec.GridSpec(4, 5, figure=fig,
                              left=0.012, right=0.988, top=0.978, bottom=0.012,
                              hspace=0.16, wspace=0.06)
    mask_outline = None
    for i, (name, title, unit, cmap, kind) in enumerate(PANELS):
        r, c = divmod(i, 5)
        cell = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[r, c], height_ratios=[1.0, 0.055], hspace=0.06)
        ax = fig.add_subplot(cell[0])
        cax = fig.add_subplot(cell[1])
        arr, transform = load(name)
        if kind == "km":
            arr = arr / 1000.0
        if mask_outline is None:
            from scipy.ndimage import binary_fill_holes
            mask_outline = binary_fill_holes(~arr.mask).astype(float)
        letter = chr(97 + i)
        ttl = f"({letter}) {title}" + (f" ({unit})" if unit else "")
        ax.set_title(ttl, pad=2.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("0.4")

        if kind == "lulc":
            vals = [v for v, _, _ in LULC_DEF]
            cmap_l = ListedColormap([cc for _, _, cc in LULC_DEF])
            bounds = [v - 0.5 for v in vals] + [vals[-1] + 0.5]
            # map raster classes onto legend indices
            data = np.asarray(arr.filled(0)).astype(int)
            idx = np.full(data.shape, -1, dtype=float)
            for k, (v, _, _) in enumerate(LULC_DEF):
                idx[data == v] = k
            idx = np.ma.masked_less(idx, 0)
            ax.imshow(idx, cmap=cmap_l, vmin=-0.5, vmax=len(vals) - 0.5,
                      interpolation="nearest")
            cax.axis("off")
        elif kind == "aspect":
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=360, interpolation="nearest")
            cb = fig.colorbar(plt.cm.ScalarMappable(
                norm=plt.Normalize(0, 360), cmap=cmap), cax=cax,
                orientation="horizontal", ticks=[0, 90, 180, 270, 360])
            cb.ax.set_xticklabels(["N", "E", "S", "W", "N"], fontsize=6)
            cb.outline.set_linewidth(0.4)
        else:
            comp = arr.compressed()
            if kind == "div":
                m = float(np.percentile(np.abs(comp), 98))
                vmin, vmax, ticks = -m, m, [-m, 0, m]
            else:
                vmin = float(np.percentile(comp, 2))
                vmax = float(np.percentile(comp, 98))
                if kind == "km" or name == "Elevation":
                    vmin = 0.0
                ticks = [vmin, (vmin + vmax) / 2, vmax]
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
            cb.ax.set_xticklabels([fmt(t, kind) for t in ticks], fontsize=6)
            labs = cb.ax.get_xticklabels()
            labs[0].set_horizontalalignment("left")
            labs[-1].set_horizontalalignment("right")
            cb.outline.set_linewidth(0.4)
        # province outline
        ax.contour(mask_outline, levels=[0.5], colors="0.35", linewidths=0.35)

    # --- cell 19: LULC legend ---
    axL = fig.add_subplot(outer[3, 3]); axL.axis("off")
    patches = [Patch(facecolor=cc, edgecolor="0.3", linewidth=0.5, label=lb)
               for _, lb, cc in LULC_DEF]
    axL.legend(handles=patches, loc="center left", frameon=False, fontsize=7,
               title="LULC classes", title_fontsize=7.6, labelspacing=0.6,
               handlelength=1.2, borderaxespad=0)

    # --- cell 20: north arrow, scale bar, notes ---
    axN = fig.add_subplot(outer[3, 4]); axN.axis("off")
    axN.set_xlim(0, 1); axN.set_ylim(0, 1)
    axN.add_patch(FancyArrow(0.18, 0.60, 0, 0.22, width=0.035,
                             head_width=0.10, head_length=0.09,
                             length_includes_head=True, color="black"))
    axN.text(0.18, 0.90, "N", ha="center", va="bottom",
             fontsize=11, fontweight="bold")
    # scale bar: map width = ncols * 1.5 km; bar drawn at the same fraction
    arr0, _ = load("Elevation")
    map_km = arr0.shape[1] * 1.5
    frac = 500.0 / map_km
    x0 = 0.42
    axN.plot([x0, x0 + frac], [0.68, 0.68], color="black", lw=2)
    axN.plot([x0, x0], [0.655, 0.705], color="black", lw=1.2)
    axN.plot([x0 + frac, x0 + frac], [0.655, 0.705], color="black", lw=1.2)
    axN.text(x0 + frac / 2, 0.715, "500 km", ha="center", va="bottom",
             fontsize=7.5)
    axN.text(0.05, 0.44,
             "Equal-area projection;\n1.5 km analysis grid.\n"
             "All panels share the same\nextent and grid.\n"
             "Colour scales span the\n2\u201398% value range\n"
             "(diverging: \u00b198th pct of |x|).",
             fontsize=7.0, va="top", fontweight="normal", linespacing=1.4)

    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT + ".tif", dpi=600, bbox_inches="tight", pad_inches=0.05,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Wrote {OUT}.png / .pdf / .tif")


if __name__ == "__main__":
    main()
