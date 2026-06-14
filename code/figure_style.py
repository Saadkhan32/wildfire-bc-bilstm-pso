import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
OKABE_ITO = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
                 blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#9A9A9A")
def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
    })
def save_figure(fig, path_noext, png_dpi=200):
    fig.savefig(path_noext + ".pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path_noext + ".tif", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path_noext + ".png", dpi=png_dpi, bbox_inches="tight", pad_inches=0.05)
    im = Image.open(path_noext + ".tif").convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    Image.alpha_composite(bg, im).convert("RGB").save(
        path_noext + ".tif", dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)
