"""
Step 7b - Map figures for the presentation (PNG).
Inputs : factor rasters + boundary in data/processed/, susceptibility rasters in outputs/
Output : outputs/figures/07_study_area.png, 08_factor_maps.png,
         09_susceptibility_maps.png, 10_susceptibility_main.png
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
OUT = BASE_DIR / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CLASS_COLORS = ["#fff1b8", "#fdc56e", "#f68a3c", "#d9412b", "#8f1d1d"]   # YlOrRd, 5 ordered steps
CLASS_NAMES = ["Very Low", "Low", "Moderate", "High", "Very High"]
plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "text.color": INK})

def read(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        return a, src.transform

dem, T = read(PROC / "dem_utm.tif")
H, W = dem.shape
extent = [T.c, T.c + W * T.a, T.f + H * T.e, T.f]       # left, right, bottom, top (metres)
boundary = gpd.read_file(PROC / "bilaspur_boundary_utm.gpkg").to_crs("EPSG:32643")
inside = geometry_mask(boundary.geometry, out_shape=(H, W), transform=T, invert=True)

# Hillshade (sun from the north-west, 45 degrees up) for a 3-D terrain backdrop
gy, gx = np.gradient(np.nan_to_num(dem, nan=np.nanmean(dem)), 30)
slope_r = np.arctan(np.hypot(gx, gy))
aspect_r = np.arctan2(-gx, gy)
az, alt = np.deg2rad(315), np.deg2rad(45)
hill = np.sin(alt) * np.cos(slope_r) + np.cos(alt) * np.sin(slope_r) * np.cos(az - aspect_r)

# Crop every map to the district's bounding box (+1 km)
minx, miny, maxx, maxy = boundary.total_bounds
pad = 1000
view = [minx - pad, maxx + pad, miny - pad, maxy + pad]

def base(ax, alpha=1.0):
    ax.imshow(hill, cmap="gray", extent=extent, alpha=alpha, vmin=-0.2, vmax=1.0)
    boundary.boundary.plot(ax=ax, color=INK, linewidth=1.2)
    ax.set_xlim(view[0], view[1]); ax.set_ylim(view[2], view[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(False)

def scalebar(ax, km=5):
    x0, y0 = view[0] + 1500, view[2] + 1500
    ax.plot([x0, x0 + km * 1000], [y0, y0], color=INK, linewidth=3)
    ax.text(x0 + km * 500, y0 + 600, f"{km} km", ha="center", fontsize=10)
    ax.annotate("N", xy=(view[1] - 1800, view[3] - 1500), xytext=(view[1] - 1800, view[3] - 4500),
                ha="center", fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))

def masked(a):
    return np.where(inside, a, np.nan)

# ======== 7. Study area + inventory ========
pts = pd.read_csv(PROC / "model_data.csv")
fig, ax = plt.subplots(figsize=(9, 8))
base(ax)
s = pts[pts.label == 0]; l = pts[pts.label == 1]
ax.scatter(s.X, s.Y, s=14, color=BLUE, edgecolor=SURFACE, linewidth=0.6, label=f"Safe points ({len(s)})")
ax.scatter(l.X, l.Y, s=22, color=ORANGE, edgecolor=SURFACE, linewidth=0.6, label=f"GSI landslides ({len(l)})")
ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="#e4e3df")
ax.set_title("Bilaspur district: landslide inventory and safe points", loc="left")
scalebar(ax)
fig.savefig(FIG / "07_study_area.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("saved 07_study_area.png")

# ======== 8. Factor maps ========
factors = [("dem_utm.tif", "Elevation (m)", "Blues"),
           ("slope.tif", "Slope (°)", "Oranges"),
           ("ndvi_30m.tif", "NDVI", "Greens"),
           ("dist_road_v2.tif", "Distance to road (m)", "Purples"),
           ("dist_stream.tif", "Distance to stream (m)", "Blues"),
           ("aspect.tif", "Aspect (°, 0 = north)", "twilight")]   # circular data -> cyclic map
fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
for ax, (fname, title, cmap) in zip(axes.flat, factors):
    a, _ = read(PROC / fname)
    if fname == "aspect.tif":
        a[a < 0] = np.nan
    a = masked(a)
    lo, hi = np.nanpercentile(a, [2, 98])
    base(ax, alpha=0.6)
    im = ax.imshow(a, cmap=cmap, extent=extent, vmin=lo, vmax=hi, alpha=0.85)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    ax.set_title(title, loc="left")
fig.suptitle("Conditioning factors (30 m grid)", x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "08_factor_maps.png", dpi=170, bbox_inches="tight"); plt.close(fig)
print("saved 08_factor_maps.png")

# ======== 9 & 10. Susceptibility maps ========
cmap = ListedColormap(CLASS_COLORS)
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
legend = [Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]

def susc(ax, name, title):
    c, _ = read(OUT / f"susceptibility_{name}_class.tif")
    c[c == 0] = np.nan
    base(ax, alpha=0.7)
    ax.imshow(c, cmap=cmap, norm=norm, extent=extent, alpha=0.85, interpolation="nearest")
    ax.set_title(title, loc="left")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 8))
susc(a1, "noroad", "A. Terrain-only model (main)\nRandom Forest without road distance · AUC 0.88")
susc(a2, "withroad", "B. Including road distance\nRandom Forest with road distance · AUC 0.92")
scalebar(a1)
fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, title="Susceptibility class")
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(FIG / "09_susceptibility_maps.png", dpi=170, bbox_inches="tight"); plt.close(fig)
print("saved 09_susceptibility_maps.png")

fig, ax = plt.subplots(figsize=(10, 9))
susc(ax, "noroad", "Landslide susceptibility, Bilaspur district\n(Random Forest, terrain-only model)")
ax.scatter(l.X, l.Y, s=10, color=INK, label="GSI landslides")
ax.legend(handles=legend + [plt.Line2D([], [], marker="o", color=INK, linestyle="", markersize=4,
          label="GSI landslides")], loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="#e4e3df")
scalebar(ax)
fig.savefig(FIG / "10_susceptibility_main.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("saved 10_susceptibility_main.png")
