"""
Step 6 - Susceptibility maps for the whole of Bilaspur district.

1. Train the final Random Forest on ALL 782 points (twice: without and with dist_road).
2. Read the 6 aligned factor rasters, treat every 30 m pixel as one row, predict P(landslide).
3. Mask pixels outside the district and on water (WorldCover 80).
4. Classify into 5 equal-interval classes on the scale 0 -> max probability.
5. Checks: (a) where the 219 known landslides fall, (b) the Oct 2025 Bhalu bridge slide.

Inputs : data/processed/model_data.csv, the factor .tif files, bilaspur_boundary_utm.gpkg
Outputs: outputs/susceptibility_<name>_prob.tif   (probability, float)
         outputs/susceptibility_<name>_class.tif  (1=Very Low ... 5=Very High, 0=no data)
         outputs/class_summary.csv
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from pyproj import Transformer
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
OUT = BASE_DIR / "outputs"
OUT.mkdir(exist_ok=True)

# ---------- File names (change here if yours differ) ----------
RASTERS = {
    "elevation":   PROC / "dem_utm.tif",
    "slope":       PROC / "slope.tif",
    "aspect":      PROC / "aspect.tif",
    "dist_road":   PROC / "dist_road_v2.tif",
    "dist_stream": PROC / "dist_stream.tif",
    "ndvi":        PROC / "ndvi_30m.tif",
    "landcover":   PROC / "landcover_30m.tif",
}
BOUNDARY = PROC / "bilaspur_boundary_utm.gpkg"

FEATURES_ALL = ["elevation", "slope", "northness", "eastness",
                "dist_road", "dist_stream", "ndvi"]
MAPS = {   # name -> features used
    "noroad":   [f for f in FEATURES_ALL if f != "dist_road"],   # MAIN map
    "withroad": FEATURES_ALL,                                     # secondary map
}
CLASS_NAMES = ["Very Low", "Low", "Moderate", "High", "Very High"]

# Oct 7 2025 slide at Bhalu bridge (scar centre found in Earth Engine), lon/lat
EVENT_LONLAT = (76.61119, 31.413439)
EVENT_RADIUS_M = 250


# ================= 1. Read and check the rasters =================
arrays, profile = {}, None
for name, path in RASTERS.items():
    with rasterio.open(path) as src:
        band = src.read(1).astype("float32")
        if src.nodata is not None:
            band[band == src.nodata] = np.nan
        if profile is None:
            profile, ref_name = src.profile, name
            transform, shape, crs = src.transform, src.shape, src.crs
        else:
            t = src.transform
            shift_x, shift_y = abs(t.c - transform.c), abs(t.f - transform.f)   # origin offset (m)
            size_diff = max(abs(t.a - transform.a), abs(t.e - transform.e))      # pixel-size diff (m)
            print(f"  {name:12s} shape {src.shape}  origin shift {shift_x:.3f} m, {shift_y:.3f} m  "
                  f"pixel-size diff {size_diff:.5f} m")
            if src.shape != shape or shift_x > 1 or shift_y > 1 or size_diff > 0.01:
                raise SystemExit(f"GRID MISMATCH: {name} differs from {ref_name} by more than "
                                 "rounding noise (see line above). All rasters must share the dem_utm grid.")
        arrays[name] = band
print(f"Grid: {shape[0]} rows x {shape[1]} cols, CRS {crs}")

# aspect -> northness / eastness (flat ground = no direction = 0, 0)
rad = np.deg2rad(arrays["aspect"])
arrays["northness"] = np.nan_to_num(np.cos(rad), nan=0.0)
arrays["eastness"] = np.nan_to_num(np.sin(rad), nan=0.0)
arrays["northness"][arrays["aspect"] < 0] = 0      # GDAL writes -9999 for flat
arrays["eastness"][arrays["aspect"] < 0] = 0

# ================= 2. Which pixels get a prediction =================
boundary = gpd.read_file(BOUNDARY).to_crs(crs)
inside = geometry_mask(boundary.geometry, out_shape=shape, transform=transform, invert=True)
not_water = arrays["landcover"] != 80
complete = np.all([np.isfinite(arrays[f]) for f in FEATURES_ALL], axis=0)
valid = inside & not_water & complete
print(f"Pixels inside district: {inside.sum():,} | to be scored: {valid.sum():,} "
      f"(water removed: {(inside & ~not_water).sum():,})")

# ================= 3. Training data (all points) =================
df = pd.read_csv(PROC / "model_data.csv")
y = df["label"]

# Pixel position of each training point and of the event
def to_rowcol(xs, ys):
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    return np.array(rows), np.array(cols)

ls = df[df.label == 1]
ls_r, ls_c = to_rowcol(ls.X.values, ls.Y.values)
ev_x, ev_y = Transformer.from_crs(4326, crs, always_xy=True).transform(*EVENT_LONLAT)
ev_r, ev_c = to_rowcol([ev_x], [ev_y])
rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
px = abs(transform.a)
event_zone = ((rr - ev_r[0]) ** 2 + (cc - ev_c[0]) ** 2) * px * px <= EVENT_RADIUS_M ** 2
dist_to_train = np.hypot(ls.X - ev_x, ls.Y - ev_y).min()

summary_rows = []

# ================= 4. One map per feature set =================
for map_name, feats in MAPS.items():
    print(f"\n========== Map: {map_name} ({len(feats)} factors) ==========")
    model = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   random_state=42, n_jobs=-1)
    model.fit(df[feats], y)

    # Predict every valid pixel, in chunks to save memory
    table = pd.DataFrame({f: arrays[f][valid] for f in feats})   # same column names as training
    probs = np.concatenate([model.predict_proba(table.iloc[i:i + 200_000])[:, 1]
                            for i in range(0, len(table), 200_000)])
    prob_map = np.full(shape, np.nan, dtype="float32")
    prob_map[valid] = probs

    # ---- 5 equal-interval classes on 0 -> max ----
    top = float(np.nanmax(probs))
    p99 = float(np.nanpercentile(probs, 99))
    edges = np.linspace(0, top, 6)                       # 0, top/5, ..., top
    cls = np.zeros(shape, dtype="uint8")                 # 0 = no data
    cls[valid] = np.clip(np.digitize(probs, edges[1:-1]) + 1, 1, 5)
    print(f"Max probability {top:.3f} (99th percentile {p99:.3f})")
    print("Class edges:", " | ".join(f"{a:.3f}-{b:.3f}" for a, b in zip(edges[:-1], edges[1:])))

    # ---- Save GeoTIFFs ----
    prof = profile.copy()
    prof.update(count=1, compress="lzw")
    with rasterio.open(OUT / f"susceptibility_{map_name}_prob.tif", "w",
                       **{**prof, "dtype": "float32", "nodata": -1}) as dst:
        dst.write(np.nan_to_num(prob_map, nan=-1).astype("float32"), 1)
    with rasterio.open(OUT / f"susceptibility_{map_name}_class.tif", "w",
                       **{**prof, "dtype": "uint8", "nodata": 0}) as dst:
        dst.write(cls, 1)

    # ---- Check (a): area share vs landslide share per class ----
    ls_cls = cls[ls_r, ls_c]
    ls_cls = ls_cls[ls_cls > 0]                          # drop any point on masked pixels
    print(f"\n{'Class':10s} {'% of area':>10s} {'% of landslides':>16s}")
    for k, nm in enumerate(CLASS_NAMES, start=1):
        a = (cls[valid] == k).mean() * 100
        l = (ls_cls == k).mean() * 100
        print(f"{nm:10s} {a:10.1f} {l:16.1f}")
        summary_rows.append({"map": map_name, "class": nm, "pct_area": round(a, 1),
                             "pct_landslides": round(l, 1),
                             "prob_from": round(edges[k - 1], 3), "prob_to": round(edges[k], 3)})
    hi_area = (cls[valid] >= 4).mean() * 100
    hi_ls = (ls_cls >= 4).mean() * 100
    print(f"High + Very High: {hi_area:.1f}% of the area holds {hi_ls:.1f}% of known landslides")

    # ---- Check (b): the Oct 2025 Bhalu bridge slide ----
    zone = event_zone & valid
    zone_cls = cls[zone]
    print(f"\nOct 2025 Bhalu bridge slide (nearest training landslide {dist_to_train:,.0f} m away):")
    ev_cls = cls[ev_r[0], ev_c[0]]
    if ev_cls:
        print(f"  At the scar pixel: probability {prob_map[ev_r[0], ev_c[0]]:.3f} "
              f"-> class {CLASS_NAMES[ev_cls - 1]}")
    else:
        print("  At the scar pixel: masked (classed as water or missing data) - see the 250 m zone")
    print(f"  Within {EVENT_RADIUS_M} m: highest class {CLASS_NAMES[zone_cls.max() - 1]}, "
          f"mean probability {np.nanmean(prob_map[zone]):.3f}")
    for k, nm in enumerate(CLASS_NAMES, start=1):
        print(f"    {nm:10s} {(zone_cls == k).mean() * 100:5.1f}% of pixels")

pd.DataFrame(summary_rows).to_csv(OUT / "class_summary.csv", index=False)
print("\nSaved: outputs/susceptibility_*_prob.tif, *_class.tif, class_summary.csv")
