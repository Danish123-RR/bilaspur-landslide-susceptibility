"""
Clean the sampled table and make it model-ready.

"""
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- 1. Load ----------
BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "data" / "processed" / "samples_v2.csv"
df = pd.read_csv(csv_path)
print("Loaded:", df.shape)

# ---------- 2. Clean label (1 = landslide, 0 = safe) ----------
# Built from the 'layer' column, which is reliable for every row
df["label"] = df["layer"].str.startswith("landslides").astype(int)

# ---------- 3. Readable column names ----------
df = df.rename(columns={"dem": "elevation", "dist_strm": "dist_stream"})

# ---------- 4. Remove points on water (WorldCover class 80) ----------
water = df["landcover"] == 80
print("\nPoints on water:", water.sum(),
      "(landslides:", (water & (df.label == 1)).sum(), ")")
df = df[~water].copy()

# ---------- 5. Aspect -> northness / eastness ----------
# Aspect is circular (359 deg and 1 deg both mean 'north').
rad = np.deg2rad(df["aspect"]) #np.deg2rad() is just for converting from degree to radians
df["northness"] = np.cos(rad) #creating the new column step is inbuilt
df["eastness"] = np.sin(rad)
# Flat ground has no aspect (NaN) -> faces nowhere -> 0, 0
flat = df["aspect"].isna()
print("Flat points (no aspect) set to 0,0:", flat.sum())
df[["northness", "eastness"]] = df[["northness", "eastness"]].fillna(0)

# ---------- 6. Keep only what we need ----------
features = ["elevation", "slope", "northness", "eastness",
            "dist_road", "dist_stream", "ndvi"]
keep = ["X", "Y", "label"] + features + ["landcover"]
df = df[keep]

# ---------- 7. Sanity checks ----------

# df["label"] = df["label"].replace({0: "safe", 1: "landslide"})
# if i want to replace numbers to str then do this

print("\nFinal shape:", df.shape)
print(df["label"].value_counts().rename({0: "safe", 1: "landslide"}))
print("\nMissing values:", int(df.isna().sum().sum()))
print("Negative NDVI left:", int((df.ndvi < 0).sum()))
print("\nMedian of each factor by class:")
print(df.groupby("label")[features].median().round(2).T
        .rename(columns={0: "safe", 1: "landslide"}))

df.to_csv(BASE_DIR / "data" / "processed" / "model_data.csv", index=False)

print("\nSaved model_data.csv")
