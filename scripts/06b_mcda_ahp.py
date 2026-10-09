"""
Step 5.7 - MCDA with AHP weights (expert-weighted baseline), same CV folds as script 06.
Input : data/processed/model_data.csv
Output: outputs/mcda_results.csv

MCDA uses NO training: every factor is turned into a 1-5 score with fixed expert rules,
multiplied by its AHP weight (build guide, Part 5), and summed.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "processed" / "model_data.csv")
OUT = BASE_DIR / "outputs"
y = df["label"]

# AHP weights from the build guide (literature consensus)
WEIGHTS = {"slope": 0.404, "dist_road": 0.163, "dist_stream": 0.152,
           "ndvi": 0.152, "elevation": 0.069, "aspect": 0.060}

def bands(values, edges, scores):
    return np.select([values <= e for e in edges], scores[:-1], default=scores[-1])

# Expert 1-5 scoring rules (5 = most susceptible), fixed in advance, not learned
SCORES = {
    "slope":       lambda d: bands(d.slope, [10, 20, 30, 40], [1, 2, 3, 4, 5]),
    "dist_road":   lambda d: bands(d.dist_road, [100, 250, 500, 1000], [5, 4, 3, 2, 1]),
    "dist_stream": lambda d: bands(d.dist_stream, [100, 250, 500, 1000], [5, 4, 3, 2, 1]),
    "ndvi":        lambda d: bands(d.ndvi, [0.2, 0.4, 0.6, 0.75], [5, 4, 3, 2, 1]),
    "elevation":   lambda d: bands(d.elevation, [500, 700, 900, 1100], [1, 2, 3, 4, 5]),
    # south-facing (northness -1) wetter/drier cycles -> higher; flat (0,0) -> middle
    "aspect":      lambda d: np.clip(np.round(3 - 2 * d.northness), 1, 5),
}

def mcda_score(d, use_road=True):
    w = {k: v for k, v in WEIGHTS.items() if use_road or k != "dist_road"}
    total = sum(w.values())                       # re-normalise if road is dropped
    return sum(SCORES[k](d) * v / total for k, v in w.items())

def block_ids(size):
    return (df.X // size).astype(int).astype(str) + "_" + (df.Y // size).astype(int).astype(str)

def splits(scheme):
    if scheme == "Random":
        return StratifiedKFold(5, shuffle=True, random_state=42).split(df, y)
    size = {"Spatial 2 km": 2000, "Spatial 5 km": 5000}[scheme]
    return StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, y, groups=block_ids(size))

rows = []
for use_road in [True, False]:
    score = mcda_score(df, use_road)
    for scheme in ["Random", "Spatial 2 km", "Spatial 5 km"]:
        aucs = [roc_auc_score(y.iloc[te], score.iloc[te]) for _, te in splits(scheme)]
        rows.append({"roads": "with dist_road" if use_road else "without dist_road",
                     "scheme": scheme, "method": "MCDA (AHP)",
                     "auc": np.mean(aucs), "std": np.std(aucs)})
res = pd.DataFrame(rows)
res.to_csv(OUT / "mcda_results.csv", index=False)
print(res.round(3).to_string(index=False))
