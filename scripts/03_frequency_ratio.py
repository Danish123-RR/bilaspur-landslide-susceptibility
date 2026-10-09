"""
Frequency Ratio (FR) model, tested with 5-fold cross-validation.

Idea: split each factor into bands. For every band,
      FR = (% of all landslides in the band) / (% of all points in the band)
      FR > 1 -> riskier than average, FR < 1 -> safer than average.
A point's risk score (LSI) = sum of the FR values of its band in each factor.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "processed" / "model_data.csv")
OUT = BASE_DIR / "outputs"
OUT.mkdir(exist_ok=True)

ALL_FEATURES = ["elevation", "slope", "northness", "eastness",
                "dist_road", "dist_stream", "ndvi"]
N_BANDS = 5   # equal-count bands for every factor except slope


# ---------- 1. Band edges ----------
def band_edges(train_values, feature):
    """Return the band boundaries for one factor, using TRAINING data only."""
    if feature == "slope":
        inner = [5, 10, 15, 20, 25, 30, 35, 40]          # fixed 5-degree bands
    else:
        qs = np.linspace(0, 1, N_BANDS + 1)[1:-1]        # 0.2, 0.4, 0.6, 0.8
        inner = list(np.unique(np.quantile(train_values, qs)))
    # -inf and +inf at the ends so every future value falls in some band
    return [-np.inf] + inner + [np.inf]


# ---------- 2. Learn FR table for one factor ----------
def fr_table(train, feature):
    edges = band_edges(train[feature], feature)
    band = pd.cut(train[feature], edges)
    counts = train.groupby(band, observed=False)["label"].agg(["count", "sum"])
    pct_points = counts["count"] / counts["count"].sum()
    pct_slides = counts["sum"] / counts["sum"].sum()
    fr = (pct_slides / pct_points).fillna(0)             # empty band -> 0
    return edges, fr.values, counts


# ---------- 3. Score points with learned tables ----------
def lsi_score(data, tables, features):
    score = np.zeros(len(data))
    for f in features:
        edges, fr_values, _ = tables[f]
        idx = pd.cut(data[f], edges, labels=False)       # band number 0,1,2...
        score += fr_values[idx]
    return score


# ---------- 4. 5-fold cross-validation ----------
def cross_validate(features, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs_fr, aucs_slope = [], []
    for fold, (tr, te) in enumerate(skf.split(df, df["label"]), start=1):
        train, test = df.iloc[tr], df.iloc[te]
        tables = {f: fr_table(train, f) for f in features}   # learn on 4 folds
        score = lsi_score(test, tables, features)            # score the 5th
        aucs_fr.append(roc_auc_score(test["label"], score))
        aucs_slope.append(roc_auc_score(test["label"], test["slope"]))
    print(f"\n{name}")
    print("  Fold AUCs (FR):   ", [round(a, 3) for a in aucs_fr])
    print(f"  FR model AUC:      {np.mean(aucs_fr):.3f} +/- {np.std(aucs_fr):.3f}")
    print(f"  Slope-only AUC:    {np.mean(aucs_slope):.3f} +/- {np.std(aucs_slope):.3f}  (same folds)")


print(f"Data: {len(df)} points ({df.label.sum()} landslides)")
cross_validate(ALL_FEATURES, "FR with all 7 factors")
cross_validate([f for f in ALL_FEATURES if f != "dist_road"],
               "FR WITHOUT dist_road (bias check)")


# ---------- 5. FR tables from ALL data, for interpretation ----------
rows = []
for f in ALL_FEATURES:
    edges, fr_values, counts = fr_table(df, f)
    for (band, c), fr in zip(counts.iterrows(), fr_values):
        rows.append({"factor": f, "band": str(band), "points": int(c["count"]),
                     "landslides": int(c["sum"]), "FR": round(fr, 2)})
tables_df = pd.DataFrame(rows)
tables_df.to_csv(OUT / "fr_tables.csv", index=False)

print("\nFR by slope band (all data):")
print(tables_df[tables_df.factor == "slope"][["band", "points", "landslides", "FR"]]
      .to_string(index=False))
print("\nSaved outputs/fr_tables.csv")
