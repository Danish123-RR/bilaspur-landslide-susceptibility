"""
Random vs spatial cross-validation, all four methods side by side.

Spatial CV: points are grouped into square blocks (2 km or 5 km) using X, Y,
and whole blocks go to the same fold, so the model is tested on areas it never saw.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "processed" / "model_data.csv")
OUT = BASE_DIR / "outputs"
OUT.mkdir(exist_ok=True)

df["log_dist_road"] = np.log1p(df["dist_road"])
df["log_dist_stream"] = np.log1p(df["dist_stream"])
y = df["label"]

RAW = ["elevation", "slope", "northness", "eastness", "dist_road", "dist_stream", "ndvi"]
LOGGED = ["elevation", "slope", "northness", "eastness", "log_dist_road", "log_dist_stream", "ndvi"]


# ================= The four methods =================
# Each takes (train, test, features) and returns risk scores for the test rows.

def score_slope(train, test, feats):
    return test["slope"].values                  # no training at all

# --- Frequency Ratio (same logic as script 03) ---
def _edges(values, f):
    if f == "slope":
        inner = [5, 10, 15, 20, 25, 30, 35, 40]
    else:
        inner = list(np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8])))
    return [-np.inf] + inner + [np.inf]

def score_fr(train, test, feats):
    total = np.zeros(len(test))
    for f in feats:
        edges = _edges(train[f], f)
        c = train.groupby(pd.cut(train[f], edges), observed=False)["label"].agg(["count", "sum"])
        fr = ((c["sum"] / c["sum"].sum()) / (c["count"] / c["count"].sum())).fillna(0).values
        total += fr[pd.cut(test[f], edges, labels=False)]
    return total

def score_lr(train, test, feats):
    feats = [LOGGED[RAW.index(f)] for f in feats]          # LR uses logged distances
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    m.fit(train[feats], train["label"])
    return m.predict_proba(test[feats])[:, 1]

def score_rf(train, test, feats):
    m = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                               random_state=42, n_jobs=-1)
    m.fit(train[feats], train["label"])
    return m.predict_proba(test[feats])[:, 1]

METHODS = {"Slope only": score_slope, "Frequency Ratio": score_fr,
           "Logistic Regression": score_lr, "Random Forest": score_rf}


# ================= The three ways of splitting =================
def block_ids(size_m):
    """Give each point the ID of the square block it falls in."""
    return ((df["X"] // size_m).astype(int).astype(str) + "_" +
            (df["Y"] // size_m).astype(int).astype(str))

def splits(scheme):
    if scheme == "Random":
        return StratifiedKFold(5, shuffle=True, random_state=42).split(df, y)
    size = {"Spatial 2 km": 2000, "Spatial 5 km": 5000}[scheme]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    return cv.split(df, y, groups=block_ids(size))

SCHEMES = ["Random", "Spatial 2 km", "Spatial 5 km"]


# ================= Run everything =================
rows = []
for with_road in [True, False]:
    feats = RAW if with_road else [f for f in RAW if f != "dist_road"]
    for scheme in SCHEMES:
        fold_list = list(splits(scheme))
        for name, scorer in METHODS.items():
            aucs = []
            for tr, te in fold_list:
                train, test = df.iloc[tr], df.iloc[te]
                aucs.append(roc_auc_score(test["label"], scorer(train, test, feats)))
            rows.append({"roads": "with dist_road" if with_road else "without dist_road",
                         "scheme": scheme, "method": name,
                         "auc": np.mean(aucs), "std": np.std(aucs)})
        print(f"done: {scheme:13s} {'with' if with_road else 'without'} dist_road")

res = pd.DataFrame(rows)
res.to_csv(OUT / "cv_comparison.csv", index=False)

# ---------- Print as two readable tables ----------
for roads in ["with dist_road", "without dist_road"]:
    t = (res[res.roads == roads]
         .assign(cell=lambda r: [f"{a:.3f} +/- {s:.3f}" for a, s in zip(r.auc, r["std"])])
         .pivot(index="method", columns="scheme", values="cell")
         .reindex(list(METHODS))[SCHEMES])
    print(f"\n=== AUC {roads} ===")
    print(t.to_string())

print("\nSaved outputs/cv_comparison.csv")
