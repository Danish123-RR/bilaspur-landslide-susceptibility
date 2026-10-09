"""
 Logistic Regression, tested with the same 5-fold cross-validation as FR.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "processed" / "model_data.csv")
OUT = BASE_DIR / "outputs"
OUT.mkdir(exist_ok=True)

# ---------- 1. Log-transform the skewed distances ----------
# log1p(x) = log(1 + x), so a distance of 0 stays 0 (log(0) would be -infinity)
df["log_dist_road"] = np.log1p(df["dist_road"])
df["log_dist_stream"] = np.log1p(df["dist_stream"])

ALL_FEATURES = ["elevation", "slope", "northness", "eastness",
                "log_dist_road", "log_dist_stream", "ndvi"]
y = df["label"]


# ---------- 2. Model = scaler + logistic regression, bundled ----------
# Inside cross-validation the Pipeline fits the scaler on the training folds only.
def make_model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000))


# ---------- 3. Same 5 folds as the FR script (same random_state) ----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(features, name):
    aucs = cross_val_score(make_model(), df[features], y, cv=skf, scoring="roc_auc")
    print(f"\n{name}")
    print("  Fold AUCs:", [round(float(a), 3) for a in aucs])
    print(f"  LR AUC:    {aucs.mean():.3f} +/- {aucs.std():.3f}")

print(f"Data: {len(df)} points ({y.sum()} landslides)")
evaluate(ALL_FEATURES, "Logistic Regression, all 7 factors")
evaluate([f for f in ALL_FEATURES if f != "log_dist_road"],
         "Logistic Regression WITHOUT dist_road (bias check)")


# ---------- 4. Weights: which factor matters, and in which direction ----------
# Fit once on ALL data, only to read the weights (not for scoring).
model = make_model().fit(df[ALL_FEATURES], y)
weights = model.named_steps["logisticregression"].coef_[0]

table = (pd.DataFrame({"factor": ALL_FEATURES, "weight": weights})
         .assign(abs_weight=lambda t: t.weight.abs())
         .sort_values("abs_weight", ascending=False)
         .drop(columns="abs_weight"))
table["direction"] = np.where(table.weight > 0, "higher = riskier", "lower = riskier")
table["weight"] = table["weight"].round(3)

print("\nStandardised weights (all data), biggest effect first:")
print(table.to_string(index=False))
table.to_csv(OUT / "lr_weights.csv", index=False)
print("\nSaved outputs/lr_weights.csv")
