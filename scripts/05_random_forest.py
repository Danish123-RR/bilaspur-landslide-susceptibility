"""
Random Forest, tested with the same 5-fold cross-validation.

Each tree's leaf holds a landslide fraction; the forest averages them into P(landslide).
Permutation importance: on the TEST fold, shuffle one factor and measure how much AUC drops.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "processed" / "model_data.csv")
OUT = BASE_DIR / "outputs"
OUT.mkdir(exist_ok=True)

# Raw values: trees don't need scaling or log transforms
ALL_FEATURES = ["elevation", "slope", "northness", "eastness",
                "dist_road", "dist_stream", "ndvi"]
y = df["label"]


def make_model():
    return RandomForestClassifier(
        n_estimators=500,        # number of trees
        min_samples_leaf=5,      # every leaf needs >= 5 points (limits memorising)
        random_state=42,
        n_jobs=-1,               # use all CPU cores
    )


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # same folds as before


def evaluate(features, name, importance=False):
    test_aucs, train_aucs, imps = [], [], []
    X = df[features]
    for tr, te in skf.split(X, y):
        model = make_model().fit(X.iloc[tr], y.iloc[tr])
        # predict_proba gives [P(safe), P(landslide)]; column 1 is the risk score
        p_test = model.predict_proba(X.iloc[te])[:, 1]
        p_train = model.predict_proba(X.iloc[tr])[:, 1]
        test_aucs.append(roc_auc_score(y.iloc[te], p_test))
        train_aucs.append(roc_auc_score(y.iloc[tr], p_train))
        if importance:
            r = permutation_importance(model, X.iloc[te], y.iloc[te],
                                       scoring="roc_auc", n_repeats=10,
                                       random_state=42, n_jobs=-1)
            imps.append(r.importances_mean)

    print(f"\n{name}")
    print("  Fold AUCs:", [round(float(a), 3) for a in test_aucs])
    print(f"  RF test AUC:   {np.mean(test_aucs):.3f} +/- {np.std(test_aucs):.3f}")
    print(f"  RF train AUC:  {np.mean(train_aucs):.3f}   (gap to test = overfitting check)")
    return np.mean(imps, axis=0) if importance else None


print(f"Data: {len(df)} points ({y.sum()} landslides)")
imp = evaluate(ALL_FEATURES, "Random Forest, all 7 factors", importance=True)
evaluate([f for f in ALL_FEATURES if f != "dist_road"],
         "Random Forest WITHOUT dist_road (bias check)")

# ---------- Permutation importance table ----------
table = (pd.DataFrame({"factor": ALL_FEATURES, "auc_drop": imp.round(3)})
         .sort_values("auc_drop", ascending=False))
print("\nPermutation importance (AUC drop when the factor is shuffled, test folds):")
print(table.to_string(index=False))
table.to_csv(OUT / "rf_importance.csv", index=False)
print("\nSaved outputs/rf_importance.csv")
