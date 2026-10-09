"""
Baseline: how well does SLOPE ALONE separate landslides from safe points?
No model is trained here - we use the slope value itself as the 'risk score'.
"""
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "processed" / "model_data.csv")

y = df["label"]          # truth: 1 = landslide, 0 = safe

# ---------- 1. The baseline: slope as the risk score ----------
auc_slope = roc_auc_score(y, df["slope"])
print(f"Baseline AUC (slope only): {auc_slope:.3f}")

# ---------- 2. Every factor on its own ----------
# For some factors, LOWER means riskier (e.g. closer to a road).
# AUC below 0.5 means 'works, but backwards'; we report both.
print("\nSingle-factor AUC (each factor used alone as the score):")
features = ["elevation", "slope", "northness", "eastness",
            "dist_road", "dist_stream", "ndvi"]
for f in features:
    auc = roc_auc_score(y, df[f])
    direction = "higher = riskier" if auc >= 0.5 else "lower = riskier"
    strength = max(auc, 1 - auc)
    print(f"  {f:12s} AUC = {auc:.3f}  -> strength {strength:.3f} ({direction})")
