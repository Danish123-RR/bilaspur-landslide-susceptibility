"""
Step 7a - Result charts for the presentation (PNG, 16:9-friendly).
Inputs : data/processed/model_data.csv and the CSVs in outputs/
Output : outputs/figures/*.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------- shared style ----------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical pair
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

df = pd.read_csv(OUT.parent / "data" / "processed" / "model_data.csv")

# ======== 1. Factor distributions: landslide vs safe ========
panels = [("slope", "Slope (°)"), ("elevation", "Elevation (m)"), ("ndvi", "NDVI (2019–24 median)"),
          ("dist_road", "Distance to road (m)"), ("dist_stream", "Distance to stream (m)"),
          ("northness", "Northness (cos aspect)")]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for ax, (col, title) in zip(axes.flat, panels):
    data = [df.loc[df.label == 0, col], df.loc[df.label == 1, col]]
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False,
                    medianprops=dict(color=SURFACE, linewidth=2))
    for patch, c in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set(facecolor=c, edgecolor=c)
    for part in ("whiskers", "caps"):
        for line, c in zip(bp[part], [BLUE, BLUE, ORANGE, ORANGE]):
            line.set(color=c, linewidth=1.5)
    ax.set_xticks([1, 2], ["Safe", "Landslide"])
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
fig.suptitle("Conditioning factors at 219 landslide and 563 safe points",
             x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.tight_layout()
save(fig, "01_factor_distributions.png")

# ======== 2. Frequency Ratio by slope band ========
fr = pd.read_csv(OUT / "fr_tables.csv")
s = fr[fr.factor == "slope"].reset_index(drop=True)
labels = ["0–5", "5–10", "10–15", "15–20", "20–25", "25–30", "30–35", "35–40", ">40"]
fig, ax = plt.subplots(figsize=(10, 5.4))
x = np.arange(len(s))
ax.bar(x, s.FR, width=0.55, color=[ORANGE if v > 1 else BLUE for v in s.FR])
ax.axhline(1, color=INK2, linewidth=1)
ax.text(-0.3, 1.05, "FR = 1 (district average)", ha="left", va="bottom", color=INK2, fontsize=10)
for xi, v in zip(x, s.FR):
    ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(x, labels)
ax.set_xlabel("Slope band (°)")
ax.set_ylabel("Frequency Ratio")
ax.grid(axis="x", visible=False)
ax.set_title("Landslide frequency rises steadily with slope\n"
             "(orange: over-represented, FR > 1 · blue: under-represented)", loc="left")
save(fig, "02_slope_frequency_ratio.png")

# ======== 3. Model comparison (2 km spatial CV) ========
cv = pd.concat([pd.read_csv(OUT / "cv_comparison.csv"), pd.read_csv(OUT / "mcda_results.csv")])
cv = cv[cv.scheme == "Spatial 2 km"]
order = ["Slope only", "MCDA (AHP)", "Frequency Ratio", "Logistic Regression", "Random Forest"]
fig, ax = plt.subplots(figsize=(10, 5.6))
y = np.arange(len(order))
h = 0.34
for off, roads, color, name in [(-h / 2, "without dist_road", BLUE, "Without road distance (main)"),
                                (h / 2, "with dist_road", ORANGE, "With road distance")]:
    sub = cv[cv.roads == roads].set_index("method").loc[order]
    ax.barh(y + off, sub.auc, height=h * 0.9, color=color, label=name,
            xerr=sub["std"], error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2))
    for yi, v, sd in zip(y + off, sub.auc, sub["std"]):
        ax.text(v + sd + 0.004, yi, f"{v:.3f}", va="center", fontsize=10, color=INK)
ax.set_yticks(y, order)
ax.invert_yaxis()
ax.set_xlim(0.70, 1.0)
ax.set_xlabel("AUC (mean of 5 folds, 2 km spatial blocks) — higher is better")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
ax.set_title("Data-driven models beat the expert-weighted MCDA baseline", loc="left", pad=34)
save(fig, "03_model_comparison_auc.png")

# ======== 4. Validation check: random vs spatial CV (main feature set) ========
cv_all = pd.concat([pd.read_csv(OUT / "cv_comparison.csv"), pd.read_csv(OUT / "mcda_results.csv")])
sub = cv_all[cv_all.roads == "without dist_road"]
schemes = ["Random", "Spatial 2 km", "Spatial 5 km"]
fig, ax = plt.subplots(figsize=(10, 5.4))
for m in order:
    vals = sub[sub.method == m].set_index("scheme").loc[schemes, "auc"]
    color = ORANGE if m == "Random Forest" else BLUE
    ax.plot(schemes, vals, marker="o", markersize=8, linewidth=2, color=color,
            alpha=1 if m == "Random Forest" else 0.45, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(2.06, vals.iloc[-1], m, va="center", fontsize=10, color=INK)
ax.set_xlim(-0.2, 2.9)
ax.set_ylabel("AUC (without road distance)")
ax.set_title("Scores hold when tested on unseen areas (spatial cross-validation)", loc="left")
ax.grid(axis="x", visible=False)
save(fig, "04_random_vs_spatial_cv.png")

# ======== 5. Factor importance: LR weights and RF permutation ========
lr = pd.read_csv(OUT / "lr_weights.csv")
lr["factor"] = lr.factor.str.replace("log_", "")
rf = pd.read_csv(OUT / "rf_importance.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4))
lr = lr.reindex(lr.weight.abs().sort_values().index)
a1.barh(lr.factor, lr.weight, height=0.5, color=[ORANGE if w > 0 else BLUE for w in lr.weight])
a1.axvline(0, color=INK2, linewidth=1)
a1.set_title("Logistic Regression weights\n(orange: higher = riskier · blue: lower = riskier)", loc="left")
a1.set_xlabel("Standardised weight")
a1.grid(axis="y", visible=False)
rf = rf.sort_values("auc_drop")
a2.barh(rf.factor, rf.auc_drop, height=0.5, color=BLUE)
for yi, v in enumerate(rf.auc_drop):
    a2.text(max(v, 0) + 0.003, yi, f"{v:.3f}", va="center", fontsize=10)
a2.set_title("Random Forest permutation importance\n(AUC lost when the factor is shuffled)", loc="left")
a2.set_xlabel("Drop in test AUC")
a2.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "05_factor_importance.png")

# ======== 6. Class summary: share of area vs share of landslides ========
cs = pd.read_csv(OUT / "class_summary.csv")
cs = cs[cs["map"] == "noroad"]
fig, ax = plt.subplots(figsize=(10, 5.4))
x = np.arange(len(cs))
w = 0.36
ax.bar(x - w / 2, cs.pct_area, width=w * 0.92, color=BLUE, label="% of district area")
ax.bar(x + w / 2, cs.pct_landslides, width=w * 0.92, color=ORANGE, label="% of known landslides")
for xi, a, l in zip(x, cs.pct_area, cs.pct_landslides):
    ax.text(xi - w / 2, a + 1, f"{a:.1f}", ha="center", fontsize=10)
    ax.text(xi + w / 2, l + 1, f"{l:.1f}", ha="center", fontsize=10)
ax.set_xticks(x, cs["class"])
ax.set_ylabel("Percent")
ax.legend(loc="upper center")
ax.grid(axis="x", visible=False)
hi_a = cs[cs["class"].isin(["High", "Very High"])].pct_area.sum()
hi_l = cs[cs["class"].isin(["High", "Very High"])].pct_landslides.sum()
ax.set_title(f"Main map: High + Very High cover {hi_a:.1f}% of the district\n"
             f"but contain about {hi_l:.0f}% of mapped landslides", loc="left")
save(fig, "06_class_area_vs_landslides.png")
