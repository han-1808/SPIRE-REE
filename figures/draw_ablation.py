"""
figures/draw_ablation.py

Grouped bar chart for feature ablation study (Exp 1 and Exp 3).
Produces: figures/fig_ablation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Data (avg ± std, 5 seeds) ──────────────────────────────────────────────

conditions  = ["Full\n(13-dim)", "− Edge-stat\n(11-dim)", "− Ensemble\n(10-dim)"]
metrics     = ["F1", "ROC-AUC", "PR-AUC"]
colors      = ["#DD8452", "#55A868", "#C44E52"]

exp1 = {
    "F1":      {"avg": [0.844, 0.828, 0.807], "std": [0.009, 0.012, 0.014]},
    "ROC-AUC": {"avg": [0.841, 0.842, 0.781], "std": [0.004, 0.007, 0.006]},
    "PR-AUC":  {"avg": [0.912, 0.915, 0.895], "std": [0.003, 0.009, 0.006]},
}

exp3 = {
    "F1":      {"avg": [0.939, 0.901, 0.820], "std": [0.022, 0.041, 0.076]},
    "ROC-AUC": {"avg": [0.947, 0.920, 0.844], "std": [0.053, 0.038, 0.071]},
    "PR-AUC":  {"avg": [0.978, 0.960, 0.913], "std": [0.024, 0.033, 0.053]},
}

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

x      = np.arange(len(conditions))
n_met  = len(metrics)
width  = 0.22
offsets = np.linspace(-(n_met - 1) / 2 * width, (n_met - 1) / 2 * width, n_met)

for ax, data, title in [
    (axes[0], exp1, "Exp 1 — Leave-one-region-out"),
    (axes[1], exp3, "Exp 3 — Global inductive"),
]:
    for i, (m, color) in enumerate(zip(metrics, colors)):
        avgs = data[m]["avg"]
        stds = data[m]["std"]
        bars = ax.bar(
            x + offsets[i], avgs, width,
            label=m, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.5,
            yerr=stds, capsize=3, error_kw={"elinewidth": 1, "ecolor": "gray"},
        )
        # Annotate
        for j, (v, s) in enumerate(zip(avgs, stds)):
            ax.text(
                x[j] + offsets[i], v + s + 0.004,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=6.5, color=color, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0.74, 1.02)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.025))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0.85, loc="lower left",
              ncol=3, columnspacing=0.8, handlelength=1.2)

fig.suptitle("Feature Ablation Study — SPIRE (avg ± std)",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
out = "figures/fig_ablation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
