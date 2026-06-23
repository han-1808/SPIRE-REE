"""
figures/draw_k_sensitivity.py

Bar chart for graph construction k sensitivity (Exp 1, SPIRE, seed=42).
Produces: figures/fig_k_sensitivity.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Load data ─────────────────────────────────────────────────────────────────

with open("result/k_sensitivity.json") as f:
    d = json.load(f)

ks      = [k for k in sorted(d["results"].keys(), key=int) if int(k) != 5]
k_vals  = [int(k) for k in ks]
metrics = ["f1", "roc_auc", "pr_auc"]
labels  = ["F1", "ROC-AUC", "PR-AUC"]
colors  = ["#DD8452", "#55A868", "#C44E52"]

data = {m: [d["results"][k]["avg_metrics"][m] for k in ks] for m in metrics}

default_k   = 15
default_idx = k_vals.index(default_k)

# ── Plot ──────────────────────────────────────────────────────────────────────

y      = np.arange(len(k_vals))
n_met  = len(metrics)
height = 0.22
offsets = np.linspace(-(n_met - 1) / 2 * height, (n_met - 1) / 2 * height, n_met)

fig, ax = plt.subplots(figsize=(9, 6))

for i, (m, label, color) in enumerate(zip(metrics, labels, colors)):
    bars = ax.barh(y + offsets[i], data[m], height, label=label,
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Highlight default k bars with a black edge
    bars[default_idx].set_edgecolor("black")
    bars[default_idx].set_linewidth(1.8)

    # Annotate value at end of each bar
    for j, v in enumerate(data[m]):
        ax.text(
            v + 0.001, y[j] + offsets[i],
            f"{v:.3f}", va="center", ha="left",
            fontsize=7, color=color, fontweight="bold",
        )

# ── Axes ──────────────────────────────────────────────────────────────────────

ax.set_yticks(y)
ax.set_yticklabels(
    [f"k={k}" for k in k_vals],
    fontsize=9,
)
ax.set_xlabel("Score", fontsize=11)
ax.set_title(
    "Graph Construction k Sensitivity — Exp 1 SPIRE",
    fontsize=12, fontweight="bold", pad=10,
)
ax.set_xlim(0.78, 0.96)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.01))
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

ax.legend(
    loc="lower right", fontsize=9, framealpha=0.85,
    ncol=3, columnspacing=1.0, handlelength=1.2,
)

fig.tight_layout()
out = "figures/fig_k_sensitivity.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
