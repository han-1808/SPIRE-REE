"""
Hyperparameter sensitivity heatmap.
Produces:
  figures/fig_hparam_sensitivity.png — validation F1 vs hidden_dim x lr
  (best num_layers and dropout fixed) for one representative region.

Requires result/hparam_sensitivity.json (run experiment/run_hparam_sensitivity.py).
"""

import json

import matplotlib.pyplot as plt
import numpy as np

# ── Load data ─────────────────────────────────────────────────────────────────

with open("result/hparam_sensitivity.json") as f:
    res = json.load(f)

records = res["records"]
region = res["region"]
best_hidden, best_layers, best_dropout, best_lr = res["best_config"]

hidden_dims = res["search_space"]["hidden_dim"]
lrs = res["search_space"]["lr"]

# Fix num_layers and dropout at their best values; pivot hidden_dim x lr
surface = np.full((len(hidden_dims), len(lrs)), np.nan)
for r in records:
    if r["num_layers"] == best_layers and r["dropout"] == best_dropout:
        i = hidden_dims.index(r["hidden_dim"])
        j = lrs.index(r["lr"])
        surface[i, j] = r["val_f1"]

# ── Figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7.2, 5.6))
fig.patch.set_facecolor("#F8F9FA")

vmin = max(0.0, np.nanmin(surface) - 0.02)
im = ax.imshow(surface, cmap="viridis", aspect="auto", vmin=vmin, vmax=1.0)

ax.set_xticks(np.arange(len(lrs)))
ax.set_xticklabels([f"{lr:g}" for lr in lrs], fontsize=10)
ax.set_yticks(np.arange(len(hidden_dims)))
ax.set_yticklabels(hidden_dims, fontsize=10)
ax.set_xlabel("Learning rate", fontsize=11)
ax.set_ylabel("Hidden dimension", fontsize=11)

mean_f1 = float(np.nanmean(surface))
std_f1 = float(np.nanstd(surface))
ax.set_title(
    f"Validation F1 — hidden dim × learning rate ({region})\n"
    f"num_layers={best_layers}, dropout={best_dropout} fixed",
    fontsize=11, fontweight="bold", pad=10,
)

# Cell annotations
for i in range(len(hidden_dims)):
    for j in range(len(lrs)):
        v = surface[i, j]
        if np.isnan(v):
            continue
        color = "white" if v < vmin + 0.6 * (1.0 - vmin) else "#1A1A2E"
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=9.5, color=color, fontweight="bold")

# Mark the best cell of the expanded grid
# Mark all cells that tie at the maximum val_f1
max_f1 = np.nanmax(surface)
for i in range(len(hidden_dims)):
    for j in range(len(lrs)):
        if np.isclose(surface[i, j], max_f1):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#E63946", linewidth=2.5))

saved = res.get("saved_orig_run")
handles = [plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#E63946",
                         linewidth=2.5, label="Best configs")]

ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.12),
          fontsize=9, framealpha=0.85, ncol=2)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Validation F1", fontsize=10)

out_path = "figures/fig_hparam_sensitivity.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
plt.close()
