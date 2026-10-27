"""
figures/draw_ablation.py

Grouped bar chart for the SPIRE feature ablation (Exp 1 and Exp 3).
Produces: figures/fig_ablation.png

Reads result/ablation_feature.json (Exp 1) and result/ablation_feature_exp3.json
(Exp 3). The numbers used to be pasted into this file as literals, which meant
the figure silently kept showing the pre-revision values (F1 0.844 / 0.939)
after the pipeline was rerun, and would have missed the 12-dim condition added
for Reviewer #2 Remark #6 entirely. Conditions are now discovered from each
file's `summary` block and ordered by input dimensionality, so whatever modes
the ablation scripts were run with is what gets plotted.

The "full" condition lives in the ablation files' own summary when present;
otherwise it falls back to the main experiment result for SPIRE, which is the
same quantity (13-dim, no features removed).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

METRICS = ["F1", "ROC-AUC", "PR-AUC"]
_METRIC_KEYS = {"F1": "f1", "ROC-AUC": "roc_auc", "PR-AUC": "pr_auc"}
COLORS = ["#DD8452", "#55A868", "#C44E52"]

# Pretty labels; dimensionality is appended from the file so it stays truthful.
_MODE_LABELS = {
    "full": "Full",
    "no_edge_stats": "− Edge-stat",
    "no_ensemble_keep_edge_stats": "− Ensemble\n(keep edge-stat)",
    "no_ensemble": "− Ensemble\n− Edge-stat",
}


def _load(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _spire_fallback(main_path):
    """'full' SPIRE metrics from the main experiment file, if the ablation omits it."""
    doc = _load(main_path)
    if doc is None:
        return None
    if "avg_metrics" in doc and "spire" in doc["avg_metrics"]:      # Exp 1 shape
        return doc["avg_metrics"]["spire"], {}
    runs = doc.get("runs")                                           # Exp 3 shape
    if runs:
        runs = list(runs.values()) if isinstance(runs, dict) else runs
        acc = {}
        for r in runs:
            for k, v in ((r.get("spire") or {}).get("avg_metrics") or {}).items():
                if isinstance(v, (int, float)):
                    acc.setdefault(k, []).append(v)
        if acc:
            return ({k: float(np.mean(v)) for k, v in acc.items()},
                    {k: float(np.std(v)) for k, v in acc.items()})
    return None


def _series(ablation_path, main_path):
    """-> (labels, {metric: {'avg': [...], 'std': [...]}}) ordered by input_dim desc."""
    doc = _load(ablation_path)
    if doc is None:
        return None, None
    summary = dict(doc.get("summary") or {})

    if "full" not in summary:
        # Provenance guard: borrowing "full" from the main experiment file is
        # only valid if both came from the same pipeline run. Mixing a fresh
        # main result with a stale ablation silently produces a figure where
        # removing features appears to IMPROVE the model.
        abl_t, main_t = Path(ablation_path).stat().st_mtime, Path(main_path).stat().st_mtime
        if main_t - abl_t > 3600:
            raise SystemExit(
                f"{Path(main_path).name} is newer than {Path(ablation_path).name} by "
                f"{(main_t - abl_t) / 3600:.1f}h -- rerun the ablation before plotting, "
                "or the panel would mix two different pipelines."
            )
        fb = _spire_fallback(main_path)
        if fb:
            summary["full"] = {"input_dim": 13, "avg_metrics": fb[0], "std_metrics": fb[1]}

    modes = sorted(summary, key=lambda m: -summary[m].get("input_dim", 0))
    labels, data = [], {m: {"avg": [], "std": []} for m in METRICS}
    for mode in modes:
        dim = summary[mode].get("input_dim")
        labels.append(f"{_MODE_LABELS.get(mode, mode)}\n({dim}-dim)")
        for m in METRICS:
            k = _METRIC_KEYS[m]
            data[m]["avg"].append(summary[mode].get("avg_metrics", {}).get(k, np.nan))
            data[m]["std"].append(summary[mode].get("std_metrics", {}).get(k, 0.0))
    return labels, data


panels = [
    ("result/ablation_feature.json", "result/gnn_inductive.json"),
    ("result/ablation_feature_exp3.json", "result/inductive_sample.json"),
]

loaded = [_series(a, m) for a, m in panels]
available = [(l, d) for l, d in loaded if l]
if not available:
    raise SystemExit("No ablation result files found — run experiment/run_ablation*.py first.")

fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 4.5), squeeze=False)
axes = axes[0]

for ax, (conditions, data) in zip(axes, available):
    x = np.arange(len(conditions))
    width = 0.22
    offsets = np.linspace(-(len(METRICS) - 1) / 2 * width,
                          (len(METRICS) - 1) / 2 * width, len(METRICS))
    for i, (m, color) in enumerate(zip(METRICS, COLORS)):
        avgs, stds = data[m]["avg"], data[m]["std"]
        ax.bar(x + offsets[i], avgs, width,
               label=m, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.5,
               yerr=stds, capsize=3, error_kw={"elinewidth": 1, "ecolor": "gray"})
        for j, (v, s) in enumerate(zip(avgs, stds)):
            if np.isnan(v):
                continue
            ax.text(x[j] + offsets[i], v + (s or 0) + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8.5)
    ax.set_ylabel("Score", fontsize=10)

    # Scale to the data rather than a fixed window -- the pre-revision limits
    # (0.74-1.02) clipped the lower conditions once the corrected numbers came in.
    vals = [v for m in METRICS for v in data[m]["avg"] if not np.isnan(v)]
    errs = [e for m in METRICS for e in data[m]["std"]]
    lo, hi = min(vals) - max(errs or [0]) - 0.04, max(vals) + max(errs or [0]) + 0.05
    ax.set_ylim(max(0.0, lo), min(1.05, hi))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.025))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0.85, loc="lower left",
              ncol=3, columnspacing=0.8, handlelength=1.2)

fig.tight_layout()
out = "figures/fig_ablation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
