"""
experiment/run_calibration_analysis.py

Comment (18)-B: Brier score + reliability plot.

Pools per-node (y_true, y_prob) pairs across all regions/seeds for
Exp4/5/6 (regional proportion prediction), using the per-node predictions
each script now saves (`y_true` per region, `y_prob` per model -- added
to run_exp4_proportion.py / run_exp5_proportion_per_region.py /
run_exp6_proportion_inductive_sample.py in the same change; previously
only the aggregate `predicted_proportion`/`abs_error` per region was
kept, discarding the per-node detail calibration needs).

For each experiment x model:
  - Brier score = mean((y_prob - y_true)^2) over all pooled
    (region, seed, node) triples.
  - A reliability/calibration curve: nodes are binned by predicted
    probability (10 equal-width bins by default); each bin's mean
    predicted probability is plotted against its observed positive
    frequency.

Requires result/exp4_proportion.json, result/exp5_proportion_per_region.json,
result/exp6_proportion_inductive_sample.json to have been (re)generated
with the Comment 18-B instrumentation -- older result files (or any
region/model missing `y_true`/`y_prob`) are skipped gracefully, reported
in the printout, not an error.

Usage:
    python experiment/run_calibration_analysis.py

Results saved to result/calibration_analysis.json, plot to
figures/fig_calibration.png.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_EXP_FILES = {
    "exp4": "result/exp4_proportion.json",
    "exp5": "result/exp5_proportion_per_region.json",
    "exp6": "result/exp6_proportion_inductive_sample.json",
}
_N_BINS = 10
_OUT_JSON = "result/calibration_analysis.json"
_OUT_PLOT = "figures/fig_calibration.png"


def _iter_runs(payload: dict):
    """Yields (seed, run) regardless of whether `runs` is dict-keyed-by-seed
    (exp4) or a list of {"seed":..., ...} entries (exp5/exp6)."""
    runs = payload.get("runs")
    if runs is None:
        return
    if isinstance(runs, dict):
        yield from runs.items()
    else:
        for run in runs:
            yield run.get("seed"), run


def _pool_predictions(payload: dict, exp_name: str) -> dict:
    """Returns {model_name: {"y_true": [...], "y_prob": [...]}} pooled
    across every (seed, region) in this experiment's result file."""
    pooled: dict = {}
    n_missing = 0
    for _seed, run in _iter_runs(payload):
        per_region = run.get("per_region", {}) if run else {}
        for _region, region_data in per_region.items():
            y_true = region_data.get("y_true") if region_data else None
            per_model = region_data.get("per_model", {}) if region_data else {}
            if y_true is None:
                n_missing += 1
                continue
            for model_name, info in per_model.items():
                y_prob = info.get("y_prob") if isinstance(info, dict) else None
                if y_prob is None or len(y_prob) != len(y_true):
                    continue
                bucket = pooled.setdefault(model_name, {"y_true": [], "y_prob": []})
                bucket["y_true"].extend(y_true)
                bucket["y_prob"].extend(y_prob)
    if n_missing:
        print(f"  [{exp_name}] {n_missing} region-run(s) missing `y_true` "
              f"(result file predates the Comment 18-B instrumentation, or "
              f"that region was skipped for other reasons) -- excluded from pooling")
    return pooled


def _brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def _reliability_bins(y_true, y_prob, n_bins: int = _N_BINS) -> list[dict]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        bins.append({
            "bin_range":          [float(edges[b]), float(edges[b + 1])],
            "n":                  n,
            "mean_predicted":     float(y_prob[mask].mean()),
            "observed_frequency": float(y_true[mask].mean()),
        })
    return bins


def run_calibration_analysis(out_json: str = _OUT_JSON, out_plot: str = _OUT_PLOT) -> dict:
    results: dict = {}
    for exp_name, path in _EXP_FILES.items():
        p = Path(path)
        if not p.exists():
            print(f"[{exp_name}] {path} not found -- skipped")
            continue
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        pooled = _pool_predictions(payload, exp_name)

        exp_result: dict = {}
        for model_name, data in pooled.items():
            n = len(data["y_true"])
            if n == 0:
                continue
            brier = _brier_score(data["y_true"], data["y_prob"])
            bins = _reliability_bins(data["y_true"], data["y_prob"])
            exp_result[model_name] = {"n": n, "brier_score": brier, "reliability_bins": bins}
            print(f"  [{exp_name}] {model_name:<16} n={n:6d}  Brier={brier:.4f}")
        if exp_result:
            results[exp_name] = exp_result
        else:
            print(f"[{exp_name}] no usable (y_true, y_prob) pairs found")

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    if results:
        _plot_reliability(results, out_plot)
    else:
        print("Nothing to plot — no experiment had usable per-node predictions "
              "(rerun Exp4/5/6 with the Comment 18-B instrumentation first).")
    return results


def _plot_reliability(results: dict, out_plot: str) -> None:
    exp_names = [e for e in _EXP_FILES if e in results]
    fig, axes = plt.subplots(1, len(exp_names), figsize=(6 * len(exp_names), 5), squeeze=False)
    axes = axes[0]
    for ax, exp_name in zip(axes, exp_names):
        exp_result = results[exp_name]
        model_name = "spire" if "spire" in exp_result else next(iter(exp_result))
        bins = exp_result[model_name]["reliability_bins"]
        if bins:
            x = [b["mean_predicted"] for b in bins]
            y = [b["observed_frequency"] for b in bins]
            sizes = [max(20, min(300, b["n"])) for b in bins]
            ax.scatter(x, y, s=sizes, alpha=0.8, edgecolors="black", linewidths=0.5, label=model_name)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        brier = exp_result[model_name]["brier_score"]
        ax.set_title(f"{exp_name} — {model_name} (Brier={brier:.3f})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out_plot}")


if __name__ == "__main__":
    run_calibration_analysis()
