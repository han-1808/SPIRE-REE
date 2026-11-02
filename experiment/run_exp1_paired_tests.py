"""
experiment/run_exp1_paired_tests.py

Reviewer #1, second round, Remark 2: region-level comparison of SPIRE with
each baseline in Experiment 1 (leave-one-region-out).

Needs no model: reads the stored per-region test metrics of
`gnn_inductive.json` (SPIRE) and `baseline.json` (tabular and spatial
baselines). The two files use different seeds (SPIRE 42, 1, 2, 4, 7;
baselines 42, 19, 34, 36, 18), so runs cannot be paired by seed. Instead,
each model's metric is averaged over its five runs within each held-out
region, and the ten regions form the pairs of a two-sided Wilcoxon
signed-rank test.

For each baseline and metric (F1, ROC-AUC, PR-AUC) the script reports the
mean region-level difference (SPIRE - baseline), the number of regions where
SPIRE is higher, and the Wilcoxon statistic and p-value. For each model it
also reports the worst-region metric and the between-region standard
deviation (population SD over the ten region means). Seeds of both sides are
recorded. p-values are not corrected for multiple comparisons.

Usage:
    python experiment/run_exp1_paired_tests.py

Results saved to result/exp1_paired_tests.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

_GNN_PATH = "result/gnn_inductive.json"
_BASELINE_PATH = "result/baseline.json"
_OUT_PATH = "result/exp1_paired_tests.json"

_METRICS = ["f1", "roc_auc", "pr_auc"]


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _region_means(runs: dict, regions: list, get_metrics) -> dict:
    """Mean of each metric over runs, per region: {metric: [value per region]}."""
    out = {m: [] for m in _METRICS}
    for region in regions:
        per_run = [get_metrics(run["per_region"][region]) for run in runs.values()]
        for m in _METRICS:
            out[m].append(float(np.mean([pm[m] for pm in per_run])))
    return out


def _summary(values: list, regions: list) -> dict:
    i_min = int(np.argmin(values))
    return {
        "mean_over_regions": float(np.mean(values)),
        "worst_region": regions[i_min],
        "worst_region_value": float(values[i_min]),
        "between_region_sd": float(np.std(values)),
    }


def _paired(spire: list, other: list) -> dict:
    diff = np.asarray(spire) - np.asarray(other)
    w_stat, w_p = wilcoxon(spire, other)
    return {
        "mean_diff": float(diff.mean()),
        "n_regions_spire_higher": int((diff > 0).sum()),
        "n_regions": int(len(diff)),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p": float(w_p),
    }


def run_exp1_paired_tests() -> dict:
    gnn = _load(_GNN_PATH)
    baseline = _load(_BASELINE_PATH)
    regions = sorted(gnn["regions"])
    if sorted(baseline["regions"]) != regions:
        raise SystemExit("Region lists of the two result files differ")

    region_means = {
        "spire": _region_means(gnn["runs"], regions,
                               lambda r: r["test_metrics"]["spire"])
    }
    for model in baseline["avg_metrics"]:
        region_means[model] = _region_means(
            baseline["runs"], regions, lambda r, m=model: r[m]["test_metrics"])

    results = {
        "experiment": "exp1_paired_tests",
        "description": (
            "Experiment 1 (leave-one-region-out). Per-region metrics averaged over "
            "five runs per model; two-sided Wilcoxon signed-rank test over the ten "
            "regions, SPIRE minus baseline. Runs are not paired by seed because the "
            "two result files use different seeds. p-values are uncorrected."
        ),
        "seeds": {
            "spire": sorted(int(s) for s in gnn["runs"]),
            "baselines": sorted(int(s) for s in baseline["runs"]),
        },
        "source_files": {"spire": _GNN_PATH, "baselines": _BASELINE_PATH},
        "regions": regions,
        "per_model": {},
        "spire_vs_baseline": {},
    }

    for model, means in region_means.items():
        results["per_model"][model] = {
            m: {"per_region": dict(zip(regions, means[m])), **_summary(means[m], regions)}
            for m in _METRICS
        }

    print(f"{'baseline':<24}" + "".join(f"{m:>26}" for m in _METRICS))
    for model in baseline["avg_metrics"]:
        results["spire_vs_baseline"][model] = {
            m: _paired(region_means["spire"][m], region_means[model][m]) for m in _METRICS
        }
        row = results["spire_vs_baseline"][model]
        print(f"{model:<24}" + "".join(
            f"   {row[m]['mean_diff']:+.3f} ({row[m]['n_regions_spire_higher']}/10) "
            f"p={row[m]['wilcoxon_p']:.3f}" for m in _METRICS))

    print("\nworst-region F1 / between-region SD of F1")
    for model, summary in results["per_model"].items():
        f1 = summary["f1"]
        print(f"  {model:<24} {f1['worst_region_value']:.3f} ({f1['worst_region']})  "
              f"SD {f1['between_region_sd']:.3f}")

    Path(_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {_OUT_PATH}")
    return results


if __name__ == "__main__":
    run_exp1_paired_tests()
