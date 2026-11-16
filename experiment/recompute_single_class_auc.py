"""
experiment/recompute_single_class_auc.py

Corrects the Experiment 2 ROC-AUC and PR-AUC summaries for single-class test
folds without retraining any model.

Earlier versions of run_gnn_per_region.py and run_baseline_per_region.py
assigned ROC-AUC = PR-AUC = 1 to a single-class test fold when every
thresholded prediction was correct and 0 otherwise. Neither metric is defined
for a single-class fold, so the substituted value depended on the threshold
rather than on the ranking. The current scripts store None for such folds and
leave them out of the regional mean, the rule already used in Experiment 3
(run_inductive_sample.py).

This script applies that rule to the saved results. For each run and region it
reads the stored test labels; if the fold holds one class, the fold's ROC-AUC
and PR-AUC are set to None for every model. The per-run regional means
(avg_metrics) are then recomputed with the same averaging functions the
experiment scripts use. Accuracy, precision, recall and F1 are unchanged, as
are the pooled and region-cluster ROC-AUC analyses, which use the stored
predictions directly.

Usage:
    python experiment/recompute_single_class_auc.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import json

import numpy as np

from run_baseline_per_region import _avg_metrics as _baseline_avg_metrics
from run_gnn_per_region import _GNN_MODELS, _avg_model_metrics

_GNN_PATH = "result/gnn_per_region.json"
_BASELINE_PATH = "result/baseline_per_region.json"
_UNDEFINED = ("roc_auc", "pr_auc")


def _fold_labels(region_entry: dict) -> list | None:
    """Return the stored test labels of a fold (identical across models)."""
    for pred in (region_entry.get("predictions") or {}).values():
        if isinstance(pred, dict) and "y_true" in pred:
            return pred["y_true"]
    return None


def _is_single_class(region_entry: dict) -> bool:
    y_true = _fold_labels(region_entry)
    return y_true is not None and len(np.unique(y_true)) < 2


def _clear(metrics: dict) -> None:
    for k in _UNDEFINED:
        if k in metrics:
            metrics[k] = None


def _fix_gnn(data: dict) -> list:
    changed = []
    for run in data["runs"]:
        per_region = run["per_region"]
        for region, entry in per_region.items():
            if entry and _is_single_class(entry):
                for metrics in entry.get("test_metrics", {}).values():
                    _clear(metrics)
                changed.append((run["seed"], region, len(_fold_labels(entry))))
        run["avg_metrics"] = {
            name: _avg_model_metrics(per_region, name) for name, _ in _GNN_MODELS
        }
    return changed


def _fix_baseline(data: dict) -> list:
    changed = []
    for run in data["runs"]:
        per_region = run["per_region"]
        for region, entry in per_region.items():
            if entry and _is_single_class(entry):
                for metrics in entry.get("per_model", {}).values():
                    _clear(metrics)
                changed.append((run["seed"], region, len(_fold_labels(entry))))
        run["avg_metrics"] = _baseline_avg_metrics(per_region)
    return changed


def _summary(data: dict, models: list) -> dict:
    out = {}
    for m in models:
        vals = [(r["avg_metrics"].get(m) or {}).get("roc_auc") for r in data["runs"]]
        vals = [v for v in vals if v is not None]
        out[m] = (float(np.mean(vals)), float(np.std(vals))) if vals else None
    return out


def main():
    for path, fixer in ((_GNN_PATH, _fix_gnn), (_BASELINE_PATH, _fix_baseline)):
        with open(path) as f:
            data = json.load(f)
        models = (
            [name for name, _ in _GNN_MODELS] if fixer is _fix_gnn else data["models"]
        )
        before = _summary(data, models)
        changed = fixer(data)
        after = _summary(data, models)

        data["_note_single_class_folds"] = (
            "ROC-AUC and PR-AUC are undefined for a single-class test fold. Such "
            "folds store None and are omitted from avg_metrics (regional mean). "
            "Applied by experiment/recompute_single_class_auc.py. Folds affected "
            "(seed, region, n_test): "
            + ", ".join(f"({s}, {r}, {n})" for s, r, n in changed)
        )
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n{path}: single-class folds = {changed}")
        print(f"{'model':<26}{'ROC-AUC before':>18}{'ROC-AUC after':>18}")
        for m in models:
            b, a = before[m], after[m]
            fmt = lambda x: f"{x[0]:.4f} ± {x[1]:.4f}" if x else "n/a"
            print(f"{m:<26}{fmt(b):>18}{fmt(a):>18}")


if __name__ == "__main__":
    main()
