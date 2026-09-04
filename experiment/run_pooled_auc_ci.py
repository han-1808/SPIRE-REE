"""
experiment/run_pooled_auc_ci.py

Comment (4)-D: pool predictions across runs, recompute ROC-AUC/PR-AUC with
a 95% CI instead of a single point estimate.

Directly answers the reviewer's "ROC-AUC/PR-AUC ~1.000 is an artifact of
sample size, not genuine performance" concern: pools per-node (y_true,
y_prob) across every (seed, region) pair for a given model, computes the
point estimate on the pooled set, and a bootstrap 95% CI (node-level
resampling with replacement, 1000 resamples by default) -- a much larger,
more stable sample than any single 5-node (now ~15%-of-region, Comment
4-A) fold alone.

Requires the Comment (4)-D per-node instrumentation: `run_gnn_per_region.py`
/ `run_baseline_per_region.py` / `run_inductive_sample.py` (added here),
`run_exp5_proportion_per_region.py` / `run_exp6_proportion_inductive_sample.py`
(already had it from Comment 18-B). Older result files (missing the new
fields) are handled gracefully -- that experiment/model is skipped and
reported, not a crash.

Note on the bootstrap: resampling is done at the flat pooled-node level,
which is the standard, simple approach -- it does not account for the
hierarchical clustering (nodes within a region within a seed), so the CI
should be read as a reasonable approximation, not an exact inference,
consistent with how bootstrap CIs are typically reported for this kind of
pooled analysis.

Usage:
    python experiment/run_pooled_auc_ci.py

Results saved to result/pooled_auc_ci.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_N_BOOTSTRAP = 1000
_CI_PERCENTILES = (2.5, 97.5)
_BOOTSTRAP_SEED = 42

_EXP_FILES = {
    "exp2_gnn":      "result/gnn_per_region.json",
    "exp2_baseline": "result/baseline_per_region.json",
    "exp3":          "result/inductive_sample.json",
    "exp5":          "result/exp5_proportion_per_region.json",
    "exp6":          "result/exp6_proportion_inductive_sample.json",
}
_OUT_PATH = "result/pooled_auc_ci.json"


def _bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap: int = _N_BOOTSTRAP) -> dict | None:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return None  # metric undefined without both classes present

    point = float(metric_fn(y_true, y_prob))
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue  # skip resamples that happen to be single-class
        scores.append(metric_fn(yt, yp))
    if not scores:
        return {"point": point, "ci_lower": None, "ci_upper": None, "n_bootstrap_valid": 0}
    lo, hi = np.percentile(scores, _CI_PERCENTILES)
    return {"point": point, "ci_lower": float(lo), "ci_upper": float(hi),
            "n_bootstrap_valid": len(scores)}


def _add(pooled: dict, model: str, y_true, y_prob) -> None:
    bucket = pooled.setdefault(model, {"y_true": [], "y_prob": []})
    bucket["y_true"].extend(y_true)
    bucket["y_prob"].extend(y_prob)


def _pool_exp2(payload: dict) -> dict:
    """gnn_per_region.json / baseline_per_region.json: runs -> per_region ->
    {"predictions": {model: {y_true, y_prob}}}."""
    pooled: dict = {}
    for run in payload.get("runs", []):
        for region_data in run.get("per_region", {}).values():
            for model, p in (region_data or {}).get("predictions", {}).items():
                if p and p.get("y_prob") is not None:
                    _add(pooled, model, p["y_true"], p["y_prob"])
    return pooled


def _pool_exp3(payload: dict) -> dict:
    """inductive_sample.json: runs -> {spire,gcn,gat}.per_region -> {"predictions"},
    and runs -> baselines.predictions_per_region -> region -> {model: {...}}."""
    pooled: dict = {}
    for run in payload.get("runs", []):
        for gnn_name in ("spire", "gcn", "gat"):
            for region_data in run.get(gnn_name, {}).get("per_region", {}).values():
                p = (region_data or {}).get("predictions")
                if p and p.get("y_prob") is not None:
                    _add(pooled, gnn_name, p["y_true"], p["y_prob"])
        preds_per_region = run.get("baselines", {}).get("predictions_per_region", {})
        for model_preds in preds_per_region.values():
            for model, p in (model_preds or {}).items():
                if p and p.get("y_prob") is not None:
                    _add(pooled, model, p["y_true"], p["y_prob"])
    return pooled


def _pool_exp5_exp6(payload: dict) -> dict:
    """exp5/exp6: runs -> per_region -> {"y_true": [...], "per_model": {model: {"y_prob": [...]}}}
    (Comment 18-B's schema, reused as-is)."""
    pooled: dict = {}
    for run in payload.get("runs", []):
        for region_data in run.get("per_region", {}).values():
            y_true = (region_data or {}).get("y_true")
            if y_true is None:
                continue
            for model, info in (region_data or {}).get("per_model", {}).items():
                y_prob = (info or {}).get("y_prob")
                if y_prob is not None and len(y_prob) == len(y_true):
                    _add(pooled, model, y_true, y_prob)
    return pooled


_POOLERS = {
    "exp2_gnn": _pool_exp2, "exp2_baseline": _pool_exp2, "exp3": _pool_exp3,
    "exp5": _pool_exp5_exp6, "exp6": _pool_exp5_exp6,
}


def run_pooled_auc_ci(out_path: str = _OUT_PATH) -> dict:
    results: dict = {}
    for exp_name, path in _EXP_FILES.items():
        p = Path(path)
        if not p.exists():
            print(f"[{exp_name}] {path} not found -- skipped")
            continue
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)

        pooled = _POOLERS[exp_name](payload)
        exp_result: dict = {}
        for model, data in pooled.items():
            n = len(data["y_true"])
            if n == 0:
                continue
            roc = _bootstrap_ci(data["y_true"], data["y_prob"], roc_auc_score)
            pr = _bootstrap_ci(data["y_true"], data["y_prob"], average_precision_score)
            if roc is None and pr is None:
                continue
            exp_result[model] = {"n": n, "roc_auc": roc, "pr_auc": pr}
            roc_str = f"{roc['point']:.3f} [{roc['ci_lower']:.3f},{roc['ci_upper']:.3f}]" if roc else "n/a"
            print(f"  [{exp_name}] {model:<24} n={n:6d}  ROC-AUC={roc_str}")

        if exp_result:
            results[exp_name] = exp_result
        else:
            print(f"[{exp_name}] no usable (y_true, y_prob) pairs found "
                  f"(result file predates the Comment 4-D instrumentation)")

    Path("result").mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")
    return results


if __name__ == "__main__":
    run_pooled_auc_ci()
