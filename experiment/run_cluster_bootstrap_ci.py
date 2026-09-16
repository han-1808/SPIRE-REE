"""
experiment/run_cluster_bootstrap_ci.py

Replacement for the per-prediction bootstrap in run_pooled_auc_ci.py.

Why: that script resamples individual prediction rows i.i.d. from a flat concatenated
array. Two levels of structure are ignored. Predictions are nested within region and
within seed, and the SAME record is re-predicted under every seed -- in Experiment 2,
2,981 pooled "observations" correspond to only 2,036 distinct records. Treating those as
independent draws makes the intervals far too narrow, and the script reports only
marginal per-model intervals, so the overlap of two intervals was being read (including
by us) as evidence about a difference it never tested.

What this does instead:

  * The resampling unit is the REGION, chosen because the paper's claim is about
    generalisation to a geographic area the model has not seen. Ten regions are drawn
    with replacement; all predictions belonging to a drawn region -- across every seed --
    enter the resample together. Seeds are therefore repeats WITHIN a cluster, not
    independent observations, which is what they are.

  * Model comparison is done by bootstrapping the DIFFERENCE, using the same region draw
    for both models on every replicate so the comparison is paired. A CI for the
    difference that excludes zero is a test; two overlapping marginal CIs are not.

Honest limits, stated rather than buried:

  * There are only ten regions, so the cluster bootstrap is coarse and its intervals are
    wide. That width is the honest reflection of how little independent spatial
    replication this dataset provides; the narrow per-prediction intervals were an
    artefact of counting correlated predictions as independent.

  * For Experiments 3 and 6 the held-out nodes come from regions that are still
    represented in training, so the region cluster captures spatial dependence but the
    inferential target is not "a new region". Those intervals should be read as
    accounting for spatial clustering, not as unseen-region generalisation.

  * We do NOT claim the ten regions are independent of one another. Clustering at the
    region level removes the within-region dependence that dominates; residual
    between-region dependence (shared geology, shared compilation practice) remains.

Usage:
    python experiment/run_cluster_bootstrap_ci.py

Results saved to result/cluster_bootstrap_ci.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_N_BOOTSTRAP = 2000
_CI_PERCENTILES = (2.5, 97.5)
_SEED = 42

# Experiment 2 lives in two files (graph models, tabular baselines). They are merged
# into ONE store so SPIRE can be compared against the baselines with paired region draws.
# This is only legitimate now that both scripts read result/split_manifest_per_region.json
# and therefore evaluate on identical held-out blocks; before the manifest they drew
# independent seed lists and shared a single seed, so no paired comparison existed.
_EXP_FILES = {
    "exp2":          ["result/gnn_per_region.json", "result/baseline_per_region.json"],
    "exp3":          "result/inductive_sample.json",
    "exp5":          "result/exp5_proportion_per_region.json",
    "exp6":          "result/exp6_proportion_inductive_sample.json",
}
_MANIFEST_PATH = "result/split_manifest_per_region.json"
_OUT_PATH = "result/cluster_bootstrap_ci.json"
_METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}


# ── collection: keep the region label, unlike the flat pooling in run_pooled_auc_ci ──

def _add(store: dict, model: str, region: str, y_true, y_prob) -> None:
    store.setdefault(model, {}).setdefault(region, {"y_true": [], "y_prob": []})
    store[model][region]["y_true"].extend(y_true)
    store[model][region]["y_prob"].extend(y_prob)


def _collect_exp2(payload: dict) -> dict:
    store: dict = {}
    for run in payload.get("runs", []):
        for region, region_data in run.get("per_region", {}).items():
            for model, p in (region_data or {}).get("predictions", {}).items():
                if p and p.get("y_prob") is not None:
                    _add(store, model, region, p["y_true"], p["y_prob"])
    return store


def _collect_exp3(payload: dict) -> dict:
    store: dict = {}
    for run in payload.get("runs", []):
        for gnn_name in ("spire", "gcn", "gat"):
            for region, region_data in run.get(gnn_name, {}).get("per_region", {}).items():
                p = (region_data or {}).get("predictions")
                if p and p.get("y_prob") is not None:
                    _add(store, gnn_name, region, p["y_true"], p["y_prob"])
        for region, model_preds in (run.get("baselines", {})
                                    .get("predictions_per_region", {})).items():
            for model, p in (model_preds or {}).items():
                if p and p.get("y_prob") is not None:
                    _add(store, model, region, p["y_true"], p["y_prob"])
    return store


def _collect_exp5_exp6(payload: dict) -> dict:
    store: dict = {}
    for run in payload.get("runs", []):
        for region, region_data in run.get("per_region", {}).items():
            y_true = (region_data or {}).get("y_true")
            if y_true is None:
                continue
            for model, info in (region_data or {}).get("per_model", {}).items():
                y_prob = (info or {}).get("y_prob")
                if y_prob is not None and len(y_prob) == len(y_true):
                    _add(store, model, region, y_true, y_prob)
    return store


_COLLECTORS = {
    "exp2": _collect_exp2,
    "exp3": _collect_exp3, "exp5": _collect_exp5_exp6, "exp6": _collect_exp5_exp6,
}


# ── cluster bootstrap ──────────────────────────────────────────────────────────

def _concat(store_model: dict, regions) -> tuple[np.ndarray, np.ndarray]:
    yt, yp = [], []
    for r in regions:
        blk = store_model.get(r)
        if blk:
            yt.extend(blk["y_true"])
            yp.extend(blk["y_prob"])
    return np.asarray(yt), np.asarray(yp)


def _safe(metric_fn, yt, yp):
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return None
    return float(metric_fn(yt, yp))


def _cluster_ci(store_model: dict, regions: list, metric_fn, draws: np.ndarray) -> dict | None:
    yt, yp = _concat(store_model, regions)
    point = _safe(metric_fn, yt, yp)
    if point is None:
        return None
    scores = []
    for row in draws:
        ryt, ryp = _concat(store_model, [regions[i] for i in row])
        s = _safe(metric_fn, ryt, ryp)
        if s is not None:
            scores.append(s)
    if len(scores) < 100:
        return {"point": point, "ci_lower": None, "ci_upper": None,
                "n_valid": len(scores), "note": "too few valid resamples"}
    lo, hi = np.percentile(scores, _CI_PERCENTILES)
    return {"point": point, "ci_lower": float(lo), "ci_upper": float(hi),
            "n_valid": len(scores), "n_pred": int(len(yt)), "n_clusters": len(regions)}


def _paired_diff_ci(store: dict, a: str, b: str, regions: list,
                    metric_fn, draws: np.ndarray) -> dict | None:
    """Bootstrap the difference metric(a) - metric(b) on the SAME region draws."""
    if a not in store or b not in store:
        return None
    ya, pa = _concat(store[a], regions)
    yb, pb = _concat(store[b], regions)
    pt_a, pt_b = _safe(metric_fn, ya, pa), _safe(metric_fn, yb, pb)
    if pt_a is None or pt_b is None:
        return None
    diffs = []
    for row in draws:
        sub = [regions[i] for i in row]
        sa = _safe(metric_fn, *_concat(store[a], sub))
        sb = _safe(metric_fn, *_concat(store[b], sub))
        if sa is not None and sb is not None:
            diffs.append(sa - sb)
    if len(diffs) < 100:
        return None
    lo, hi = np.percentile(diffs, _CI_PERCENTILES)
    return {"model_a": a, "model_b": b,
            "point_a": pt_a, "point_b": pt_b, "diff": pt_a - pt_b,
            "ci_lower": float(lo), "ci_upper": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_valid": len(diffs)}


def main() -> int:
    rng = np.random.default_rng(_SEED)
    out: dict = {}

    for exp, path in _EXP_FILES.items():
        paths = path if isinstance(path, list) else [path]
        missing = [q for q in paths if not Path(q).exists()]
        if missing:
            print(f"[{exp}] {missing} not found -- skipped")
            continue
        store: dict = {}
        seed_sets = []
        for q in paths:
            with open(q, encoding="utf-8") as f:
                payload = json.load(f)
            runs = payload.get("runs", [])
            runs = list(runs.values()) if isinstance(runs, dict) else runs
            seed_sets.append({r.get("seed") for r in runs if r.get("seed") is not None})
            part = _COLLECTORS[exp](payload)
            for model, by_region in part.items():
                for region, blk in by_region.items():
                    _add(store, model, region, blk["y_true"], blk["y_prob"])
        if len(paths) > 1:
            # Refuse to pair across files that did not evaluate the same seeds.
            if len(set(map(frozenset, seed_sets))) != 1:
                raise SystemExit(
                    f"[{exp}] files disagree on seeds {seed_sets}: they did not evaluate "
                    f"the same held-out blocks, so a paired comparison is not defined. "
                    f"Rerun both against {_MANIFEST_PATH}."
                )
            print(f"   merged {len(paths)} files on shared seeds {sorted(seed_sets[0])}")
        if not store:
            print(f"[{exp}] no per-node predictions -- skipped")
            continue

        regions = sorted({r for m in store.values() for r in m})
        n_clusters = len(regions)
        draws = rng.integers(0, n_clusters, size=(_N_BOOTSTRAP, n_clusters))

        print(f"\n[{exp}] {n_clusters} region clusters, {len(store)} models")
        block: dict = {"n_clusters": n_clusters, "regions": regions,
                       "resampling_unit": "region", "n_bootstrap": _N_BOOTSTRAP,
                       "models": {}, "paired_differences": {}}

        for model in sorted(store):
            entry = {}
            for mname, fn in _METRICS.items():
                r = _cluster_ci(store[model], regions, fn, draws)
                if r:
                    entry[mname] = r
            if entry:
                block["models"][model] = entry
                roc = entry.get("roc_auc")
                if roc and roc.get("ci_lower") is not None:
                    print(f"   {model:<22} ROC-AUC {roc['point']:.3f} "
                          f"[{roc['ci_lower']:.3f},{roc['ci_upper']:.3f}]  "
                          f"n_pred={roc.get('n_pred')}")

        # Paired differences against SPIRE, where SPIRE is present in this file.
        if "spire" in store:
            for other in sorted(store):
                if other == "spire":
                    continue
                for mname, fn in _METRICS.items():
                    d = _paired_diff_ci(store, "spire", other, regions, fn, draws)
                    if d:
                        block["paired_differences"].setdefault(other, {})[mname] = d
            print("   -- paired differences (SPIRE - other), ROC-AUC --")
            for other, md in sorted(block["paired_differences"].items()):
                d = md.get("roc_auc")
                if d:
                    flag = "  *" if d["excludes_zero"] else ""
                    print(f"      vs {other:<20}{d['diff']:+.3f} "
                          f"[{d['ci_lower']:+.3f},{d['ci_upper']:+.3f}]{flag}")
        out[exp] = block

    Path(_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {_OUT_PATH}")
    print("\n'*' marks a difference whose 95% cluster-bootstrap CI excludes zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
