"""
experiment/run_split_strategy_sensitivity.py

Comment (4)-B follow-up: justifies the spatial cross-validation protocol used
by Exp 2/3/5/6, by measuring the two candidate strategies side by side.

Reviewer #2 asked for "spatial block or buffered cross-validation". Both are
implemented in `create_buffered_region_split_indices`:

  buffer : hold out `test_ratio` of the region's nodes at RANDOM, then drop
           every train/val candidate within `buffer_km` of any test node.
  block  : cut the region into contiguous KMeans blocks and hold out whole
           blocks, then apply the same buffer ring around the held-out block.

They enforce the SAME guarantee -- no training node lies within `buffer_km`
of a test node -- but at very different cost. With randomly scattered test
points the buffer discs tile the region and take most of the training pool
with them; around one contiguous block the ring scales with the block's
perimeter instead.

That distinction matters for the paper's conclusions rather than just for
tidiness. Discarding two-thirds of the training nodes penalises SPIRE twice
over: it is more data-hungry than the tabular baselines, and the discarded
nodes are exactly the spatially adjacent neighbours its message passing runs
on. A "message passing does not help" result measured under the random-point
buffer would be an artefact of the split, not a finding -- so the two
strategies have to be shown to differ, and the protection has to be shown to
be equivalent.

This script reports, per region and strategy:
  - split geometry: n_train / n_val / n_test, nodes dropped by the buffer
  - protection    : min / mean haversine distance from each test node to its
                    nearest TRAINING node (the quantity the buffer exists to
                    control -- it must clear `buffer_km` under BOTH strategies,
                    otherwise the comparison is not like-for-like)
  - a fast tabular read-out: the same XGBoost+CatBoost+RandomForest ensemble
                    the pipeline uses, fit on train and scored on test

The read-out is the tabular ensemble rather than SPIRE: it is the model whose
training-set-size sensitivity is *lowest*, so whatever degradation it shows
under the random-point buffer is a floor on what the GNN suffers, and it runs
in seconds per region instead of requiring full GNN tuning.

Usage:
    python experiment/run_split_strategy_sensitivity.py
    python experiment/run_split_strategy_sensitivity.py --seeds 42 1 2
    python experiment/run_split_strategy_sensitivity.py --regions Oceania Africa

Results saved to result/split_strategy_sensitivity.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors

from config import OUTPUT_PATH
from feature_engineering import (
    _make_ensemble_models,
    create_buffered_region_split_indices,
    prepare_xgboost_inputs,
)

_EARTH_RADIUS_KM = 6371.0
_TEST_RATIO = 0.15
_GRAPH_K = 10  # matches run_gnn_per_region.py, whose split this mirrors
_MIN_REGION_NODES = 20
_STRATEGIES = ["block", "buffer"]
_DEFAULT_SEEDS = [42, 1, 2]
_RESULTS_PATH = "result/split_strategy_sensitivity.json"


def _test_to_train_distance_km(coords_test: np.ndarray, coords_train: np.ndarray) -> dict:
    """
    Haversine distance from every test node to its nearest TRAINING node.
    `min` is the binding number: it is what the buffer is supposed to hold
    above `buffer_km`, and it must hold under both strategies for the
    comparison below to be measuring training-set size rather than leakage.
    """
    nbrs = NearestNeighbors(n_neighbors=1, metric="haversine").fit(np.radians(coords_train))
    dist_rad, _ = nbrs.kneighbors(np.radians(coords_test))
    dist_km = dist_rad[:, 0] * _EARTH_RADIUS_KM
    return {
        "min_km": float(dist_km.min()),
        "mean_km": float(dist_km.mean()),
        "median_km": float(np.median(dist_km)),
    }


def _score_ensemble(X_region, y_region, train_local, test_local, seed):
    """Fit the pipeline's 3-model ensemble on train, score the test nodes."""
    y_train = y_region.iloc[train_local]
    y_test = y_region.iloc[test_local].to_numpy()
    if y_train.nunique() < 2 or len(np.unique(y_test)) < 2:
        return None  # single-class fold: metrics undefined, reported as null

    models = _make_ensemble_models(random_state=seed)
    X_train = X_region.iloc[train_local].astype(float)
    X_test = X_region.iloc[test_local].astype(float)
    probs = [m.fit(X_train, y_train).predict_proba(X_test)[:, 1] for m in models]
    y_prob = np.mean(probs, axis=0)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
    }


def _run_region(processed_df: pd.DataFrame, region: str, seed: int) -> dict:
    region_global_idx = np.where(processed_df["Region"].values == region)[0]
    n_nodes = len(region_global_idx)
    if n_nodes < _MIN_REGION_NODES:
        return {}

    coords_region = processed_df[["Latitude", "Longitude"]].values[region_global_idx]
    out = {"n_nodes": int(n_nodes)}

    for strategy in _STRATEGIES:
        split = create_buffered_region_split_indices(
            region_global_idx, coords_region, test_ratio=_TEST_RATIO,
            k_for_buffer=_GRAPH_K, random_seed=seed, split_strategy=strategy,
        )
        train_local = split["train_local"]
        test_local = split["test_local"]
        info = dict(split["split_info"])

        # Scaling/imputation is fit on this strategy's own training nodes, so
        # each arm gets the pipeline it would really run under. SHAP top-k
        # selection is deliberately NOT applied here: it is by far the most
        # expensive step, and letting each arm pick a different feature subset
        # would add a second source of variation on top of the training-set
        # size this script is trying to isolate. The ensemble reads the full
        # prepared matrix in both arms.
        train_global = region_global_idx[train_local]
        _, X_region, y_region = prepare_xgboost_inputs(processed_df, fit_idx=train_global)
        X_region = X_region.iloc[region_global_idx].reset_index(drop=True)
        y_region = y_region.iloc[region_global_idx].reset_index(drop=True)

        info["test_to_train_distance"] = _test_to_train_distance_km(
            coords_region[test_local], coords_region[train_local]
        )
        info["n_test_positive"] = int(y_region.iloc[test_local].sum())
        info["ensemble_metrics"] = _score_ensemble(
            X_region, y_region, train_local, test_local, seed
        )
        out[strategy] = info

    return out


def _aggregate(per_seed: dict) -> dict:
    """Mean over regions and seeds, per strategy."""
    agg: dict = {}
    for strategy in _STRATEGIES:
        n_train, n_test, excluded, min_km = [], [], [], []
        metrics: dict = {"f1": [], "roc_auc": [], "pr_auc": []}
        for regions in per_seed.values():
            for res in regions.values():
                if strategy not in res:
                    continue
                info = res[strategy]
                n_train.append(info["n_train"])
                n_test.append(info["n_test"])
                excluded.append(info["n_excluded_by_buffer"])
                min_km.append(info["test_to_train_distance"]["min_km"])
                if info["ensemble_metrics"]:
                    for k, v in info["ensemble_metrics"].items():
                        metrics[k].append(v)
        agg[strategy] = {
            "mean_n_train": float(np.mean(n_train)),
            "mean_n_test": float(np.mean(n_test)),
            "mean_n_excluded_by_buffer": float(np.mean(excluded)),
            "mean_min_test_to_train_km": float(np.mean(min_km)),
            "ensemble_metrics": {
                k: (float(np.mean(v)) if v else None) for k, v in metrics.items()
            },
            "n_folds": len(n_train),
        }
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()

    processed_df = pd.read_excel(OUTPUT_PATH)
    regions = args.regions or sorted(processed_df["Region"].dropna().unique().tolist())

    per_seed: dict = {}
    for seed in args.seeds:
        print(f"\n{'#'*60}\n# seed={seed}\n{'#'*60}")
        per_region: dict = {}
        for region in regions:
            res = _run_region(processed_df, region, seed)
            if not res:
                print(f"  {region}: skipped (too few nodes)")
                continue
            per_region[region] = res
            b, f = res["block"], res["buffer"]
            print(
                f"  {region:<24} n={res['n_nodes']:>4} | "
                f"block: train={b['n_train']:>4} test={b['n_test']:>3} "
                f"minkm={b['test_to_train_distance']['min_km']:>6.1f} | "
                f"buffer: train={f['n_train']:>4} test={f['n_test']:>3} "
                f"minkm={f['test_to_train_distance']['min_km']:>6.1f}"
            )
        per_seed[str(seed)] = per_region

    payload = {
        "config": {
            "test_ratio": _TEST_RATIO,
            "k_for_buffer": _GRAPH_K,
            "seeds": args.seeds,
            "strategies": _STRATEGIES,
            "readout_model": "XGBoost+CatBoost+RandomForest ensemble",
        },
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed),
    }
    Path(_RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*60}\nAGGREGATE (mean over regions x seeds)\n{'='*60}")
    for strategy, a in payload["aggregate"].items():
        m = a["ensemble_metrics"]
        print(
            f"  {strategy:<7} train={a['mean_n_train']:>7.1f} "
            f"test={a['mean_n_test']:>6.1f} "
            f"dropped_by_buffer={a['mean_n_excluded_by_buffer']:>7.1f} "
            f"min_test_to_train={a['mean_min_test_to_train_km']:>7.1f}km | "
            f"F1={m['f1']:.3f} ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f}"
        )
    print(f"\nSaved -> {_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
