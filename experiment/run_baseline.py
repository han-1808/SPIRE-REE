"""
experiment/run_baseline.py

Transductive baseline evaluation: leave-one-region-out sweep.

For each region, that region is held out as test; remaining nodes are split
into train / val (val_ratio).  Feature pipeline (scaler, imputer, XGBoost,
SHAP selection) is fitted on train_idx only — no leakage into test or val.

Models
------
  Tabular : XGBoost, LightGBM, CatBoost, RandomForest, SVM, MLP, Autoencoder
  Spatial : Spatial KNN (Haversine), Kriging

Results are saved to result/baseline.json.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import json

import numpy as np
import pandas as pd

from config import BASELINE_METRICS_PATH, OUTPUT_PATH
from feature_engineering import (
    _make_single_xgb_model,
    compute_ensemble_scores_no_leakage,
    create_spatial_split_indices,
    prepare_xgboost_inputs,
    train_xgboost_and_select_features,
)
from baseline import (
    train_autoencoder_predictor_baseline,
    train_catboost_baseline,
    train_kriging_baseline,
    train_lightgbm_baseline,
    train_mlp_baseline,
    train_random_forest_baseline,
    train_spatial_knn_regression_baseline,
    train_svm_baseline,
    train_xgboost_baseline,
)


MODELS = [
    "xgboost", "lightgbm", "catboost", "random_forest",
    "svm", "mlp", "autoencoder_predictor",
    "spatial_knn_regression", "kriging",
]
N_RUNS = 5


# ── Feature building (per region, leakage-safe) ───────────────────────────────

def _build_region_inputs(processed_df: pd.DataFrame, regions: list[str], seed: int = 42) -> dict:
    """
    Build baseline features for every region, fitting pipeline on train_idx only.

    seed : (Comment 17 fairness fixes)
      1. Threaded into `create_spatial_split_indices` so the train/val split
         is re-randomized per seed, like SPIRE's `_build_graph_artifacts`
         already does — previously this always used the default seed=42
         regardless of which "seed run" was in progress, so all 5 baseline
         runs silently shared one split and only the downstream models'
         own randomness varied (not a fair comparison to SPIRE's 5 runs).
      2. `xgb_score` is now computed leakage-free via
         `compute_ensemble_scores_no_leakage(..., models_factory=_make_single_xgb_model)`
         — train rows get a K-fold out-of-fold score, val/test rows get the
         score from the model refit on the full train set. Previously
         `xgb_model.predict_proba(X)` scored train rows in-sample (the same
         bug Comment 12 fixed for SPIRE's `ensemble_score`, but never
         patched here since `build_baseline_features` was outside that fix's
         scope).
    """
    region_inputs = {}
    for region in regions:
        print(f"  [build] {region}")
        split = create_spatial_split_indices(processed_df, test_region=region, val_ratio=0.2, random_seed=seed)
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]
        other_idx = np.concatenate([val_idx, test_idx])

        graph_df, X, y = prepare_xgboost_inputs(processed_df, fit_idx=train_idx)
        _, _, X_selected = train_xgboost_and_select_features(X, y, fit_idx=train_idx)

        train_scores, other_scores, _ = compute_ensemble_scores_no_leakage(
            X, y, train_idx, other_idx, random_state=seed,
            models_factory=_make_single_xgb_model,
        )
        xgb_score = np.empty(len(X), dtype=float)
        xgb_score[train_idx] = train_scores
        xgb_score[other_idx] = other_scores
        X_baseline = X_selected.copy()
        X_baseline["xgb_score"] = xgb_score

        region_inputs[region] = {
            "graph_df": graph_df,
            "X": X_baseline,
            "y": y,
        }
    return region_inputs


# ── Baseline runners ──────────────────────────────────────────────────────────

def _make_runners(processed_df: pd.DataFrame, region_inputs: dict, seed: int = 42) -> dict:
    ri = region_inputs
    return {
        "xgboost": lambda r: train_xgboost_baseline(
            graph_df=ri[r]["graph_df"], X=ri[r]["X"], y=ri[r]["y"],
            test_region=r, metrics_path=None, seed=seed,
        )["results"],
        "lightgbm": lambda r: train_lightgbm_baseline(
            graph_df=ri[r]["graph_df"], X=ri[r]["X"], y=ri[r]["y"],
            test_region=r, metrics_path=None, seed=seed,
        )["results"],
        "catboost": lambda r: train_catboost_baseline(
            graph_df=ri[r]["graph_df"], X=ri[r]["X"], y=ri[r]["y"],
            test_region=r, metrics_path=None, seed=seed,
        )["results"],
        "random_forest": lambda r: train_random_forest_baseline(
            graph_df=ri[r]["graph_df"], X=ri[r]["X"], y=ri[r]["y"],
            test_region=r, metrics_path=None, seed=seed,
        )["results"],
        "svm": lambda r: train_svm_baseline(
            graph_df=ri[r]["graph_df"], X=ri[r]["X"], y=ri[r]["y"],
            test_region=r, metrics_path=None, seed=seed,
        )["results"],
        "mlp": lambda r: train_mlp_baseline(
            graph_df=ri[r]["graph_df"], X=ri[r]["X"], y=ri[r]["y"],
            test_region=r, metrics_path=None, seed=seed,
        )["results"],
        "autoencoder_predictor": lambda r: train_autoencoder_predictor_baseline(
            graph_df=ri[r]["graph_df"], X=ri[r]["X"], y=ri[r]["y"],
            test_region=r, metrics_path=None, seed=seed,
        )["results"],
        "spatial_knn_regression": lambda r: train_spatial_knn_regression_baseline(
            processed_df=processed_df, graph_df=ri[r]["graph_df"], y=ri[r]["y"],
            test_region=r, metrics_path=None,
        )["results"],
        "kriging": lambda r: train_kriging_baseline(
            processed_df=processed_df, graph_df=ri[r]["graph_df"], y=ri[r]["y"],
            test_region=r, metrics_path=None,
        )["results"],
    }


def _avg_metrics(metrics_per_region: dict) -> dict:
    valid = {r: m for r, m in metrics_per_region.items() if m}
    if not valid:
        return {}
    keys = list(next(iter(valid.values())).keys())
    return {k: sum(v[k] for v in valid.values()) / len(valid) for k in keys}


# ── Main sweep ─────────────────────────────────────────────────────────────────

def _run_one_seed(
    processed_df: pd.DataFrame,
    regions: list,
    seed: int,
    skip_kriging: bool = False,
) -> dict:
    """
    Run all baseline models for one seed. Returns {avg_metrics, per_region}.

    Rebuilds region_inputs (features + split) fresh for this seed — see
    `_build_region_inputs` docstring (Comment 17 fairness fix).
    """
    print(f"\nBuilding per-region inputs for seed={seed}...")
    region_inputs = _build_region_inputs(processed_df, regions, seed=seed)
    runners = _make_runners(processed_df, region_inputs, seed=seed)
    per_region: dict = {r: {} for r in regions}
    models_to_run = [m for m in MODELS if not (skip_kriging and m == "kriging")]

    for model_name in models_to_run:
        print(f"\n=== {model_name} (seed={seed}) ===")
        for region in regions:
            print(f"  [{region}]")
            try:
                per_region[region][model_name] = runners[model_name](region)
            except Exception as e:
                print(f"  [WARN] {model_name} failed for {region}: {e}")
                per_region[region][model_name] = None

    avg_metrics = {
        model: _avg_metrics({
            r: (per_region[r][model].get("test_metrics") if per_region[r].get(model) else None)
            for r in regions
        })
        for model in models_to_run
    }
    return {"avg_metrics": avg_metrics, "per_region": per_region}


def run_baseline_sweep(
    processed_df: pd.DataFrame,
    metrics_path: str = BASELINE_METRICS_PATH,
    skip_kriging: bool = False,
    seeds: list[int] | None = None,
) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    print(f"Regions ({len(regions)}): {regions}")

    if seeds is None:
        seeds = [42] + random.sample([s for s in range(61) if s != 42], N_RUNS - 1)
    print(f"Seeds for this run: {seeds}")

    runs: dict = {}
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"  RUN seed={seed}")
        print(f"{'#'*60}")
        runs[str(seed)] = _run_one_seed(processed_df, regions, seed, skip_kriging)

    # Merge with any existing results file so a partial re-run (e.g. a
    # single seed) doesn't discard other seeds already saved there.
    if Path(metrics_path).exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing_runs = json.load(f).get("runs", {})
        runs = {**existing_runs, **runs}
        print(f"Merged with existing results → {len(runs)} seed(s) total: {sorted(runs.keys(), key=int)}")

    # Cross-seed average metrics
    models_to_run = [m for m in MODELS if not (skip_kriging and m == "kriging")]
    metric_sample = next(iter(runs.values()))["avg_metrics"]
    avg_metrics: dict = {}
    for m in models_to_run:
        keys = list((metric_sample.get(m) or {}).keys())
        avg_metrics[m] = {
            k: float(np.mean([v["avg_metrics"][m][k] for v in runs.values()
                               if (v["avg_metrics"].get(m) or {}).get(k) is not None]))
            for k in keys
        } if keys else {}

    # Top-level per_region from first run for convenience
    per_region = next(iter(runs.values()))["per_region"]

    payload = {
        "regions": regions,
        "per_region": per_region,
        "avg_metrics": avg_metrics,
        "runs": runs,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nBaseline results saved → {metrics_path}")
    return payload


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-kriging", action="store_true", help="Skip Kriging (slow)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                         help="Explicit seed list (default: 42 + 4 random)")
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    run_baseline_sweep(processed_df, skip_kriging=args.skip_kriging, seeds=args.seeds)



if __name__ == "__main__":
    main()
