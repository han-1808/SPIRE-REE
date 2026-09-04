"""
experiment/run_exp6_proportion_inductive_sample.py

Exp 6 — Global inductive proportion prediction, random split.

Same data split as Exp 3 (run_inductive_sample.py):
  - 5 nodes randomly sampled from each region → test set.
  - ALL remaining nodes (including the rest of the same region) → global train+val.
  - GNNs (SPIRE / GCN / GAT) predicted inductively; 6 tabular baselines
    trained on the global train set.
  - Predicted proportion = mean(y_prob) over the 5 test nodes per region.
  - Actual proportion   = fraction of has_ree=1 among those 5 test nodes.
  - Metric: MAE and RMSE of |predicted − actual| across 10 regions.

Results saved to result/exp6_proportion_inductive_sample.json.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import json

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from baseline import (
    predict_proportion_autoencoder,
    predict_proportion_kriging,
    predict_proportion_spatial_knn,
)
from config import OUTPUT_PATH
from feature_engineering import (
    _make_single_xgb_model,
    build_node_features,
    compute_ensemble_scores_no_leakage,
    prepare_xgboost_inputs,
    select_geo_similarity_columns,
    train_xgboost_and_select_features,
)
from gnn_training import GAT, GCN, GraphSAGE
from run_gnn_per_region import _predict_node_aug
from run_inductive_sample import (
    N_SAMPLE,
    SEED,
    _build_graph_artifacts,
    _make_sample_split,
    _train_gnn_model_on_artifacts,
)

_RESULTS_PATH = "result/exp6_proportion_inductive_sample.json"
_GNN_CLASSES = [("spire", GraphSAGE), ("gcn", GCN), ("gat", GAT)]
N_RUNS = 5


def _make_baseline_models(seed: int = 42) -> dict:
    return {
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            loss_function="Logloss", random_seed=seed, verbose=False,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1,
        ),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, kernel="rbf", random_state=seed)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)),
        ]),
    }


def _run_region(
    processed_df: pd.DataFrame,
    split: dict,
    actual_proportion: float,
    seed: int = 42,
) -> dict:
    """Run all baselines + GNNs for one region and return proportion metrics."""
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]

    # ── Shared feature pipeline (fitted on train only) ────────────────────────
    _, X, y = prepare_xgboost_inputs(processed_df, fit_idx=train_idx)
    _, _, X_selected = train_xgboost_and_select_features(X, y, fit_idx=train_idx)
    # Leakage-free xgb_score (Comment 17 fairness fix, same OOF machinery as
    # Comment 12-A): train rows get a K-fold out-of-fold score, val/test
    # rows get the score from the model refit on the full train_idx.
    other_idx = np.concatenate([val_idx, test_idx])
    train_scores, other_scores, _ = compute_ensemble_scores_no_leakage(
        X, y, train_idx, other_idx, random_state=seed,
        models_factory=_make_single_xgb_model,
    )
    xgb_score = np.empty(len(X), dtype=float)
    xgb_score[train_idx] = train_scores
    xgb_score[other_idx] = other_scores
    X_bl = X_selected.copy()
    X_bl["xgb_score"] = xgb_score

    X_train = X_bl.iloc[train_idx].astype(float)
    y_train = y.iloc[train_idx]
    X_test  = X_bl.iloc[test_idx].astype(float)

    per_model: dict = {}

    # ── Tabular baselines ─────────────────────────────────────────────────────
    for name, model in _make_baseline_models(seed=seed).items():
        try:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            pred_prop = float(np.mean(y_prob))
            abs_err   = float(abs(pred_prop - actual_proportion))
            per_model[name] = {
                "predicted_proportion": pred_prop,
                "abs_error":            abs_err,
                "y_prob":               y_prob.tolist(),  # Comment 18-B
            }
        except Exception as exc:
            per_model[name] = {"error": str(exc)}
        info = per_model[name]
        if "abs_error" in info:
            print(f"    [{name}]  pred={info['predicted_proportion']:.3f}  |err|={info['abs_error']:.3f}")
        else:
            print(f"    [{name}]  ERROR: {info.get('error')}")

    # ── Spatial / autoencoder baselines ─────────────────────────────────────
    coords_train = processed_df[["Latitude", "Longitude"]].values[train_idx]
    coords_test  = processed_df[["Latitude", "Longitude"]].values[test_idx]
    y_train_arr  = y_train.to_numpy()

    for name, fn, kwargs in [
        ("autoencoder", predict_proportion_autoencoder,
         {"X_train": X_train.values, "X_test": X_test.values, "y_train": y_train_arr}),
        ("spatial_knn", predict_proportion_spatial_knn,
         {"coords_train": coords_train, "coords_test": coords_test, "y_train": y_train_arr}),
        ("kriging",     predict_proportion_kriging,
         {"coords_train": coords_train, "coords_test": coords_test, "y_train": y_train_arr}),
    ]:
        try:
            y_prob    = fn(**kwargs)
            pred_prop = float(np.mean(y_prob))
            abs_err   = float(abs(pred_prop - actual_proportion))
            per_model[name] = {
                "predicted_proportion": pred_prop,
                "abs_error":            abs_err,
                "y_prob":               np.asarray(y_prob).tolist(),  # Comment 18-B
            }
        except Exception as exc:
            per_model[name] = {"error": str(exc)}
        info = per_model[name]
        if "abs_error" in info:
            print(f"    [{name}]  pred={info['predicted_proportion']:.3f}  |err|={info['abs_error']:.3f}")
        else:
            print(f"    [{name}]  ERROR: {info.get('error')}")

    # ── GNNs (inductive) ──────────────────────────────────────────────────────
    graph_artifacts = _build_graph_artifacts(processed_df, split, seed=seed)
    test_coords     = processed_df[["Latitude", "Longitude"]].values[test_idx]
    X_selected_test = graph_artifacts["X_selected_test"]
    X_test_raw      = graph_artifacts["X_test"]
    ensemble_models = graph_artifacts["ensemble_models"]

    best_config = None
    for name, model_class in _GNN_CLASSES:
        label = "Tuning+Training" if best_config is None else f"Training (shared config={best_config})"
        print(f"    [{name.upper()}] {label}")

        arts, found_config = _train_gnn_model_on_artifacts(model_class, graph_artifacts, best_config)
        if best_config is None:
            best_config = found_config

        y_probs = []
        for i in range(len(test_idx)):
            ens_score = float(np.mean([
                m.predict_proba(X_test_raw.iloc[[i]])[0, 1] for m in ensemble_models
            ]))
            x_sel = torch.tensor(
                X_selected_test.iloc[i].astype(float).values, dtype=torch.float
            ).unsqueeze(0)
            x_new_geo = torch.tensor(
                select_geo_similarity_columns(X_selected_test.iloc[[i]]).astype(float).values,
                dtype=torch.float,
            )
            x_ens = torch.tensor([[ens_score]], dtype=torch.float)
            x_new_base = torch.cat([x_sel, x_ens], dim=1)
            lat, lon = test_coords[i]
            y_probs.append(_predict_node_aug(arts, lat, lon, x_new_base, x_new_geo))

        pred_prop = float(np.mean(y_probs))
        abs_err   = float(abs(pred_prop - actual_proportion))
        per_model[name] = {
            "predicted_proportion": pred_prop,
            "abs_error":            abs_err,
            "y_prob":               [float(p) for p in y_probs],  # Comment 18-B
        }
        print(f"    [{name}]  pred={pred_prop:.3f}  |err|={abs_err:.3f}")

    return per_model


def _aggregate_metrics(per_region: dict) -> dict:
    model_names = [
        "xgboost", "lightgbm", "catboost", "random_forest", "svm", "mlp",
        "autoencoder", "spatial_knn", "kriging",
        "spire", "gcn", "gat",
    ]
    result = {}
    for m in model_names:
        errors = []
        for region_data in per_region.values():
            mm = region_data.get("per_model", {}).get(m, {})
            if mm and "abs_error" in mm:
                errors.append(mm["abs_error"])
        if errors:
            result[m] = {
                "mae":       float(np.mean(errors)),
                "rmse":      float(np.sqrt(np.mean(np.square(errors)))),
                "n_regions": len(errors),
            }
        else:
            result[m] = None
    return result


def run_exp6(
    processed_df: pd.DataFrame,
    n_sample: int = N_SAMPLE,
    results_path: str = _RESULTS_PATH,
    seeds: list[int] | None = None,
) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    print(f"Regions ({len(regions)}): {regions}")

    if seeds is None:
        seeds = [42] + random.sample([s for s in range(61) if s != 42], N_RUNS - 1)
    print(f"Seeds for this run: {seeds}")

    runs: list = []
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"  RUN seed={seed}  n_sample={n_sample}  sampling=random")
        print(f"{'#'*60}")
        import torch as _torch
        _torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        per_region: dict = {}
        for i, region in enumerate(regions):
            print(f"\n{'='*60}")
            print(f"  Test region: {region}  (seed={seed})")

            split = _make_sample_split(processed_df, region, n_sample, rng, val_seed=seed + i)
            n = split["n"]
            actual_proportion = float(processed_df["has_ree"].values[split["test_idx"]].mean())
            print(f"  n_test={n}  actual_proportion={actual_proportion:.3f}")

            per_model = _run_region(processed_df, split, actual_proportion, seed=seed)

            per_region[region] = {
                "n_test":            int(n),
                "actual_proportion": actual_proportion,
                "sample_idx":        split["test_idx"].tolist(),
                "y_true":            processed_df["has_ree"].values[split["test_idx"]].astype(int).tolist(),  # Comment 18-B
                "per_model":         per_model,
            }

        agg = _aggregate_metrics(per_region)
        print(f"\n  Aggregated metrics (seed={seed}):")
        for m, v in agg.items():
            if v:
                print(f"    {m:20s}  MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}")
        runs.append({"seed": seed, "aggregated_metrics": agg, "per_region": per_region})

    # Merge with any existing results file so a partial re-run (e.g. a
    # single seed) doesn't discard other seeds already saved there.
    if Path(results_path).exists():
        with open(results_path, encoding="utf-8") as f:
            existing_runs = json.load(f).get("runs", [])
        by_seed = {r["seed"]: r for r in existing_runs}
        for r in runs:
            by_seed[r["seed"]] = r  # new runs overwrite same-seed old ones
        runs = [by_seed[s] for s in sorted(by_seed.keys())]
        print(f"Merged with existing results → {len(runs)} seed(s) total: {sorted(by_seed.keys())}")

    payload = {
        "experiment": "exp6_proportion_inductive_sample",
        "description": (
            f"Global inductive proportion prediction, random split (same as Exp 3). "
            f"{n_sample} nodes randomly sampled from each region as test; "
            "all remaining nodes form the global train+val set. "
            "GNNs predicted inductively. "
            "predicted_proportion = mean(y_prob) over test nodes. "
            "Metric: MAE and RMSE of |predicted − actual| across 10 regions."
        ),
        "n_sample": n_sample,
        "sampling": "random",
        "models": [
            "xgboost", "lightgbm", "catboost", "random_forest", "svm", "mlp",
            "autoencoder", "spatial_knn", "kriging",
            "spire", "gcn", "gat",
        ],
        "regions": regions,
        "runs": runs,
    }

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {results_path}")
    return payload


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sample", type=int, default=N_SAMPLE)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                         help="Explicit seed list (default: 42 + 4 random)")
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)
    run_exp6(processed_df, n_sample=args.n_sample, seeds=args.seeds)


if __name__ == "__main__":
    main()
