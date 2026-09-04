"""
experiment/run_exp4_proportion.py

Exp 4 — Region-level REE proportion prediction (leave-one-region-out).

Same data split as Exp 1 (create_spatial_split_indices): each region is held
out in turn; models are trained on the 9 remaining regions.

For the held-out region every node is predicted.  The average predicted
probability is treated as the model's estimate of the region's REE proportion
and compared to the true proportion (fraction of has_ree=1 nodes).

Metric: MAE and RMSE of |predicted_proportion − actual_proportion| across the
10 regions.

Both SPIRE (inductive, same mechanism as Exp 1) and 6 tabular baselines
(11 features: 10 SHAP + xgb_score, same as Exp 2) are evaluated.

Results saved to result/exp4_proportion.json.
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

from config import OUTPUT_PATH
from feature_engineering import (
    _make_single_xgb_model,
    build_node_features_no_leakage,
    build_weighted_graph,
    compute_ensemble_scores_no_leakage,
    create_spatial_split_indices,
    prepare_xgboost_inputs,
    select_geo_similarity_columns,
    train_xgboost_and_select_features,
)
from baseline import (
    predict_proportion_autoencoder,
    predict_proportion_kriging,
    predict_proportion_spatial_knn,
)
from gnn_training import (
    GAT,
    GCN,
    GraphSAGE,
    augment_with_edge_stats,
    build_pyg_data,
    build_train_val_subgraph,
    train_gnn_model,
    tune_gnn_model,
)
from run_gnn import _predict_node, _predict_test_nodes

_RESULTS_PATH = "result/exp4_proportion.json"
_GRAPH_K = 15
_GRAPH_ALPHA = 0.5
_GRAPH_MAX_DIST_KM = 2000.0
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


def _run_region(processed_df: pd.DataFrame, region: str, seed: int = 42) -> dict:
    # random_seed=seed (Comment 17 fairness fix): previously this split was
    # always seed=42 regardless of which "seed run" was in progress, so all
    # 5 runs (baselines AND SPIRE/GCN/GAT) silently shared one split.
    split = create_spatial_split_indices(processed_df, test_region=region, random_seed=seed)
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]

    n_test = len(test_idx)

    _, X_full, y_full = prepare_xgboost_inputs(processed_df, fit_idx=train_idx)
    actual_proportion = float(y_full.iloc[test_idx].mean())
    print(f"  n_test={n_test}  actual_proportion={actual_proportion:.3f}")

    _, _, X_selected = train_xgboost_and_select_features(
        X_full, y_full, fit_idx=train_idx
    )
    # Leakage-free xgb_score (Comment 17 fairness fix, same OOF machinery as
    # Comment 12-A): train rows get a K-fold out-of-fold score, val/test
    # rows get the score from the model refit on the full train_idx.
    other_idx = np.concatenate([val_idx, test_idx])
    train_scores, other_scores, _ = compute_ensemble_scores_no_leakage(
        X_full, y_full, train_idx, other_idx, random_state=seed,
        models_factory=_make_single_xgb_model,
    )
    xgb_score = np.empty(len(X_full), dtype=float)
    xgb_score[train_idx] = train_scores
    xgb_score[other_idx] = other_scores
    X_bl = X_selected.copy()
    X_bl["xgb_score"] = xgb_score

    X_train = X_bl.iloc[train_idx].astype(float)
    y_train = y_full.iloc[train_idx]
    X_test  = X_bl.iloc[test_idx].astype(float)

    # ── Tabular baselines ────────────────────────────────────────────────────
    per_model: dict = {}
    for name, model in _make_baseline_models(seed=seed).items():
        try:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            pred_prop = float(np.mean(y_prob))
            abs_err   = float(abs(pred_prop - actual_proportion))
            per_model[name] = {
                "predicted_proportion": pred_prop,
                "abs_error":            abs_err,
                "y_prob":               y_prob.tolist(),  # Comment 18-B: Brier/calibration
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
    y_train_arr  = y_full.iloc[train_idx].to_numpy()

    for name, fn, kwargs in [
        ("autoencoder",  predict_proportion_autoencoder,
         {"X_train": X_train.values, "X_test": X_test.values, "y_train": y_train_arr}),
        ("spatial_knn",  predict_proportion_spatial_knn,
         {"coords_train": coords_train, "coords_test": coords_test, "y_train": y_train_arr}),
        ("kriging",      predict_proportion_kriging,
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

    tv_pos    = np.sort(np.concatenate([train_idx, val_idx]))
    old_to_new = {int(old): new for new, old in enumerate(tv_pos)}
    new_train  = np.array([old_to_new[i] for i in train_idx], dtype=np.int64)
    new_val    = np.array([old_to_new[i] for i in val_idx],   dtype=np.int64)

    coords_tv   = processed_df[["Latitude", "Longitude"]].values[tv_pos]
    # GNN ensemble node score, leakage-free (Comment 12): train nodes get
    # K-fold OOF scores, val nodes get scores from the ensemble refit on the
    # full train_idx (also reused for test-time inference below).
    X_node_base, ensemble_models = build_node_features_no_leakage(
        X_selected.iloc[tv_pos], X_full, y_full, train_idx, val_idx, new_train, new_val,
        random_state=seed,
    )
    # Geo-only vector for cosine similarity (Comment 13): excludes
    # Latitude/Longitude and ensemble_score to avoid double-counting them
    # into the "geological similarity" edge weight.
    X_geo_tv = torch.tensor(
        select_geo_similarity_columns(X_selected.iloc[tv_pos]).astype(float).values,
        dtype=torch.float32,
    )
    edge_index, edge_weight, sigma = build_weighted_graph(
        coords_tv, X_node_base,
        alpha=_GRAPH_ALPHA, k=_GRAPH_K,
        max_distance_km=_GRAPH_MAX_DIST_KM,
        fit_idx=new_train,
        return_sigma=True,
        X_geo=X_geo_tv,
        unify_edge_rule=True,  # Comment 14 Solution A (decided)
    )

    X_node_tv_base = X_node_base.clone().cpu()
    X_geo_tv_base = X_geo_tv.clone().cpu()
    X_node_base = augment_with_edge_stats(X_node_base, edge_index, edge_weight)

    N_tv = len(tv_pos)
    data = build_pyg_data(X_node_base, edge_index, edge_weight, y_full.iloc[tv_pos])
    train_mask = torch.zeros(N_tv, dtype=torch.bool)
    val_mask   = torch.zeros(N_tv, dtype=torch.bool)
    train_mask[torch.tensor(new_train)] = True
    val_mask[torch.tensor(new_val)]     = True
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = torch.zeros(N_tv, dtype=torch.bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = data.to(device)
    data.x = data.x.detach()

    train_idx_t = torch.tensor(new_train, dtype=torch.long, device=device)
    val_idx_t   = torch.tensor(new_val,   dtype=torch.long, device=device)
    tv_data, sub_train, sub_val = build_train_val_subgraph(data, train_idx_t, val_idx_t)

    print(f"  Tuning SPIRE (config shared with GCN/GAT)...")
    best_config, best_val_f1 = tune_gnn_model(GraphSAGE, tv_data, sub_train, sub_val, device)
    hidden_dim, num_layers, dropout, lr = best_config
    print(f"  Best config={best_config}  val_f1={best_val_f1:.4f}")

    gnn_artifacts_base = {
        "device":          device,
        "coords_tv":       coords_tv,
        "X_node_tv":       tv_data.x,
        "X_node_tv_base":  X_node_tv_base,
        "X_geo_tv_base":   X_geo_tv_base,
        "edge_index_tv":   tv_data.edge_index,
        "edge_weight_tv":  tv_data.edge_weight,
        "graph_sigma":     sigma,
        "ensemble_models": ensemble_models,
        "test_idx":        test_idx,
        "X_test":          X_full.iloc[test_idx].reset_index(drop=True),
        "X_selected_test": X_selected.iloc[test_idx].reset_index(drop=True),
        "y_test":          y_full.iloc[test_idx].reset_index(drop=True),
    }

    for model_class, model_key in [(GraphSAGE, "spire"), (GCN, "gcn"), (GAT, "gat")]:
        model, _, _ = train_gnn_model(
            model_class=model_class,
            data=tv_data,
            train_idx=sub_train,
            val_idx=sub_val,
            test_idx=None,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            epochs=200,
            patience=20,
            device=device,
            criterion=torch.nn.CrossEntropyLoss(),
        )

        gnn_artifacts = {**gnn_artifacts_base, "model": model}
        print(f"  Predicting {n_test} test nodes inductively ({model_key.upper()})...")
        y_prob_gnn, _ = _predict_test_nodes(gnn_artifacts, processed_df)
        pred_prop_gnn = float(np.mean(y_prob_gnn))
        abs_err_gnn   = float(abs(pred_prop_gnn - actual_proportion))
        per_model[model_key] = {
            "predicted_proportion": pred_prop_gnn,
            "abs_error":            abs_err_gnn,
            "best_config":          list(best_config),
            "best_val_f1":          float(best_val_f1),
            "y_prob":               np.asarray(y_prob_gnn).tolist(),  # Comment 18-B
        }
        print(f"    [{model_key}]  pred={pred_prop_gnn:.3f}  |err|={abs_err_gnn:.3f}")

    return {
        "n_test":            int(n_test),
        "actual_proportion": actual_proportion,
        "y_true":            y_full.iloc[test_idx].to_numpy(dtype=int).tolist(),  # Comment 18-B
        "per_model":         per_model,
    }


def _run_gcn_gat(
    processed_df: pd.DataFrame,
    region: str,
    best_config: list,
    actual_proportion: float,
) -> dict:
    """Train GCN and GAT with a given best_config (no tuning). Returns {gcn, gat} results."""
    split = create_spatial_split_indices(processed_df, test_region=region)
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]
    n_test    = len(test_idx)

    _, X_full, y_full = prepare_xgboost_inputs(processed_df, fit_idx=train_idx)
    _, _, X_selected = train_xgboost_and_select_features(
        X_full, y_full, fit_idx=train_idx
    )

    tv_pos     = np.sort(np.concatenate([train_idx, val_idx]))
    old_to_new = {int(old): new for new, old in enumerate(tv_pos)}
    new_train  = np.array([old_to_new[i] for i in train_idx], dtype=np.int64)
    new_val    = np.array([old_to_new[i] for i in val_idx],   dtype=np.int64)

    coords_tv   = processed_df[["Latitude", "Longitude"]].values[tv_pos]
    # Leakage-free (Comment 12): train nodes get K-fold OOF scores, val nodes
    # get scores from the ensemble refit on the full train_idx.
    X_node_base, ensemble_models = build_node_features_no_leakage(
        X_selected.iloc[tv_pos], X_full, y_full, train_idx, val_idx, new_train, new_val,
    )
    # Geo-only vector for cosine similarity (Comment 13).
    X_geo_tv = torch.tensor(
        select_geo_similarity_columns(X_selected.iloc[tv_pos]).astype(float).values,
        dtype=torch.float32,
    )
    edge_index, edge_weight, sigma = build_weighted_graph(
        coords_tv, X_node_base,
        alpha=_GRAPH_ALPHA, k=_GRAPH_K,
        max_distance_km=_GRAPH_MAX_DIST_KM,
        fit_idx=new_train,
        return_sigma=True,
        X_geo=X_geo_tv,
        unify_edge_rule=True,  # Comment 14 Solution A (decided)
    )
    X_node_tv_base = X_node_base.clone().cpu()
    X_geo_tv_base = X_geo_tv.clone().cpu()
    X_node_base    = augment_with_edge_stats(X_node_base, edge_index, edge_weight)

    N_tv = len(tv_pos)
    data = build_pyg_data(X_node_base, edge_index, edge_weight, y_full.iloc[tv_pos])
    train_mask = torch.zeros(N_tv, dtype=torch.bool)
    val_mask   = torch.zeros(N_tv, dtype=torch.bool)
    train_mask[torch.tensor(new_train)] = True
    val_mask[torch.tensor(new_val)]     = True
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = torch.zeros(N_tv, dtype=torch.bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = data.to(device)
    data.x = data.x.detach()

    train_idx_t = torch.tensor(new_train, dtype=torch.long, device=device)
    val_idx_t   = torch.tensor(new_val,   dtype=torch.long, device=device)
    tv_data, sub_train, sub_val = build_train_val_subgraph(data, train_idx_t, val_idx_t)

    hidden_dim, num_layers, dropout, lr = best_config
    gnn_artifacts_base = {
        "device":          device,
        "coords_tv":       coords_tv,
        "X_node_tv":       tv_data.x,
        "X_node_tv_base":  X_node_tv_base,
        "X_geo_tv_base":   X_geo_tv_base,
        "edge_index_tv":   tv_data.edge_index,
        "edge_weight_tv":  tv_data.edge_weight,
        "graph_sigma":     sigma,
        "ensemble_models": ensemble_models,
        "test_idx":        test_idx,
        "X_test":          X_full.iloc[test_idx].reset_index(drop=True),
        "X_selected_test": X_selected.iloc[test_idx].reset_index(drop=True),
        "y_test":          y_full.iloc[test_idx].reset_index(drop=True),
    }

    results = {}
    for model_class, model_key in [(GCN, "gcn"), (GAT, "gat")]:
        model, _, _ = train_gnn_model(
            model_class=model_class,
            data=tv_data,
            train_idx=sub_train,
            val_idx=sub_val,
            test_idx=None,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            epochs=200,
            patience=20,
            device=device,
            criterion=torch.nn.CrossEntropyLoss(),
        )
        gnn_artifacts = {**gnn_artifacts_base, "model": model}
        y_prob_gnn, _ = _predict_test_nodes(gnn_artifacts, processed_df)
        pred_prop_gnn = float(np.mean(y_prob_gnn))
        abs_err_gnn   = float(abs(pred_prop_gnn - actual_proportion))
        results[model_key] = {
            "predicted_proportion": pred_prop_gnn,
            "abs_error":            abs_err_gnn,
            "best_config":          list(best_config),
        }
        print(f"    [{model_key}]  pred={pred_prop_gnn:.3f}  |err|={abs_err_gnn:.3f}")

    return results


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


def run_exp4_proportion(
    processed_df: pd.DataFrame,
    results_path: str = _RESULTS_PATH,
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
        import torch as _torch
        _torch.manual_seed(seed)

        per_region: dict = {}
        for region in regions:
            print(f"\n{'='*60}")
            print(f"  Test region: {region}  (seed={seed})")
            print(f"{'='*60}")
            per_region[region] = _run_region(processed_df, region, seed=seed)

        agg = _aggregate_metrics(per_region)
        print(f"\n  Aggregated metrics (seed={seed}):")
        for m, v in agg.items():
            if v:
                print(f"    {m:20s}  MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}")
        runs[str(seed)] = {"aggregated_metrics": agg, "per_region": per_region}

    # Merge with any existing results file so a partial re-run (e.g. a
    # single seed) doesn't discard other seeds already saved there.
    if Path(results_path).exists():
        with open(results_path, encoding="utf-8") as f:
            existing_runs = json.load(f).get("runs", {})
        runs = {**existing_runs, **runs}
        print(f"Merged with existing results → {len(runs)} seed(s) total: {sorted(runs.keys(), key=int)}")

    # Top-level aggregated_metrics and per_region from first run for convenience
    first = next(iter(runs.values()))

    payload = {
        "experiment": "exp4_proportion",
        "description": (
            "Leave-one-region-out proportion prediction. "
            "For each held-out region, every node is predicted; "
            "predicted_proportion = mean(y_prob). "
            "Metric: MAE and RMSE of |predicted − actual| across 10 regions. "
            "Baselines use 11 features (10 SHAP + xgb_score). "
            "SPIRE is inductive (same mechanism as Exp 1)."
        ),
        "models": [
            "xgboost", "lightgbm", "catboost", "random_forest", "svm", "mlp",
            "autoencoder", "spatial_knn", "kriging",
            "spire", "gcn", "gat",
        ],
        "regions": regions,
        "aggregated_metrics": first["aggregated_metrics"],
        "per_region": first["per_region"],
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
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                         help="Explicit seed list (default: 42 + 4 random)")
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)
    run_exp4_proportion(processed_df, seeds=args.seeds)


if __name__ == "__main__":
    main()
