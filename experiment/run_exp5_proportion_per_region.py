"""
experiment/run_exp5_proportion_per_region.py

Exp 5 — Per-region proportion prediction, random split.

Same data split and architecture as Exp 2 (run_gnn_per_region.py /
run_baseline_per_region.py):
  - Each region forms its own isolated graph.
  - 5 nodes randomly held out as test (identical split to Exp 2).
  - GNNs (SPIRE / GCN / GAT) + 6 tabular baselines trained per-region.
  - Predicted proportion = mean(y_prob) over the 5 test nodes.
  - Actual proportion   = fraction of has_ree=1 among the 5 test nodes.
  - Metric: MAE and RMSE of |predicted − actual| across 10 regions.

Results saved to result/exp5_proportion_per_region.json.
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
    build_baseline_features,
    build_node_features,
    build_weighted_graph,
    prepare_xgboost_inputs,
    train_xgboost_and_select_features,
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
from run_gnn_per_region import _predict_node_aug

_GNN_MODELS = [("spire", GraphSAGE), ("gcn", GCN), ("gat", GAT)]
_N_TEST = 5
_GRAPH_K = 10
_GRAPH_ALPHA = 0.5
_GRAPH_MAX_DIST_KM = 2000.0
_RESULTS_PATH = "result/exp5_proportion_per_region.json"
N_RUNS = 5


def _split_region_indices(
    region_local_idx: np.ndarray,
    n_test: int = _N_TEST,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(region_local_idx))
    test_local = perm[:n_test]
    rest = perm[n_test:]
    n_val = max(1, int(len(rest) * val_ratio))
    val_local = rest[:n_val]
    train_local = rest[n_val:]
    return train_local, val_local, test_local


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


def _run_region(processed_df: pd.DataFrame, region: str, random_seed: int = 42) -> dict:
    region_global_idx = np.where(processed_df["Region"] == region)[0]
    n_nodes = len(region_global_idx)
    print(f"  Region '{region}': {n_nodes} nodes total")

    if n_nodes < _N_TEST + 5:
        print(f"  Skipping — too few nodes ({n_nodes})")
        return {}

    train_local, val_local, test_local = _split_region_indices(region_global_idx, random_seed=random_seed)
    train_global = region_global_idx[train_local]

    _, X_full, y_full = prepare_xgboost_inputs(processed_df, fit_idx=train_global)
    X_region = X_full.iloc[region_global_idx].reset_index(drop=True)
    y_region = y_full.iloc[region_global_idx].reset_index(drop=True)

    xgb_model, _, X_selected = train_xgboost_and_select_features(
        X_region, y_region, fit_idx=train_local
    )
    X_bl = build_baseline_features(X_selected, X_region, xgb_model)

    X_train = X_bl.iloc[train_local].astype(float)
    y_train = y_region.iloc[train_local]
    X_test  = X_bl.iloc[test_local].astype(float)

    actual_proportion = float(y_region.iloc[test_local].mean())
    n_test = len(test_local)
    print(f"  n_test={n_test}  actual_proportion={actual_proportion:.3f}")

    # ── Tabular baselines ─────────────────────────────────────────────────────
    per_model: dict = {}
    for name, model in _make_baseline_models(seed=random_seed).items():
        try:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            pred_prop = float(np.mean(y_prob))
            abs_err = float(abs(pred_prop - actual_proportion))
            per_model[name] = {"predicted_proportion": pred_prop, "abs_error": abs_err}
        except Exception as exc:
            per_model[name] = {"error": str(exc)}
        info = per_model[name]
        if "abs_error" in info:
            print(f"    [{name}]  pred={info['predicted_proportion']:.3f}  |err|={info['abs_error']:.3f}")
        else:
            print(f"    [{name}]  ERROR: {info.get('error')}")

    # ── Spatial / autoencoder baselines ─────────────────────────────────────
    coords_region = processed_df[["Latitude", "Longitude"]].values[region_global_idx]
    coords_train = coords_region[train_local]
    coords_test  = coords_region[test_local]
    y_train_arr  = y_region.iloc[train_local].to_numpy()

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
            per_model[name] = {"predicted_proportion": pred_prop, "abs_error": abs_err}
        except Exception as exc:
            per_model[name] = {"error": str(exc)}
        info = per_model[name]
        if "abs_error" in info:
            print(f"    [{name}]  pred={info['predicted_proportion']:.3f}  |err|={info['abs_error']:.3f}")
        else:
            print(f"    [{name}]  ERROR: {info.get('error')}")

    # ── GNN ensemble ─────────────────────────────────────────────────────────
    X_tr = X_region.iloc[train_local].astype(float)
    y_tr = y_region.iloc[train_local]

    cat_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        loss_function="Logloss", random_seed=random_seed, verbose=False,
    )
    cat_model.fit(X_tr, y_tr)
    rf_model = RandomForestClassifier(n_estimators=300, random_state=random_seed, n_jobs=-1)
    rf_model.fit(X_tr, y_tr)
    ensemble_models = [xgb_model, cat_model, rf_model]

    tv_local = np.sort(np.concatenate([train_local, val_local]))
    old_to_new = {int(old): new for new, old in enumerate(tv_local)}
    new_train = np.array([old_to_new[i] for i in train_local], dtype=np.int64)
    new_val   = np.array([old_to_new[i] for i in val_local],   dtype=np.int64)

    coords_tv = coords_region[tv_local]

    X_node_tv = build_node_features(
        X_selected.iloc[tv_local], X_region.iloc[tv_local], ensemble_models,
    )
    edge_index, edge_weight, sigma = build_weighted_graph(
        coords_tv, X_node_tv,
        alpha=_GRAPH_ALPHA, k=min(_GRAPH_K, len(tv_local) - 1),
        max_distance_km=_GRAPH_MAX_DIST_KM,
        fit_idx=new_train, return_sigma=True,
    )

    X_node_tv_base = X_node_tv.clone().cpu()
    X_node_tv = augment_with_edge_stats(X_node_tv, edge_index, edge_weight)

    N_tv = len(tv_local)
    data = build_pyg_data(X_node_tv, edge_index, edge_weight, y_region.iloc[tv_local])
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

    base_artifacts = {
        "device":          device,
        "coords_tv":       coords_tv,
        "X_node_tv":       tv_data.x,
        "X_node_tv_base":  X_node_tv_base,
        "edge_index_tv":   tv_data.edge_index,
        "edge_weight_tv":  tv_data.edge_weight,
        "graph_sigma":     sigma,
    }

    best_config = None
    best_val_f1 = None
    for name, model_class in _GNN_MODELS:
        if best_config is None:
            print(f"  Tuning {name.upper()} for region: {region}")
            best_config, best_val_f1 = tune_gnn_model(
                model_class, tv_data, sub_train, sub_val, device,
            )
            print(f"  Best config={best_config}  val_f1={best_val_f1:.4f}")
        else:
            print(f"  Training {name.upper()} with shared config={best_config}")

        hidden_dim, num_layers, dropout, lr = best_config
        model, _, _ = train_gnn_model(
            model_class=model_class, data=tv_data,
            train_idx=sub_train, val_idx=sub_val, test_idx=None,
            hidden_channels=hidden_dim, num_layers=num_layers,
            dropout=dropout, lr=lr, epochs=200, patience=20,
            device=device, criterion=torch.nn.CrossEntropyLoss(),
        )

        artifacts = {**base_artifacts, "model": model}
        y_probs = []
        for local_i in test_local:
            x_new_base = build_node_features(
                X_selected.iloc[[local_i]], X_region.iloc[[local_i]], ensemble_models,
            )
            lat, lon = coords_region[local_i]
            y_probs.append(_predict_node_aug(artifacts, lat, lon, x_new_base))

        pred_prop = float(np.mean(y_probs))
        abs_err   = float(abs(pred_prop - actual_proportion))
        per_model[name] = {"predicted_proportion": pred_prop, "abs_error": abs_err}
        print(f"    [{name}]  pred={pred_prop:.3f}  |err|={abs_err:.3f}")

    return {
        "n_total":           int(n_nodes),
        "n_train":           int(len(train_local)),
        "n_val":             int(len(val_local)),
        "n_test":            int(n_test),
        "actual_proportion": actual_proportion,
        "best_config":       list(best_config) if best_config is not None else None,
        "best_val_f1":       float(best_val_f1) if best_val_f1 is not None else None,
        "per_model":         per_model,
    }


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


def run_exp5(
    processed_df: pd.DataFrame,
    results_path: str = _RESULTS_PATH,
) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    print(f"Regions ({len(regions)}): {regions}")

    seeds = [42] + random.sample([s for s in range(61) if s != 42], N_RUNS - 1)
    print(f"Seeds for this run: {seeds}")

    runs: list = []
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
            per_region[region] = _run_region(processed_df, region, random_seed=seed)

        agg = _aggregate_metrics(per_region)
        print(f"\n  Aggregated metrics (seed={seed}):")
        for m, v in agg.items():
            if v:
                print(f"    {m:20s}  MAE={v['mae']:.4f}  RMSE={v['rmse']:.4f}")
        runs.append({"seed": seed, "aggregated_metrics": agg, "per_region": per_region})

    payload = {
        "experiment": "exp5_proportion_per_region",
        "description": (
            "Per-region proportion prediction, random split (same as Exp 2). "
            f"Each region uses its own isolated graph; {_N_TEST} nodes randomly "
            "held out per run. predicted_proportion = mean(y_prob) over test nodes. "
            "Metric: MAE and RMSE of |predicted − actual| across 10 regions."
        ),
        "n_test_per_region": _N_TEST,
        "sampling":  "random",
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
    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)
    run_exp5(processed_df)


if __name__ == "__main__":
    main()
