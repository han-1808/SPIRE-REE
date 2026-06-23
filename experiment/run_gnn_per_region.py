"""
experiment/run_gnn_per_region.py

Per-region transductive GNN evaluation.

For each of the 11 regions:
  - Build a graph using ONLY nodes from that region.
  - Hold out 5 randomly sampled nodes as test (excluded from graph construction).
  - Split remaining nodes into train (80%) and val (20%).
  - Train SPIRE on the region graph (num_layers restricted to [1, 2]).
  - Evaluate on the 5 held-out test nodes.

Results saved to result/gnn_per_region.json.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from config import OUTPUT_PATH
from feature_engineering import (
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

_GNN_MODELS = [("spire", GraphSAGE), ("gcn", GCN), ("gat", GAT)]

_N_TEST = 5
_GRAPH_K = 10
_GRAPH_ALPHA = 0.5
_GRAPH_MAX_DIST_KM = 2000.0
_RESULTS_PATH = "result/gnn_per_region.json"
_EARTH_RADIUS_KM = 6371.0
_PREDICT_K = 10
N_RUNS = 5


def _predict_node_aug(
    artifacts: dict,
    lat: float,
    lon: float,
    x_new_base: torch.Tensor,   # (1, 11) base features, no edge stats
) -> float:
    """
    Inductive prediction when the TV graph uses augmented (13-dim) node features.
    Uses base features for cosine similarity, then derives edge stats for the new
    node before appending it (13-dim) to the augmented training graph.
    """
    device         = artifacts["device"]
    model          = artifacts["model"]
    coords_tv      = artifacts["coords_tv"]
    X_node_tv      = artifacts["X_node_tv"]        # (N, 13) on device
    X_node_tv_base = artifacts["X_node_tv_base"]   # (N, 11) on CPU, tv_local order
    edge_index_tv  = artifacts["edge_index_tv"]
    edge_weight_tv = artifacts["edge_weight_tv"]
    sigma          = artifacts["graph_sigma"]

    nbrs = NearestNeighbors(n_neighbors=_PREDICT_K, metric="haversine")
    nbrs.fit(np.radians(coords_tv))
    dist_rad, indices = nbrs.kneighbors(np.radians([[lat, lon]]))
    dist_km  = dist_rad[0] * _EARTH_RADIUS_KM
    indices  = indices[0]

    within = dist_km <= _GRAPH_MAX_DIST_KM
    neighbour_idx = indices[within] if within.sum() >= 1 else indices[:1]

    mini_coords = np.vstack([[lat, lon], coords_tv[neighbour_idx]])
    mini_X_base = torch.cat(
        [x_new_base.cpu(), X_node_tv_base[neighbour_idx].cpu()], dim=0
    )
    ei_mini, ew_mini = build_weighted_graph(
        mini_coords, mini_X_base,
        alpha=_GRAPH_ALPHA, k=len(neighbour_idx),
        weighting_strategy="mixed", sigma=sigma,
    )

    mask0    = (ei_mini[0] == 0) | (ei_mini[1] == 0)
    ei_local = ei_mini[:, mask0]
    ew_local = ew_mini[mask0]

    incoming = ei_local[1] == 0
    if int(incoming.sum()) > 0:
        ew_in   = ew_local[incoming]
        mean_ew = float(ew_in.mean())
        max_ew  = float(ew_in.max())
    else:
        mean_ew = max_ew = 0.0

    edge_stats = torch.tensor([[mean_ew, max_ew]], dtype=torch.float32)
    x_new_aug  = torch.cat([x_new_base.cpu(), edge_stats], dim=1).to(device)  # (1, 13)

    N = X_node_tv.shape[0]
    l2g = torch.full((1 + len(neighbour_idx),), -1, dtype=torch.long)
    l2g[0]  = N
    l2g[1:] = torch.tensor(neighbour_idx, dtype=torch.long)
    ei_global = l2g[ei_local].to(device)
    ew_global = ew_local.to(device)

    X_aug  = torch.cat([X_node_tv, x_new_aug], dim=0)
    ei_aug = torch.cat([edge_index_tv, ei_global], dim=1)
    ew_aug = torch.cat([edge_weight_tv, ew_global], dim=0)

    y_dummy  = torch.zeros(N + 1, dtype=torch.long, device=device)
    aug_data = Data(x=X_aug, edge_index=ei_aug, edge_weight=ew_aug, y=y_dummy)

    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(aug_data), dim=1)[N, 1].item()
    return prob


def _split_region_indices(
    region_local_idx: np.ndarray,
    n_test: int = _N_TEST,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given local indices of all nodes in a region, randomly sample `n_test` as
    test and split the rest into train / val.

    Returns (train_local, val_local, test_local) — all in local (region) space.
    """
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(region_local_idx))

    test_local = perm[:n_test]
    rest = perm[n_test:]

    n_val = max(1, int(len(rest) * val_ratio))
    val_local = rest[:n_val]
    train_local = rest[n_val:]

    return train_local, val_local, test_local


def _run_region(processed_df: pd.DataFrame, region: str, random_seed: int = 42) -> dict:
    region_mask = processed_df["Region"] == region
    region_global_idx = np.where(region_mask)[0]

    n_nodes = len(region_global_idx)
    print(f"  Region '{region}': {n_nodes} nodes total")

    if n_nodes < _N_TEST + 5:
        print(f"  Skipping — too few nodes ({n_nodes})")
        return {}

    train_local, val_local, test_local = _split_region_indices(region_global_idx, random_seed=random_seed)

    train_global = region_global_idx[train_local]
    val_global   = region_global_idx[val_local]
    test_global  = region_global_idx[test_local]

    # XGBoost + SHAP feature selection fitted on train only (global indices)
    region_df = processed_df.iloc[region_global_idx].reset_index(drop=True)
    _, X_region, y_region = prepare_xgboost_inputs(processed_df, fit_idx=train_global)

    # Restrict X, y to this region's rows
    X_region = X_region.iloc[region_global_idx].reset_index(drop=True)
    y_region = y_region.iloc[region_global_idx].reset_index(drop=True)

    xgb_model, _, X_selected = train_xgboost_and_select_features(
        X_region, y_region, fit_idx=train_local
    )

    # Ensemble: CatBoost + RandomForest for richer node score
    X_tr = X_region.iloc[train_local].astype(float)
    y_tr = y_region.iloc[train_local]

    cat_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        loss_function="Logloss", random_seed=random_seed, verbose=False,
    )
    cat_model.fit(X_tr, y_tr)

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=random_seed, n_jobs=-1,
    )
    rf_model.fit(X_tr, y_tr)

    ensemble_models = [xgb_model, cat_model, rf_model]

    # Build graph from train+val nodes only (test nodes excluded)
    tv_local = np.sort(np.concatenate([train_local, val_local]))
    old_to_new = {int(old): new for new, old in enumerate(tv_local)}
    new_train = np.array([old_to_new[i] for i in train_local], dtype=np.int64)
    new_val   = np.array([old_to_new[i] for i in val_local],   dtype=np.int64)

    coords_region = processed_df[["Latitude", "Longitude"]].values[region_global_idx]
    coords_tv = coords_region[tv_local]

    X_node_tv = build_node_features(
        X_selected.iloc[tv_local],
        X_region.iloc[tv_local],
        ensemble_models,
    )

    edge_index, edge_weight, sigma = build_weighted_graph(
        coords_tv, X_node_tv,
        alpha=_GRAPH_ALPHA, k=min(_GRAPH_K, len(tv_local) - 1),
        max_distance_km=_GRAPH_MAX_DIST_KM,
        fit_idx=new_train,
        return_sigma=True,
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
    data = data.to(device)
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
    model_metrics: dict = {}

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

        artifacts = {**base_artifacts, "model": model}
        y_probs = []
        for local_i in test_local:
            x_new_base = build_node_features(
                X_selected.iloc[[local_i]],
                X_region.iloc[[local_i]],
                ensemble_models,
            )
            lat, lon = coords_region[local_i]
            y_probs.append(_predict_node_aug(artifacts, lat, lon, x_new_base))

        y_true = y_region.iloc[test_local].to_numpy()
        y_prob = np.array(y_probs)
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            metrics = {
                "accuracy":  float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
                "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc":   float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else (1.0 if np.all(y_pred == y_true) else 0.0),
                "pr_auc":    float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else (1.0 if np.all(y_pred == y_true) else 0.0),
            }
        except Exception as exc:
            metrics = {"error": str(exc)}

        print(f"  [{name}] Test metrics: {metrics}")
        model_metrics[name] = metrics

    return {
        "n_total":     int(n_nodes),
        "n_train":     int(len(train_local)),
        "n_val":       int(len(val_local)),
        "n_test":      int(len(test_local)),
        "best_config": list(best_config),
        "best_val_f1": float(best_val_f1),
        "test_metrics": model_metrics,
    }


def _avg_model_metrics(per_region: dict, model_name: str) -> dict:
    valid = {}
    for r, d in per_region.items():
        if not d:
            continue
        m = d.get("test_metrics", {}).get(model_name, {})
        if m and "error" not in m:
            valid[r] = m
    if not valid:
        return {}
    scalar_keys = [k for k, v in next(iter(valid.values())).items() if v is not None]
    return {k: float(np.mean([v[k] for v in valid.values() if v.get(k) is not None])) for k in scalar_keys}


def run_gnn_per_region(processed_df: pd.DataFrame, results_path: str = _RESULTS_PATH) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    print(f"Regions ({len(regions)}): {regions}")
    model_names = [name for name, _ in _GNN_MODELS]

    seeds = [42] + random.sample([s for s in range(61) if s != 42], N_RUNS - 1)
    print(f"Seeds for this run: {seeds}")

    runs: list = []
    for run_id, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"  RUN {run_id}  seed={seed}")
        print(f"{'#'*60}")
        import torch as _torch
        _torch.manual_seed(seed)

        per_region: dict = {}
        for region in regions:
            print(f"\n{'='*60}")
            print(f"  Region: {region}  (seed={seed})")
            print(f"{'='*60}")
            per_region[region] = _run_region(processed_df, region, random_seed=seed)

        avg = {name: _avg_model_metrics(per_region, name) for name in model_names}
        print(f"\n  Avg metrics (seed={seed}): {avg}")
        runs.append({"run_id": run_id, "seed": seed, "avg_metrics": avg, "per_region": per_region})

    payload = {
        "experiment": "gnn_per_region_inductive",
        "description": (
            f"SPIRE / GCN / GAT trained per-region. Each region forms its own graph. "
            f"{_N_TEST} nodes randomly held out as test (strictly inductive: "
            f"each test node augmented one-by-one into the train+val graph). "
            f"SPIRE tuned; GCN and GAT reuse the same best config."
        ),
        "models": model_names,
        "n_test_per_region": _N_TEST,
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
    run_gnn_per_region(processed_df)


if __name__ == "__main__":
    main()
