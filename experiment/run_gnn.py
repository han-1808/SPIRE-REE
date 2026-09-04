"""
experiment/run_gnn.py

Inductive GNN evaluation: leave-one-region-out sweep.

Same split as run_baseline.py (create_spatial_split_indices), but inductive:
  - SPIRE is trained on the train+val graph with test region nodes absent.
  - Each test node is added to the graph one-by-one for prediction (true inductive).

Leakage audit
-------------
  - prepare_xgboost_inputs / train_xgboost_and_select_features fitted on train_idx only.
  - Graph (coords, X_node, edges) built from train+val nodes only.
  - tune_gnn_model uses train+val subgraph only.
  - Test node coordinates never appear in the training graph.

Results are saved to result/gnn_inductive.json.
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
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from config import GNN_INDUCTIVE_METRICS_PATH, OUTPUT_PATH
from feature_engineering import (
    build_node_features_no_leakage,
    build_weighted_graph,
    create_spatial_split_indices,
    prepare_xgboost_inputs,
    select_geo_similarity_columns,
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

_EARTH_RADIUS_KM = 6371.0
_GRAPH_K = 15
_GRAPH_ALPHA = 0.5
_GRAPH_MAX_DIST_KM = 2000.0
_PREDICT_K = 15
N_RUNS = 5


# ── Training ──────────────────────────────────────────────────────────────────

def _build_graph_artifacts(processed_df: pd.DataFrame, region: str, random_seed: int = 42) -> dict:
    """
    Build train+val graph for a region (test region fully excluded).
    Does NOT train any GNN. Returns shared artifacts for all model classes.
    """
    split = create_spatial_split_indices(processed_df, test_region=region, val_ratio=0.2, random_seed=random_seed)
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]

    graph_df, X, y = prepare_xgboost_inputs(processed_df, fit_idx=train_idx)
    _, _, X_selected = train_xgboost_and_select_features(X, y, fit_idx=train_idx)

    tv_pos = np.sort(np.concatenate([train_idx, val_idx]))
    old_to_new = {int(old): new for new, old in enumerate(tv_pos)}
    new_train  = np.array([old_to_new[i] for i in train_idx], dtype=np.int64)
    new_val    = np.array([old_to_new[i] for i in val_idx],   dtype=np.int64)

    coords_tv   = processed_df[["Latitude", "Longitude"]].values[tv_pos]
    # Leakage-free (Comment 12): train nodes get K-fold OOF scores, val nodes
    # get scores from the ensemble refit on the full train_idx (also reused
    # for test-time inference below).
    X_node_base, ensemble_models = build_node_features_no_leakage(
        X_selected.iloc[tv_pos], X, y, train_idx, val_idx, new_train, new_val,
        random_state=random_seed,
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
    data = build_pyg_data(X_node_base, edge_index, edge_weight, y.iloc[tv_pos])
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

    return {
        # For GNN training
        "tv_data":   tv_data,
        "sub_train": sub_train,
        "sub_val":   sub_val,
        "device":    device,
        # For inductive prediction
        "coords_tv":      coords_tv,
        "X_node_tv":      tv_data.x,
        "X_node_tv_base": X_node_tv_base,
        "X_geo_tv_base":  X_geo_tv_base,
        "edge_index_tv":  tv_data.edge_index,
        "edge_weight_tv": tv_data.edge_weight,
        "graph_sigma":    sigma,
        # For test node feature construction
        "ensemble_models": ensemble_models,
        "X_selected_cols": X_selected.columns.tolist(),
        "X_full_cols":     X.columns.tolist(),
        "test_idx":        test_idx,
        "X_test":          X.iloc[test_idx].reset_index(drop=True),
        "X_selected_test": X_selected.iloc[test_idx].reset_index(drop=True),
        "y_test":          y.iloc[test_idx].reset_index(drop=True),
    }


def _train_model_on_artifacts(
    model_class,
    graph_artifacts: dict,
    best_config: tuple | None = None,
) -> tuple[dict, tuple, float]:
    """
    Tune (if best_config is None) and train model_class on pre-built graph.
    Returns (prediction_artifacts, best_config, best_val_f1).
    """
    tv_data   = graph_artifacts["tv_data"]
    sub_train = graph_artifacts["sub_train"]
    sub_val   = graph_artifacts["sub_val"]
    device    = graph_artifacts["device"]

    if best_config is None:
        best_config, best_val_f1 = tune_gnn_model(model_class, tv_data, sub_train, sub_val, device)
    else:
        best_val_f1 = None

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

    artifacts = {k: v for k, v in graph_artifacts.items()
                 if k not in {"tv_data", "sub_train", "sub_val"}}
    artifacts["model"] = model
    return artifacts, best_config, best_val_f1


# ── Inductive prediction ──────────────────────────────────────────────────────

def _predict_node(
    artifacts: dict,
    lat: float,
    lon: float,
    x_new_base: torch.Tensor,
    x_new_geo: torch.Tensor,
) -> float:
    """
    Augment one new node to the train+val graph and return its REE probability.
    x_new_base : (1, 11) base feature tensor on CPU (no edge stats).
    Edge stats are derived from the mini-graph before appending to the 13-dim graph.
    """
    device         = artifacts["device"]
    model          = artifacts["model"]
    coords_tv      = artifacts["coords_tv"]
    X_node_tv      = artifacts["X_node_tv"]        # (N, 13) augmented, on device
    X_node_tv_base = artifacts["X_node_tv_base"]   # (N, 11) base, on CPU
    X_geo_tv_base  = artifacts["X_geo_tv_base"]
    edge_index_tv  = artifacts["edge_index_tv"]
    edge_weight_tv = artifacts["edge_weight_tv"]
    sigma          = artifacts["graph_sigma"]

    # k-NN neighbours in train+val graph
    nbrs = NearestNeighbors(n_neighbors=_PREDICT_K, metric="haversine")
    nbrs.fit(np.radians(coords_tv))
    dist_rad, indices = nbrs.kneighbors(np.radians([[lat, lon]]))
    dist_km  = dist_rad[0] * _EARTH_RADIUS_KM
    indices  = indices[0]

    within = dist_km <= _GRAPH_MAX_DIST_KM
    neighbour_idx = indices[within] if within.sum() >= 1 else indices[:1]

    mini_coords  = np.vstack([[lat, lon], coords_tv[neighbour_idx]])
    mini_X_base  = torch.cat([x_new_base.cpu(), X_node_tv_base[neighbour_idx].cpu()], dim=0)
    mini_X_geo   = torch.cat([x_new_geo.cpu(), X_geo_tv_base[neighbour_idx].cpu()], dim=0)

    ei_mini, ew_mini = build_weighted_graph(
        mini_coords, mini_X_base,
        alpha=_GRAPH_ALPHA, k=len(neighbour_idx),
        weighting_strategy="mixed", sigma=sigma,
        X_geo=mini_X_geo,
    )

    mask0    = (ei_mini[0] == 0) | (ei_mini[1] == 0)
    ei_local = ei_mini[:, mask0]
    ew_local = ew_mini[mask0]

    # Derive edge stats for the new node
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


def _predict_test_nodes(artifacts: dict, processed_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict all test nodes for a region. Returns (y_prob, y_pred)."""
    test_idx        = artifacts["test_idx"]
    X_test          = artifacts["X_test"]
    X_selected_test = artifacts["X_selected_test"]
    ensemble_models = artifacts["ensemble_models"]

    test_coords = processed_df[["Latitude", "Longitude"]].values[test_idx]
    y_probs = []

    for i in range(len(test_idx)):
        ens_score = float(np.mean([
            m.predict_proba(X_test.iloc[[i]])[0, 1] for m in ensemble_models
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
        prob = _predict_node(artifacts, lat, lon, x_new_base, x_new_geo)
        y_probs.append(prob)

    y_prob = np.array(y_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_prob, y_pred


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    try:
        return {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_true, y_prob)),
            "pr_auc":    float(average_precision_score(y_true, y_prob)),
        }
    except Exception:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        }


def _avg_metrics(metrics_per_region: dict) -> dict:
    valid = {r: m for r, m in metrics_per_region.items() if m}
    if not valid:
        return {}
    keys = list(next(iter(valid.values())).keys())
    return {k: sum(v[k] for v in valid.values()) / len(valid) for k in keys}


# ── Main sweep ─────────────────────────────────────────────────────────────────

_GNN_MODELS = [("spire", GraphSAGE), ("gcn", GCN), ("gat", GAT)]


def _run_one_seed(processed_df: pd.DataFrame, regions: list, random_seed: int) -> dict:
    """Full leave-one-region-out sweep for a single seed. Returns {avg_metrics, per_region}."""
    import torch as _torch
    _torch.manual_seed(random_seed)
    per_region: dict = {}

    for region in regions:
        print(f"\n{'='*60}")
        print(f"  Test region: {region}  (seed={random_seed})")
        print(f"{'='*60}")

        graph_artifacts = _build_graph_artifacts(processed_df, region, random_seed=random_seed)
        y_true = graph_artifacts["y_test"].to_numpy()
        n_test = int(len(graph_artifacts["test_idx"]))

        best_config = None
        best_val_f1 = None
        model_results: dict = {}
        model_predictions: dict = {}  # Comment 7-A

        for name, model_class in _GNN_MODELS:
            print(f"  Training {name.upper()}...")
            artifacts, found_config, found_val_f1 = _train_model_on_artifacts(
                model_class, graph_artifacts, best_config
            )
            if best_config is None:
                best_config = found_config
                best_val_f1 = found_val_f1

            print(f"  Predicting {n_test} test nodes inductively ({name})...")
            y_prob, y_pred = _predict_test_nodes(artifacts, processed_df)
            metrics = _compute_metrics(y_true, y_pred, y_prob)
            print(f"  [{name}] Metrics: {metrics}")
            model_results[name] = metrics
            # Comment 7-A: per-node predictions, needed for the region-specific
            # error decomposition (TP/FP/FN/TN by deposit-type/documentation
            # completeness/kNN distance). Kept in a separate `predictions`
            # dict (not inside `test_metrics[name]`) since `_avg_metrics`
            # below sums every key of `test_metrics[name]` across regions --
            # mixing in list-valued keys there breaks that aggregation.
            model_predictions[name] = {"y_prob": y_prob.tolist(), "y_pred": y_pred.tolist()}

        per_region[region] = {
            "n_test": n_test,
            "best_config": list(best_config),
            "best_val_f1": best_val_f1,
            "test_idx": graph_artifacts["test_idx"].tolist(),  # Comment 7-A
            "y_true": y_true.tolist(),  # Comment 7-A
            "test_metrics": model_results,
            "predictions": model_predictions,  # Comment 7-A
        }

    avg: dict = {}
    for name, _ in _GNN_MODELS:
        avg[name] = _avg_metrics({r: d["test_metrics"][name] for r, d in per_region.items()})
    return {"avg_metrics": avg, "per_region": per_region}


def run_gnn_sweep(
    processed_df: pd.DataFrame,
    metrics_path: str = GNN_INDUCTIVE_METRICS_PATH,
    seeds: list[int] | None = None,
) -> dict:
    """
    Leave-one-region-out inductive sweep with SPIRE, GCN, and GAT.
    Runs N_RUNS times with different random seeds and saves all runs.
    SPIRE is tuned; GCN and GAT reuse the same best config per run.
    """
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
        runs[str(seed)] = _run_one_seed(processed_df, regions, seed)

    # Merge with any existing results file so a partial re-run (e.g. a
    # single seed) doesn't discard other seeds already saved there.
    if Path(metrics_path).exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing_runs = json.load(f).get("runs", {})
        runs = {**existing_runs, **runs}
        print(f"Merged with existing results → {len(runs)} seed(s) total: {sorted(runs.keys(), key=int)}")

    # Cross-seed average metrics
    avg: dict = {}
    for name, _ in _GNN_MODELS:
        metric_keys = list(next(iter(runs.values()))["avg_metrics"][name].keys())
        avg[name] = {
            k: float(np.mean([v["avg_metrics"][name][k] for v in runs.values()
                               if (v["avg_metrics"].get(name) or {}).get(k) is not None]))
            for k in metric_keys
        }
    print(f"\n{'='*60}")
    print(f"  Cross-seed avg metrics: {avg}")

    # Top-level per_region from first run for convenience
    per_region = next(iter(runs.values()))["per_region"]

    payload = {
        "experiment": "gnn_inductive_leave_one_region_out",
        "description": (
            "SPIRE / GCN / GAT trained with test region fully excluded from graph. "
            "SPIRE tuned; GCN and GAT reuse the same best config. "
            "Each test node predicted by augmenting it into the train+val graph."
        ),
        "models": [name for name, _ in _GNN_MODELS],
        "regions": regions,
        "avg_metrics": avg,
        "per_region": per_region,
        "runs": runs,
    }

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {metrics_path}")
    return payload


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                         help="Explicit seed list (default: 42 + 4 random)")
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    run_gnn_sweep(processed_df, seeds=args.seeds)


if __name__ == "__main__":
    main()