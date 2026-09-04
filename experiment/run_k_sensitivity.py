"""
experiment/run_k_sensitivity.py

Graph construction k sensitivity for SPIRE — Exp 1 (leave-one-region-out inductive).

Runs the full pipeline (13-dim features, edge-stat augmentation) with k ∈ {5, 10, 15, 20}
nearest neighbours in the spatial graph (training phase). Seed=42 throughout.

Note: k=15 is the default used in run_gnn.py, so its avg F1 can be cross-checked
against result/gnn_inductive.json (seed=42 run).

Usage:
    python experiment/run_k_sensitivity.py
    python experiment/run_k_sensitivity.py --ks 5 10 20   # skip k=15 (already run)

Results saved to result/k_sensitivity.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from config import OUTPUT_PATH
from feature_engineering import (
    build_node_features_no_leakage,
    build_weighted_graph,
    create_spatial_split_indices,
    prepare_xgboost_inputs,
    select_geo_similarity_columns,
    train_xgboost_and_select_features,
)
from gnn_training import (
    GraphSAGE,
    augment_with_edge_stats,
    build_pyg_data,
    build_train_val_subgraph,
    train_gnn_model,
    tune_gnn_model,
)

_EARTH_RADIUS_KM = 6371.0
_GRAPH_ALPHA = 0.5
_GRAPH_K = 15  # default k when varying max_dist_km instead (run_max_dist_sensitivity.py)
_GRAPH_MAX_DIST_KM = 2000.0
_PREDICT_K = 10
_DEFAULT_KS = [5, 10, 15, 20]
_RESULTS_PATH = "result/k_sensitivity.json"


# ── Graph + feature construction ──────────────────────────────────────────────

def _build_graph_artifacts(
    processed_df: pd.DataFrame,
    region: str,
    graph_k: int = _GRAPH_K,
    max_dist_km: float = _GRAPH_MAX_DIST_KM,
) -> dict:
    """
    graph_k, max_dist_km : swept independently by run_k_sensitivity.py
        (Comment 7) and run_max_dist_sensitivity.py (Comment 13-C), which
        both reuse this function and `_predict_node`/`_predict_test_nodes`
        below rather than duplicating them.
    """
    split = create_spatial_split_indices(processed_df, test_region=region, val_ratio=0.2)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    graph_df, X, y = prepare_xgboost_inputs(processed_df, fit_idx=train_idx)
    _, _, X_selected = train_xgboost_and_select_features(X, y, fit_idx=train_idx)

    tv_pos = np.sort(np.concatenate([train_idx, val_idx]))
    old_to_new = {int(old): new for new, old in enumerate(tv_pos)}
    new_train = np.array([old_to_new[i] for i in train_idx], dtype=np.int64)
    new_val = np.array([old_to_new[i] for i in val_idx], dtype=np.int64)

    coords_tv = processed_df[["Latitude", "Longitude"]].values[tv_pos]
    # Leakage-free (Comment 12): train nodes get K-fold OOF scores, val nodes
    # get scores from the ensemble refit on the full train_idx.
    X_node_base, ensemble_models = build_node_features_no_leakage(
        X_selected.iloc[tv_pos], X, y, train_idx, val_idx, new_train, new_val,
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
        alpha=_GRAPH_ALPHA, k=graph_k,          # <-- varied by run_k_sensitivity.py
        max_distance_km=max_dist_km,            # <-- varied by run_max_dist_sensitivity.py
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
    val_mask = torch.zeros(N_tv, dtype=torch.bool)
    train_mask[torch.tensor(new_train)] = True
    val_mask[torch.tensor(new_val)] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = torch.zeros(N_tv, dtype=torch.bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    data.x = data.x.detach()

    train_idx_t = torch.tensor(new_train, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(new_val, dtype=torch.long, device=device)
    tv_data, sub_train, sub_val = build_train_val_subgraph(data, train_idx_t, val_idx_t)

    return {
        "tv_data": tv_data,
        "sub_train": sub_train,
        "sub_val": sub_val,
        "device": device,
        "coords_tv": coords_tv,
        "X_node_tv": tv_data.x,
        "X_node_tv_base": X_node_tv_base,
        "X_geo_tv_base": X_geo_tv_base,
        "max_dist_km": max_dist_km,
        "edge_index_tv": tv_data.edge_index,
        "edge_weight_tv": tv_data.edge_weight,
        "graph_sigma": sigma,
        "ensemble_models": ensemble_models,
        "test_idx": test_idx,
        "X_test": X.iloc[test_idx].reset_index(drop=True),
        "X_selected_test": X_selected.iloc[test_idx].reset_index(drop=True),
        "y_test": y.iloc[test_idx].reset_index(drop=True),
    }


# ── Inductive prediction (full pipeline, same as run_gnn.py) ─────────────────

def _predict_node(
    artifacts: dict,
    lat: float,
    lon: float,
    x_new_base: torch.Tensor,
    x_new_geo: torch.Tensor,
) -> float:
    device = artifacts["device"]
    model = artifacts["model"]
    coords_tv = artifacts["coords_tv"]
    X_node_tv = artifacts["X_node_tv"]
    X_node_tv_base = artifacts["X_node_tv_base"]
    X_geo_tv_base = artifacts["X_geo_tv_base"]
    max_dist_km = artifacts["max_dist_km"]
    edge_index_tv = artifacts["edge_index_tv"]
    edge_weight_tv = artifacts["edge_weight_tv"]
    sigma = artifacts["graph_sigma"]

    nbrs = NearestNeighbors(n_neighbors=_PREDICT_K, metric="haversine")
    nbrs.fit(np.radians(coords_tv))
    dist_rad, indices = nbrs.kneighbors(np.radians([[lat, lon]]))
    dist_km = dist_rad[0] * _EARTH_RADIUS_KM
    indices = indices[0]

    within = dist_km <= max_dist_km
    neighbour_idx = indices[within] if within.sum() >= 1 else indices[:1]

    mini_coords = np.vstack([[lat, lon], coords_tv[neighbour_idx]])
    mini_X_base = torch.cat([x_new_base.cpu(), X_node_tv_base[neighbour_idx].cpu()], dim=0)
    mini_X_geo = torch.cat([x_new_geo.cpu(), X_geo_tv_base[neighbour_idx].cpu()], dim=0)

    ei_mini, ew_mini = build_weighted_graph(
        mini_coords, mini_X_base,
        alpha=_GRAPH_ALPHA, k=len(neighbour_idx),
        weighting_strategy="mixed", sigma=sigma,
        X_geo=mini_X_geo,
    )

    mask0 = (ei_mini[0] == 0) | (ei_mini[1] == 0)
    ei_local = ei_mini[:, mask0]
    ew_local = ew_mini[mask0]

    incoming = ei_local[1] == 0
    if int(incoming.sum()) > 0:
        ew_in = ew_local[incoming]
        mean_ew = float(ew_in.mean())
        max_ew = float(ew_in.max())
    else:
        mean_ew = max_ew = 0.0

    edge_stats = torch.tensor([[mean_ew, max_ew]], dtype=torch.float32)
    x_new_aug = torch.cat([x_new_base.cpu(), edge_stats], dim=1).to(device)  # (1, 13)

    N = X_node_tv.shape[0]
    l2g = torch.full((1 + len(neighbour_idx),), -1, dtype=torch.long)
    l2g[0] = N
    l2g[1:] = torch.tensor(neighbour_idx, dtype=torch.long)
    ei_global = l2g[ei_local].to(device)
    ew_global = ew_local.to(device)

    X_aug = torch.cat([X_node_tv, x_new_aug], dim=0)
    ei_aug = torch.cat([edge_index_tv, ei_global], dim=1)
    ew_aug = torch.cat([edge_weight_tv, ew_global], dim=0)

    y_dummy = torch.zeros(N + 1, dtype=torch.long, device=device)
    aug_data = Data(x=X_aug, edge_index=ei_aug, edge_weight=ew_aug, y=y_dummy)

    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(aug_data), dim=1)[N, 1].item()
    return prob


def _predict_test_nodes(
    artifacts: dict, processed_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    test_idx = artifacts["test_idx"]
    X_test = artifacts["X_test"]
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
        x_new_base = torch.cat([x_sel, x_ens], dim=1)  # (1, 11)

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
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
        }
    except Exception:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }


def _avg_metrics(metrics_per_region: dict) -> dict:
    valid = {r: m for r, m in metrics_per_region.items() if m}
    if not valid:
        return {}
    keys = list(next(iter(valid.values())).keys())
    return {k: round(sum(v[k] for v in valid.values()) / len(valid), 4) for k in keys}


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_k_sweep(processed_df: pd.DataFrame, ks: list[int]) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    print(f"Regions ({len(regions)}): {regions}")
    print(f"k values: {ks}")

    results = {}
    for graph_k in ks:
        print(f"\n{'='*60}")
        print(f"  k = {graph_k}")
        print(f"{'='*60}")

        per_region: dict = {}
        best_config_global = None

        for region in regions:
            print(f"\n  Region: {region}")
            graph_artifacts = _build_graph_artifacts(processed_df, region, graph_k)
            y_true = graph_artifacts["y_test"].to_numpy()
            n_test = int(len(graph_artifacts["test_idx"]))

            tv_data = graph_artifacts["tv_data"]
            sub_train = graph_artifacts["sub_train"]
            sub_val = graph_artifacts["sub_val"]
            device = graph_artifacts["device"]

            best_config, best_val_f1 = tune_gnn_model(
                GraphSAGE, tv_data, sub_train, sub_val, device
            )
            hidden_dim, num_layers, dropout, lr = best_config

            model, _, _ = train_gnn_model(
                model_class=GraphSAGE,
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

            y_prob, y_pred = _predict_test_nodes(artifacts, processed_df)
            metrics = _compute_metrics(y_true, y_pred, y_prob)
            print(f"  [k={graph_k}] {region}: {metrics}")

            per_region[region] = {
                "n_test": n_test,
                "best_config": list(best_config),
                "best_val_f1": float(best_val_f1),
                "metrics": metrics,
            }

        avg = _avg_metrics({r: d["metrics"] for r, d in per_region.items()})
        print(f"\n  [k={graph_k}] Avg: {avg}")
        results[str(graph_k)] = {"graph_k": graph_k, "avg_metrics": avg, "per_region": per_region}

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph k sensitivity study (Exp 1 SPIRE)")
    parser.add_argument(
        "--ks", nargs="+", type=int, default=_DEFAULT_KS,
        help=f"k values to sweep (default: {_DEFAULT_KS})",
    )
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    results = run_k_sweep(processed_df, args.ks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing results if partial run
    if out_path.exists() and args.ks != _DEFAULT_KS:
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f).get("results", {})
        existing.update(results)
        results = existing

    payload = {
        "experiment": "k_sensitivity",
        "description": (
            "Graph construction k sensitivity for SPIRE on Exp 1 "
            "(leave-one-region-out inductive). Full 13-dim pipeline. Seed=42."
        ),
        "model": "spire",
        "k_values": sorted(int(k) for k in results.keys()),
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Print k-sensitivity table
    print("\n── k sensitivity summary ─────────────────────────")
    print(f"{'k':>4}  {'Avg F1':>7}  {'Accuracy':>9}")
    print("-" * 25)
    for k in sorted(results.keys(), key=int):
        avg = results[k]["avg_metrics"]
        f1 = avg.get("f1", float("nan"))
        acc = avg.get("accuracy", float("nan"))
        marker = " *" if int(k) == 15 else ""
        print(f"{k:>4}  {f1:>7.4f}  {acc:>9.4f}{marker}")
    print("  * default k in run_gnn.py")


if __name__ == "__main__":
    main()
