"""
experiment/run_hparam_sensitivity.py

Hyperparameter sensitivity analysis for one representative region.

Rebuilds the exact same per-region graph as run_gnn_per_region.py (same
seed-42 split, same XGBoost/CatBoost/RF ensemble features, same weighted
graph + edge-stat augmentation), then runs tune_gnn_model over an expanded
hidden_dim x lr grid and logs the FULL grid-search surface — not just the
best configuration.

The expanded grid is a superset of the original tuning grid
(hidden in {32,64,128} x lr in {0.01,0.005,0.001}), so the original-grid
argmax can be extracted from the same records and compared against the
best_config stored in result/gnn_per_region.json.

Usage:
    python experiment/run_hparam_sensitivity.py
    python experiment/run_hparam_sensitivity.py --region "Oceania" --seed 42

Results saved to result/hparam_sensitivity.json.
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

from config import OUTPUT_PATH
from feature_engineering import (
    build_node_features_no_leakage,
    build_weighted_graph,
    prepare_xgboost_inputs,
    select_geo_similarity_columns,
    train_xgboost_and_select_features,
)
from gnn_training import (
    GraphSAGE,
    augment_with_edge_stats,
    build_pyg_data,
    build_train_val_subgraph,
    tune_gnn_model,
)
from run_gnn_per_region import (
    _GRAPH_ALPHA,
    _GRAPH_K,
    _GRAPH_MAX_DIST_KM,
    _split_region_indices,
)

_RESULTS_PATH = "result/hparam_sensitivity.json"
_SAVED_RESULTS_PATH = "result/gnn_per_region.json"

# Expanded grid: superset of the original tuning grid
_HIDDEN_DIMS = [16, 32, 64, 128, 256]
_LRS = [0.05, 0.01, 0.005, 0.001, 0.0005]
_ORIG_HIDDEN_DIMS = [32, 64, 128]
_ORIG_LRS = [0.01, 0.005, 0.001]


def _build_region_graph(processed_df: pd.DataFrame, region: str, seed: int):
    """Same graph-building pipeline as run_gnn_per_region._run_region,
    stopping right before model tuning."""
    region_mask = processed_df["Region"] == region
    region_global_idx = np.where(region_mask)[0]
    n_nodes = len(region_global_idx)
    print(f"Region '{region}': {n_nodes} nodes total")

    train_local, val_local, test_local = _split_region_indices(
        region_global_idx, random_seed=seed
    )

    train_global = region_global_idx[train_local]
    _, X_region, y_region = prepare_xgboost_inputs(processed_df, fit_idx=train_global)
    X_region = X_region.iloc[region_global_idx].reset_index(drop=True)
    y_region = y_region.iloc[region_global_idx].reset_index(drop=True)

    _, _, X_selected = train_xgboost_and_select_features(
        X_region, y_region, fit_idx=train_local
    )

    tv_local = np.sort(np.concatenate([train_local, val_local]))
    old_to_new = {int(old): new for new, old in enumerate(tv_local)}
    new_train = np.array([old_to_new[i] for i in train_local], dtype=np.int64)
    new_val   = np.array([old_to_new[i] for i in val_local],   dtype=np.int64)

    coords_region = processed_df[["Latitude", "Longitude"]].values[region_global_idx]
    coords_tv = coords_region[tv_local]

    # Leakage-free (Comment 12): train nodes get K-fold OOF scores, val nodes
    # get scores from the ensemble refit on the full train_local.
    X_node_tv, ensemble_models = build_node_features_no_leakage(
        X_selected.iloc[tv_local], X_region, y_region, train_local, val_local, new_train, new_val,
    )
    # Geo-only vector for cosine similarity (Comment 13): excludes
    # Latitude/Longitude and ensemble_score to avoid double-counting them
    # into the "geological similarity" edge weight.
    X_geo_tv = torch.tensor(
        select_geo_similarity_columns(X_selected.iloc[tv_local]).astype(float).values,
        dtype=torch.float32,
    )

    edge_index, edge_weight = build_weighted_graph(
        coords_tv, X_node_tv,
        alpha=_GRAPH_ALPHA, k=min(_GRAPH_K, len(tv_local) - 1),
        max_distance_km=_GRAPH_MAX_DIST_KM,
        fit_idx=new_train,
        X_geo=X_geo_tv,
        unify_edge_rule=True,  # Comment 14 Solution A (decided)
    )

    X_node_tv = augment_with_edge_stats(X_node_tv, edge_index, edge_weight)
    data = build_pyg_data(X_node_tv, edge_index, edge_weight, y_region.iloc[tv_local])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    data.x = data.x.detach()

    train_idx_t = torch.tensor(new_train, dtype=torch.long, device=device)
    val_idx_t   = torch.tensor(new_val,   dtype=torch.long, device=device)
    tv_data, sub_train, sub_val = build_train_val_subgraph(data, train_idx_t, val_idx_t)

    split_sizes = {
        "n_total": int(n_nodes),
        "n_train": int(len(train_local)),
        "n_val":   int(len(val_local)),
        "n_test":  int(len(test_local)),
    }
    return tv_data, sub_train, sub_val, device, split_sizes


def _grid_argmax(records: list, hidden_dims=None, lrs=None) -> dict:
    """Best record, optionally restricted to a sub-grid."""
    subset = [
        r for r in records
        if (hidden_dims is None or r["hidden_dim"] in hidden_dims)
        and (lrs is None or r["lr"] in lrs)
    ]
    return max(subset, key=lambda r: r["val_f1"])


def _load_saved_best_config(region: str):
    path = Path(_SAVED_RESULTS_PATH)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    for run in saved.get("runs", []):
        if run.get("run_id") == "orig":
            region_entry = run.get("per_region", {}).get(region)
            if region_entry:
                return {
                    "best_config": region_entry["best_config"],
                    "best_val_f1": region_entry["best_val_f1"],
                }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, default="South and Central Asia",
                        help="Representative region (default: largest region)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Split + torch seed (default: 42, same as the orig run)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    tv_data, sub_train, sub_val, device, split_sizes = _build_region_graph(
        processed_df, args.region, args.seed
    )

    print(f"\nGrid search: hidden={_HIDDEN_DIMS} x lr={_LRS} x layers=[1,2] x dropout=[0.3,0.5]")
    best_config, best_val_f1, records = tune_gnn_model(
        GraphSAGE, tv_data, sub_train, sub_val, device,
        hidden_dims=_HIDDEN_DIMS, lrs=_LRS, return_records=True,
    )

    orig_grid_best = _grid_argmax(records, _ORIG_HIDDEN_DIMS, _ORIG_LRS)
    saved = _load_saved_best_config(args.region)

    print(f"\n{'='*60}")
    print(f"  Expanded-grid best : {list(best_config)}  val_f1={best_val_f1:.4f}")
    print(
        f"  Original-grid best : "
        f"[{orig_grid_best['hidden_dim']}, {orig_grid_best['num_layers']}, "
        f"{orig_grid_best['dropout']}, {orig_grid_best['lr']}]  "
        f"val_f1={orig_grid_best['val_f1']:.4f}"
    )
    if saved is not None:
        print(
            f"  Saved (orig run)   : {saved['best_config']}  "
            f"val_f1={saved['best_val_f1']:.4f}"
        )
    print(f"{'='*60}")

    payload = {
        "experiment": "hparam_sensitivity",
        "description": (
            "Full grid-search surface (validation F1) for SPIRE on one "
            "representative region. Same graph pipeline and seed-42 split as "
            "result/gnn_per_region.json; expanded hidden_dim x lr grid."
        ),
        "region": args.region,
        "seed": args.seed,
        "model": "spire",
        **split_sizes,
        "search_space": {
            "hidden_dim": _HIDDEN_DIMS,
            "num_layers": [1, 2],
            "dropout": [0.3, 0.5],
            "lr": _LRS,
        },
        "original_search_space": {
            "hidden_dim": _ORIG_HIDDEN_DIMS,
            "num_layers": [1, 2],
            "dropout": [0.3, 0.5],
            "lr": _ORIG_LRS,
        },
        "best_config": list(best_config),
        "best_val_f1": float(best_val_f1),
        "original_grid_best": orig_grid_best,
        "saved_orig_run": saved,
        "records": records,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
