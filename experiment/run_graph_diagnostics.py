"""
experiment/run_graph_diagnostics.py

Comment 13-D — graph diagnostics on the graph fixed in Comment 13-A:
  - Node degree distribution, broken down by Region.
  - Isolated / near-isolated node counts (and which nodes, by ID_No).
  - Edge-weight histograms for the spatial-only, geological-only (fixed,
    Comment 13-A), and combined (mixed, alpha=0.5) terms.

Builds ONE graph over the full dataset (same construction as
preprocessing/pipeline.py Steps 6-9, with the Comment 13-A geo-only fix) —
this is a structural diagnostic, not a predictive-performance experiment, so
there is no train/val/test split here; every node is described.

Comment 14-C (added here per the reviewer's own instruction to fold it into
this diagnostic) — compares neighbourhood size (degree), w_mean, w_max
between TRAIN and VAL nodes on one representative region's TV graph, under
both the asymmetric (current) and unified (Comment 14 Solution A) edge
rules, to quantify the train/val distribution shift the comment describes.
Reuses run_ablation._build_graph_artifacts directly rather than rebuilding
the TV-graph pipeline here.

Usage:
    python experiment/run_graph_diagnostics.py
    python experiment/run_graph_diagnostics.py --near-isolated-threshold 3
    python experiment/run_graph_diagnostics.py --shift-region Oceania

Results saved to result/graph_diagnostics.json, histogram plots to
result/graph_diagnostics_edge_weights.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import numpy as np
import pandas as pd

from config import KNN_K, OUTPUT_PATH
from feature_engineering import (
    build_node_features,
    build_weighted_graph,
    create_spatial_split_indices,
    prepare_xgboost_inputs,
    select_geo_similarity_columns,
    train_xgboost_and_select_features,
)

_GRAPH_ALPHA = 0.5
_GRAPH_MAX_DIST_KM = 2000.0  # same adaptive-k threshold used in every experiment script
_RESULTS_PATH = "result/graph_diagnostics.json"
_PLOT_PATH = "result/graph_diagnostics_edge_weights.png"
_DEFAULT_NEAR_ISOLATED_THRESHOLD = 2
_DEFAULT_TEST_REGION = "South and Central Asia"  # same default as preprocessing/pipeline.py


def _build_full_graph(processed_df: pd.DataFrame):
    """
    Same node-feature + graph construction as preprocessing/pipeline.py
    Steps 6-9 (with the Comment 13-A geo-only fix), over the WHOLE dataset —
    fit_idx is a representative train split (not the whole dataset) purely to
    match how features are normally fitted elsewhere in the codebase; every
    node still gets a feature vector and a place in the graph.
    """
    import torch

    split_indices = create_spatial_split_indices(
        processed_df, test_region=_DEFAULT_TEST_REGION, val_ratio=0.2,
    )
    fit_idx = split_indices["train_idx"]

    _, X, y = prepare_xgboost_inputs(processed_df, fit_idx=fit_idx)
    model, _, X_selected = train_xgboost_and_select_features(X, y, fit_idx=fit_idx)
    X_node = build_node_features(X_selected, X, model)
    X_geo = torch.tensor(
        select_geo_similarity_columns(X_selected).astype(float).values, dtype=torch.float32,
    )

    coords = processed_df[["Latitude", "Longitude"]].values

    edge_index = None
    edge_weights = {}
    for strategy_name, strategy in [
        ("spatial", "distance"),
        ("geo", "feature_similarity"),
        ("combined", "mixed"),
    ]:
        ei, ew = build_weighted_graph(
            coords, X_node, alpha=_GRAPH_ALPHA, k=KNN_K,
            max_distance_km=_GRAPH_MAX_DIST_KM,
            weighting_strategy=strategy, X_geo=X_geo,
        )
        if edge_index is None:
            edge_index = ei  # topology is identical across strategies (same k-NN + threshold)
        edge_weights[strategy_name] = ew.numpy()

    return edge_index.numpy(), edge_weights


def _degree_by_region(edge_index: np.ndarray, processed_df: pd.DataFrame) -> dict:
    n_nodes = len(processed_df)
    degree = np.bincount(edge_index[1], minlength=n_nodes)
    regions = processed_df["Region"].values

    per_region = {}
    for region in sorted(processed_df["Region"].dropna().unique().tolist()):
        mask = regions == region
        deg_r = degree[mask]
        per_region[region] = {
            "n_nodes": int(mask.sum()),
            "mean_degree": float(deg_r.mean()),
            "min_degree": int(deg_r.min()),
            "max_degree": int(deg_r.max()),
            "median_degree": float(np.median(deg_r)),
        }
    return degree, per_region


def _isolated_nodes(degree: np.ndarray, processed_df: pd.DataFrame, near_isolated_threshold: int) -> dict:
    id_col = processed_df["ID_No"] if "ID_No" in processed_df.columns else processed_df.index.to_series()

    isolated_mask = degree == 0
    near_isolated_mask = (degree > 0) & (degree <= near_isolated_threshold)

    return {
        "n_isolated": int(isolated_mask.sum()),
        "isolated_ids": [str(x) for x in id_col[isolated_mask].tolist()],
        "n_near_isolated": int(near_isolated_mask.sum()),
        "near_isolated_threshold": near_isolated_threshold,
        "near_isolated_ids": [str(x) for x in id_col[near_isolated_mask].tolist()],
    }


def _edge_weight_histograms(edge_weights: dict, n_bins: int = 20) -> dict:
    histograms = {}
    for name, ew in edge_weights.items():
        counts, bin_edges = np.histogram(ew, bins=n_bins, range=(0.0, 1.0))
        histograms[name] = {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "mean": float(ew.mean()),
            "std": float(ew.std()),
            "min": float(ew.min()),
            "max": float(ew.max()),
        }
    return histograms


def _save_histogram_plot(edge_weights: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    titles = {"spatial": "Spatial only", "geo": "Geological only (fixed)", "combined": "Combined (mixed, α=0.5)"}
    for ax, (name, ew) in zip(axes, edge_weights.items()):
        ax.hist(ew, bins=20, range=(0.0, 1.0), color="steelblue", edgecolor="white")
        ax.set_title(titles.get(name, name))
        ax.set_xlabel("edge weight")
    axes[0].set_ylabel("edge count")
    fig.suptitle("Edge-weight distributions (Comment 13-D)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _train_val_group_stats(degree: np.ndarray, w_mean_col: np.ndarray, w_max_col: np.ndarray, idx: np.ndarray) -> dict:
    idx = np.asarray(idx)
    return {
        "n": int(len(idx)),
        "degree_mean": float(degree[idx].mean()),
        "degree_std": float(degree[idx].std()),
        "w_mean_mean": float(w_mean_col[idx].mean()),
        "w_mean_std": float(w_mean_col[idx].std()),
        "w_max_mean": float(w_max_col[idx].mean()),
        "w_max_std": float(w_max_col[idx].std()),
    }


def run_train_val_shift_diagnostics(processed_df: pd.DataFrame, region: str) -> dict:
    """
    Comment 14-C: for one held-out-region TV graph, compare neighbourhood
    size (degree) / w_mean / w_max between train and val nodes, under both
    the asymmetric (current) and unified (Comment 14 Solution A) edge rules.
    """
    from run_ablation import _build_graph_artifacts

    stats_by_condition = {}
    for cond_name, unify in [("asymmetric", False), ("unified", True)]:
        print(f"  [train/val shift] region={region} condition={cond_name}...")
        art = _build_graph_artifacts(processed_df, region, mode="full", unify_edge_rule=unify)
        X = art["tv_data"].x.cpu().numpy()          # (N_tv, 13): last 2 cols are w_mean, w_max
        edge_index = art["edge_index_tv"].cpu().numpy()
        degree = np.bincount(edge_index[1], minlength=X.shape[0])
        w_mean_col, w_max_col = X[:, -2], X[:, -1]
        sub_train = art["sub_train"].cpu().numpy()
        sub_val = art["sub_val"].cpu().numpy()

        stats_by_condition[cond_name] = {
            "train": _train_val_group_stats(degree, w_mean_col, w_max_col, sub_train),
            "val": _train_val_group_stats(degree, w_mean_col, w_max_col, sub_val),
        }

    return {"region": region, "by_condition": stats_by_condition}


def run_graph_diagnostics(
    processed_df: pd.DataFrame,
    near_isolated_threshold: int = _DEFAULT_NEAR_ISOLATED_THRESHOLD,
    shift_region: str | None = _DEFAULT_TEST_REGION,
) -> tuple[dict, dict]:
    """Returns (json_safe_result, raw_edge_weights) — the latter for plotting."""
    print("Building full-dataset graph (Comment 13-A fix applied)...")
    edge_index, edge_weights = _build_full_graph(processed_df)

    print("Computing degree distribution by region...")
    degree, per_region = _degree_by_region(edge_index, processed_df)

    print("Finding isolated / near-isolated nodes...")
    isolation = _isolated_nodes(degree, processed_df, near_isolated_threshold)

    print("Computing edge-weight histograms (spatial / geo / combined)...")
    histograms = _edge_weight_histograms(edge_weights)

    result = {
        "n_nodes": len(processed_df),
        "n_edges": int(edge_index.shape[1]),
        "degree_by_region": per_region,
        "isolation": isolation,
        "edge_weight_histograms": histograms,
    }

    if shift_region is not None:
        print(f"Computing Comment 14-C train/val shift stats (region={shift_region})...")
        result["train_val_shift"] = run_train_val_shift_diagnostics(processed_df, shift_region)

    return result, edge_weights


def main():
    parser = argparse.ArgumentParser(description="Comment 13-D: graph diagnostics")
    parser.add_argument("--near-isolated-threshold", type=int, default=_DEFAULT_NEAR_ISOLATED_THRESHOLD)
    parser.add_argument("--shift-region", type=str, default=_DEFAULT_TEST_REGION,
                         help="Region to use for the Comment 14-C train/val shift comparison "
                              "(pass '' to skip it)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    parser.add_argument("--plot-out", type=str, default=_PLOT_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    shift_region = args.shift_region or None
    result, edge_weights = run_graph_diagnostics(processed_df, args.near_isolated_threshold, shift_region)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "graph_diagnostics",
        "description": (
            "Comment 13-D: graph diagnostics (degree distribution by region, "
            "isolated/near-isolated nodes, edge-weight histograms) on the "
            "full-dataset graph built with the Comment 13-A geo-only fix. "
            "Comment 14-C: train/val neighbourhood-size/w_mean/w_max shift "
            "under the asymmetric vs unified edge rule."
        ),
        **result,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    _save_histogram_plot(edge_weights, args.plot_out)
    print(f"Histogram plot saved → {args.plot_out}")

    print("\n── Degree by region ──────────────────────────────")
    print(f"{'Region':<28} {'n':>5} {'mean':>7} {'min':>5} {'max':>5}")
    for region, stats in result["degree_by_region"].items():
        print(f"{region:<28} {stats['n_nodes']:>5} {stats['mean_degree']:>7.2f} "
              f"{stats['min_degree']:>5} {stats['max_degree']:>5}")

    iso = result["isolation"]
    print(f"\nIsolated nodes: {iso['n_isolated']}")
    print(f"Near-isolated nodes (degree <= {iso['near_isolated_threshold']}): {iso['n_near_isolated']}")

    print("\n── Edge-weight summary ───────────────────────────")
    for name, h in result["edge_weight_histograms"].items():
        print(f"{name:<10} mean={h['mean']:.4f} std={h['std']:.4f} min={h['min']:.4f} max={h['max']:.4f}")

    if "train_val_shift" in result:
        shift = result["train_val_shift"]
        print(f"\n── Comment 14-C: train/val shift (region={shift['region']}) ──────")
        print(f"{'condition':<12} {'group':<6} {'n':>5} {'degree':>10} {'w_mean':>10} {'w_max':>10}")
        for cond, groups in shift["by_condition"].items():
            for grp_name, s in groups.items():
                print(f"{cond:<12} {grp_name:<6} {s['n']:>5} "
                      f"{s['degree_mean']:>6.2f}±{s['degree_std']:<3.2f} "
                      f"{s['w_mean_mean']:>6.3f}±{s['w_mean_std']:<3.3f} "
                      f"{s['w_max_mean']:>6.3f}±{s['w_max_std']:<3.3f}")


if __name__ == "__main__":
    main()
