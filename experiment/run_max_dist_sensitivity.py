"""
experiment/run_max_dist_sensitivity.py

Comment 13-C — graph construction max_distance_km sensitivity for SPIRE —
Exp 1 (leave-one-region-out inductive).

Reuses run_k_sensitivity.py's graph-building and inductive-prediction
functions (`_build_graph_artifacts`, `_predict_test_nodes`, `_compute_metrics`,
`_avg_metrics` — generalized there to take `max_dist_km` as a parameter
instead of duplicating them here) and only varies max_distance_km instead of
k. k=15 and alpha=0.5 fixed throughout, over the reviewer's own threshold
list {500, 1000, 1500, 2000, 2500, 3000} km. Seed=42.

Note: 2000km is the default used in run_gnn.py, so its avg F1 can be
cross-checked against result/gnn_inductive.json (seed=42 run).

Usage:
    python experiment/run_max_dist_sensitivity.py
    python experiment/run_max_dist_sensitivity.py --dists 1000 2000 3000

Results saved to result/max_dist_sensitivity.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import pandas as pd
import torch

from config import OUTPUT_PATH
from gnn_training import GraphSAGE, train_gnn_model, tune_gnn_model
from run_k_sensitivity import (
    _avg_metrics,
    _build_graph_artifacts,
    _compute_metrics,
    _predict_test_nodes,
)

_DEFAULT_DISTS = [500, 1000, 1500, 2000, 2500, 3000]
_RESULTS_PATH = "result/max_dist_sensitivity.json"


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_max_dist_sweep(processed_df: pd.DataFrame, dists: list[float]) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    print(f"Regions ({len(regions)}): {regions}")
    print(f"max_distance_km values: {dists}")

    results = {}
    for max_dist_km in dists:
        print(f"\n{'='*60}")
        print(f"  max_distance_km = {max_dist_km}")
        print(f"{'='*60}")

        per_region: dict = {}

        for region in regions:
            print(f"\n  Region: {region}")
            graph_artifacts = _build_graph_artifacts(processed_df, region, max_dist_km=max_dist_km)
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
            print(f"  [max_dist={max_dist_km}] {region}: {metrics}")

            per_region[region] = {
                "n_test": n_test,
                "best_config": list(best_config),
                "best_val_f1": float(best_val_f1),
                "metrics": metrics,
            }

        avg = _avg_metrics({r: d["metrics"] for r, d in per_region.items()})
        print(f"\n  [max_dist={max_dist_km}] Avg: {avg}")
        results[str(max_dist_km)] = {
            "max_distance_km": max_dist_km, "avg_metrics": avg, "per_region": per_region,
        }

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Graph max_distance_km sensitivity study (Exp 1 SPIRE)")
    parser.add_argument(
        "--dists", nargs="+", type=float, default=_DEFAULT_DISTS,
        help=f"max_distance_km values to sweep (default: {_DEFAULT_DISTS})",
    )
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    results = run_max_dist_sweep(processed_df, args.dists)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing results if partial run
    if out_path.exists() and args.dists != _DEFAULT_DISTS:
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f).get("results", {})
        existing.update(results)
        results = existing

    payload = {
        "experiment": "max_dist_sensitivity",
        "description": (
            "Comment 13-C: graph construction max_distance_km sensitivity for "
            "SPIRE on Exp 1 (leave-one-region-out inductive). Full 13-dim "
            "pipeline, k=15, alpha=0.5 fixed. Seed=42."
        ),
        "model": "spire",
        "max_distance_km_values": sorted(float(d) for d in results.keys()),
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Print sensitivity table
    print("\n── max_distance_km sensitivity summary ─────────────────────────")
    print(f"{'max_dist_km':>12}  {'Avg F1':>7}  {'Accuracy':>9}")
    print("-" * 35)
    for d in sorted(results.keys(), key=float):
        avg = results[d]["avg_metrics"]
        f1 = avg.get("f1", float("nan"))
        acc = avg.get("accuracy", float("nan"))
        marker = " *" if float(d) == 2000.0 else ""
        print(f"{d:>12}  {f1:>7.4f}  {acc:>9.4f}{marker}")
    print("  * default max_distance_km in run_gnn.py")


if __name__ == "__main__":
    main()
