"""
experiment/run_ablation_graph_weighting.py

Comment 13-B — report 4 parallel conditions for the graph edge weight.

IMPORTANT — runs on GCN, not SPIRE: while building this comparison we found
that SAGEConv (SPIRE/GraphSAGE) and the GATConv usage in this codebase do NOT
consume `edge_weight` in message passing (confirmed against PyTorch Geometric
2.6.1's `SAGEConv.forward` signature, which has no edge_weight parameter at
all). Only GCNConv (GCN) does. So on SPIRE, all 4 conditions below produce
byte-identical training curves and predictions in mode="no_edge_stats" (edge
weights only reach SPIRE indirectly via `augment_with_edge_stats` in
mode="full", as w_mean/w_max node features) — the comparison would be
uninformative there. This script therefore trains GCN (`model_class=GCN`)
instead, the only one of the three architectures where the 4 conditions
actually produce different graphs' worth of message passing.

4 conditions:
  spatial_only  : weighting_strategy="distance" — haversine spatial term only,
                  geological-similarity term unused.
  geo_only      : weighting_strategy="feature_similarity", geo_fix=True —
                  cosine similarity on the geo-only vector (Comment 13-A fix:
                  excludes Latitude/Longitude/ensemble_score).
  mixed_current : weighting_strategy="mixed", geo_fix=False — the pre-fix
                  "current formulation": cosine similarity computed on the
                  FULL node vector (still includes Latitude/Longitude/
                  ensemble_score), reproduced only for this comparison via
                  `geo_fix=False`.
  mixed_fixed   : weighting_strategy="mixed", geo_fix=True — alpha=0.5
                  combination of spatial + geo-only similarity (Comment 13-A
                  fix applied). This is the default used everywhere else in
                  the codebase.

All 4 conditions use mode="no_edge_stats" (11-dim: SHAP + ensemble) — edge-
stat augmentation (mode="full") is deliberately excluded so this comparison
isolates the graph-weighting effect: edge stats (w_mean/w_max) are themselves
derived from the edge weights being compared here, so including them would
fold the very thing under test back into the input, muddying the comparison.

Usage:
    python experiment/run_ablation_graph_weighting.py
    python experiment/run_ablation_graph_weighting.py --seeds 42 1 2
    python experiment/run_ablation_graph_weighting.py --regions Oceania "South and Central Asia"

Results saved to result/ablation_graph_weighting.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import pandas as pd

from config import OUTPUT_PATH
from gnn_training import GCN
from run_ablation import (
    _DEFAULT_SEEDS,
    run_ablation_sweep,
)

_RESULTS_PATH = "result/ablation_graph_weighting.json"
_MODE = "no_edge_stats"
_MODEL_CLASS = GCN  # SAGEConv/GATConv ignore edge_weight — see module docstring.
_CONDITIONS = {
    "spatial_only":  {"weighting_strategy": "distance",           "geo_fix": True},
    "geo_only":      {"weighting_strategy": "feature_similarity", "geo_fix": True},
    "mixed_current": {"weighting_strategy": "mixed",              "geo_fix": False},
    "mixed_fixed":   {"weighting_strategy": "mixed",              "geo_fix": True},
}


def run_graph_weighting_comparison(
    processed_df: pd.DataFrame,
    seeds: list[int],
    regions: list[str] | None = None,
) -> dict:
    # NOTE: `processed_df` is always passed in FULL (never pre-filtered) —
    # every region not currently held out is training data. `regions`
    # only restricts which regions are swept as the held-out test region.
    results: dict = {}

    for cond_name, cond_kwargs in _CONDITIONS.items():
        print(f"\n{'#'*60}\n  Condition: {cond_name}  ({cond_kwargs})\n{'#'*60}")
        cond_results, cond_summary = run_ablation_sweep(
            processed_df, modes=[_MODE], seeds=seeds, regions=regions,
            model_class=_MODEL_CLASS, **cond_kwargs,
        )
        results[cond_name] = {
            "runs": cond_results[_MODE],
            "summary": cond_summary[_MODE],
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comment 13-B: spatial-only vs geo-only vs mixed(current/buggy) vs mixed(fixed)"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", type=str, default=None,
                         help="Restrict to a subset of regions (default: all)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    results = run_graph_weighting_comparison(processed_df, args.seeds, args.regions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "ablation_graph_weighting",
        "description": (
            "Comment 13-B: graph edge-weighting comparison on Exp 1 "
            "(leave-one-region-out inductive), mode=no_edge_stats (11-dim). "
            "spatial_only=haversine distance only; geo_only=cosine similarity on "
            "the geo-only vector (Comment 13-A fix); mixed_current=alpha=0.5 "
            "combination with cosine similarity on the FULL node vector "
            "(pre-fix double-counting, reproduced only for this comparison); "
            "mixed_fixed=alpha=0.5 combination with the Comment 13-A fix applied "
            "(the default used everywhere else in the codebase). Trained on GCN, "
            "not SPIRE: SAGEConv/GATConv do not consume edge_weight in message "
            "passing (only GCNConv does), so on SPIRE these 4 conditions would "
            "produce identical predictions in mode=no_edge_stats — see module "
            "docstring."
        ),
        "model": "gcn",
        "conditions": {k: v for k, v in _CONDITIONS.items()},
        "seeds": args.seeds,
        "regions": args.regions,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print("\n── Comment 13-B summary (avg ± std across seeds) ──────────")
    print(f"{'Condition':<16} {'F1 (avg±std)':>18}")
    for cond in _CONDITIONS:
        avg = results[cond]["summary"]["avg_metrics"].get("f1")
        std = results[cond]["summary"]["std_metrics"].get("f1")
        print(f"{cond:<16} {avg:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
