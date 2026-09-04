"""
experiment/run_ablation_edge_rule.py

Comment 14-B — report the asymmetric (current) vs unified (Solution A) edge
rule in parallel:
  asymmetric : unify_edge_rule=False (default, current shipped behaviour) —
               the combined weight (spatial+geo) applies only to edges where
               BOTH endpoints are train nodes; edges touching a val node fall
               back to spatial-only.
  unified    : unify_edge_rule=True (Comment 14 Solution A) — the same
               combined-weight formula applies to every edge regardless of
               train/val membership. Safe because w_geo depends only on
               observable features (x), never the target (y).

Runs on mode="full" (13-dim, WITH edge-stat augmentation) and on SPIRE: unlike
Comment 13's graph-weighting comparison (which needs GCN, since SAGEConv
ignores edge_weight in message passing), Comment 14's concern is specifically
that w_mean/w_max — node features derived from edge weights via
augment_with_edge_stats — differ in distribution between train and val nodes
under the asymmetric rule. That channel reaches SPIRE directly as an input
feature, regardless of what SAGEConv does with edge_weight itself, so SPIRE
mode="full" is the right place to test it (no GCN workaround needed here).

Usage:
    python experiment/run_ablation_edge_rule.py
    python experiment/run_ablation_edge_rule.py --seeds 42 1 2
    python experiment/run_ablation_edge_rule.py --regions Oceania "South and Central Asia"

Results saved to result/ablation_edge_rule.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import pandas as pd

from config import OUTPUT_PATH
from run_ablation import (
    _DEFAULT_SEEDS,
    run_ablation_sweep,
)

_RESULTS_PATH = "result/ablation_edge_rule.json"
_MODE = "full"
_CONDITIONS = {
    "asymmetric": {"unify_edge_rule": False},
    "unified":    {"unify_edge_rule": True},
}


def run_edge_rule_comparison(
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
            processed_df, modes=[_MODE], seeds=seeds, regions=regions, **cond_kwargs,
        )
        results[cond_name] = {
            "runs": cond_results[_MODE],
            "summary": cond_summary[_MODE],
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comment 14-B: asymmetric (current) vs unified (Solution A) edge rule"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", type=str, default=None,
                         help="Restrict to a subset of regions (default: all)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    results = run_edge_rule_comparison(processed_df, args.seeds, args.regions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "ablation_edge_rule",
        "description": (
            "Comment 14-B: asymmetric (current, train-train edges get the "
            "combined weight, edges touching val nodes fall back to "
            "spatial-only) vs unified (Comment 14 Solution A, same formula "
            "for every edge) on Exp 1 (leave-one-region-out inductive), "
            "mode=full (13-dim, with edge-stat augmentation — this is the "
            "channel Comment 14 is actually about)."
        ),
        "model": "spire",
        "conditions": _CONDITIONS,
        "seeds": args.seeds,
        "regions": args.regions,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print("\n── Comment 14-B summary (avg ± std across seeds) ──────────")
    print(f"{'Condition':<12} {'F1 (avg±std)':>18}")
    for cond in _CONDITIONS:
        avg = results[cond]["summary"]["avg_metrics"].get("f1")
        std = results[cond]["summary"]["std_metrics"].get("f1")
        print(f"{cond:<12} {avg:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
