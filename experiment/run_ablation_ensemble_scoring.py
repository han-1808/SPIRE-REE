"""
experiment/run_ablation_ensemble_scoring.py

Comment 12-C — report 3 parallel conditions for the ensemble prior score:
  oof        : leakage-free (Comment 12-A fix) — training nodes get K-fold
               out-of-fold scores, val/test nodes get scores from the
               ensemble refit on the full train_idx.
  in_sample  : pre-fix behaviour — training nodes scored by the same models
               that were fit on them (overfit / leaky), reproduced only for
               this comparison via `train_scoring="in_sample"`.
  no_ensemble: no ensemble prior at all (SHAP features only).

All 3 conditions use mode="no_edge_stats" (11-dim: SHAP + ensemble) for the
first two, and mode="no_ensemble" (10-dim: SHAP only) for the third — edge-
stat augmentation (mode="full") is deliberately excluded so this comparison
isolates the ensemble-scoring effect from the separate double-counting issue
in graph construction (Comment 13, not yet fixed).

Usage:
    python experiment/run_ablation_ensemble_scoring.py
    python experiment/run_ablation_ensemble_scoring.py --seeds 42 1 2
    python experiment/run_ablation_ensemble_scoring.py --regions Oceania "South and Central Asia"

Results saved to result/ablation_ensemble_scoring.json.
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
    _avg_metrics,
    _std_metrics,
    run_ablation_sweep,
)

_RESULTS_PATH = "result/ablation_ensemble_scoring.json"
_CONDITIONS = ["oof", "in_sample", "no_ensemble"]


def run_ensemble_scoring_comparison(
    processed_df: pd.DataFrame,
    seeds: list[int],
    regions: list[str] | None = None,
) -> dict:
    # NOTE: `processed_df` is always passed in FULL (never pre-filtered) —
    # every region not currently held out is training data. `regions`
    # only restricts which regions are swept as the held-out test region.
    results: dict = {}

    print(f"\n{'#'*60}\n  Condition: oof (leakage-free, Comment 12-A)\n{'#'*60}")
    oof_results, oof_summary = run_ablation_sweep(
        processed_df, modes=["no_edge_stats"], seeds=seeds, train_scoring="oof", regions=regions,
    )
    results["oof"] = {
        "runs": oof_results["no_edge_stats"],
        "summary": oof_summary["no_edge_stats"],
    }

    print(f"\n{'#'*60}\n  Condition: in_sample (pre-fix, leaky — comparison only)\n{'#'*60}")
    ins_results, ins_summary = run_ablation_sweep(
        processed_df, modes=["no_edge_stats"], seeds=seeds, train_scoring="in_sample", regions=regions,
    )
    results["in_sample"] = {
        "runs": ins_results["no_edge_stats"],
        "summary": ins_summary["no_edge_stats"],
    }

    print(f"\n{'#'*60}\n  Condition: no_ensemble (no ensemble prior)\n{'#'*60}")
    ne_results, ne_summary = run_ablation_sweep(
        processed_df, modes=["no_ensemble"], seeds=seeds, train_scoring="oof", regions=regions,
    )
    results["no_ensemble"] = {
        "runs": ne_results["no_ensemble"],
        "summary": ne_summary["no_ensemble"],
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comment 12-C: OOF vs in-sample vs no-ensemble comparison"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", type=str, default=None,
                         help="Restrict to a subset of regions (default: all)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    results = run_ensemble_scoring_comparison(processed_df, args.seeds, args.regions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "ablation_ensemble_scoring",
        "description": (
            "Comment 12-C: ensemble-prior scoring comparison for training nodes on "
            "Exp 1 (leave-one-region-out inductive), mode=no_edge_stats (11-dim) for "
            "oof/in_sample, mode=no_ensemble (10-dim) for the third condition. "
            "oof=leakage-free (Comment 12-A fix), in_sample=pre-fix leaky behaviour "
            "(reproduced only for this comparison), no_ensemble=no ensemble prior."
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

    print("\n── Comment 12-C summary (avg ± std across seeds) ──────────")
    print(f"{'Condition':<14} {'F1 (avg±std)':>18}")
    for cond in _CONDITIONS:
        avg = results[cond]["summary"]["avg_metrics"].get("f1")
        std = results[cond]["summary"]["std_metrics"].get("f1")
        print(f"{cond:<14} {avg:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
