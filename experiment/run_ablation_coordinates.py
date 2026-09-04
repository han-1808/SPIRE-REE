"""
experiment/run_ablation_coordinates.py

Comment (1)-A follow-up: does SPIRE's F1 actually depend on having raw
Latitude/Longitude as a node feature, or does the mineral/lithology signal
carry it on its own?

Motivated by the SHAP-by-deposit-type diagnostic (run_deposit_type_analysis.py):
with Lat/Long available, Lat+Longitude combined SHAP is 3.4-6.3x larger than
the best mineral feature in 8/9 deposit-type groups -- i.e. the tabular
ensemble/feature-selection stage prefers the geographic shortcut over
mineralogy when both are available. This ablation checks whether that
SHAP-level preference actually translates into a real F1 dependency.

  full         : exclude_coords=False (current shipped behaviour) -- SHAP
                 top-10 selection may pick Latitude/Longitude if they rank
                 high (they usually do).
  no_coords    : exclude_coords=True -- Latitude/Longitude are removed from
                 the SHAP *candidate* pool before top-10 selection (reuses
                 prepare_xgboost_inputs's drop_coordinate_features flag), so
                 the freed slots go to the next-best mineral/lithology
                 features. Node-vector dimensionality is unchanged (13-dim
                 in mode="full" either way) -- a fair like-for-like
                 comparison, not just "fewer features".

Graph *topology* (spatial k-NN, still built from real Lat/Long via
coords_tv) is untouched in both conditions -- this ablation is specifically
about node-feature reliance, not about the graph's spatial structure
(a separate, already-settled design decision from Comment 13/14).

Runs on mode="full" (13-dim, WITH edge-stat augmentation) and SPIRE
(model_class=GraphSAGE) since that's the model the SHAP diagnostic itself
was run against, and the model the reviewer's comment (1) concerns.

Usage:
    python experiment/run_ablation_coordinates.py
    python experiment/run_ablation_coordinates.py --seeds 42 1 2
    python experiment/run_ablation_coordinates.py --regions Oceania "South and Central Asia"

Results saved to result/ablation_coordinates.json.
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

_RESULTS_PATH = "result/ablation_coordinates.json"
_MODE = "full"
_CONDITIONS = {
    "full":      {"exclude_coords": False},
    "no_coords": {"exclude_coords": True},
}


def run_coordinates_comparison(
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
        description="Comment (1)-A follow-up: full (Lat/Long allowed in node "
                     "features) vs no_coords (Lat/Long excluded from node features)"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", type=str, default=None,
                         help="Restrict to a subset of regions (default: all)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    results = run_coordinates_comparison(processed_df, args.seeds, args.regions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "ablation_coordinates",
        "description": (
            "Comment (1)-A follow-up: full (Latitude/Longitude allowed as node "
            "features if SHAP selects them) vs no_coords (Latitude/Longitude "
            "excluded from the SHAP candidate pool -- freed slots go to the "
            "next-best mineral/lithology features, same 13-dim total) on "
            "Exp 1 (leave-one-region-out inductive), mode=full, SPIRE. Graph "
            "topology (spatial k-NN) is unchanged in both conditions -- this "
            "tests node-feature reliance only."
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

    print("\n── Comment (1)-A coordinate-ablation summary (avg ± std across seeds) ──")
    print(f"{'Condition':<12} {'F1 (avg±std)':>18}")
    for cond in _CONDITIONS:
        avg = results[cond]["summary"]["avg_metrics"].get("f1")
        std = results[cond]["summary"]["std_metrics"].get("f1")
        print(f"{cond:<12} {avg:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
