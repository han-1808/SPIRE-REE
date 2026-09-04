"""
experiment/run_ablation_minerals.py

Comment (2)-B: mineral / inferred-feature ablation.

Compares SPIRE (mode="full") with vs without the 6 mineral-derived
columns (`has_ilmenite`, `has_magnetite`, `has_barite`, `has_zircon`,
`has_pyrochlore`, `has_fluorite`) available anywhere in the pipeline --
not just dropped from the final node vector, but excluded from the SHAP
*candidate* pool before top-10 selection, so all 3 leak-back paths named
in the solution are patched simultaneously (reuses `run_ablation.py`'s
`exclude_minerals` flag, added for this purpose):
  - Mineral Feature Engineering : the node vector itself can't contain
    them (SHAP can't select what isn't in the candidate pool).
  - Feature Selection/Ensemble prior : the ensemble (XGBoost/CatBoost/RF)
    is fit on the same mineral-stripped `X`, so `ensemble_score` can't
    carry mineral signal back in either.
  - Graph Construction : `w_feature` (cosine similarity) is built from
    `X_selected`, which is now mineral-free too, so it can't leak through
    `w_mean`/`w_max` either.

Dimension compensation: the 9 `is_group` deposit-type indicators
(Comment 1) are appended to the stripped node vector (skipping any
already SHAP-selected), so `full` (with_minerals) and `no_minerals` both
stay at a comparable ~13-dim, isolating the mineral features' own
contribution rather than confounding it with "fewer total dimensions".

Interpretation note (per the solution text): a sharp F1 drop here does
NOT by itself mean "the model learns documentation patterns" -- these are
genuine geological signal (diagnostic minerals). This ablation answers
"how much do these specific mineral features contribute", nothing more;
see (2)-C's documentation-completeness probe for the artifact question.

Usage:
    python experiment/run_ablation_minerals.py
    python experiment/run_ablation_minerals.py --seeds 42 1 2
    python experiment/run_ablation_minerals.py --regions Oceania "South and Central Asia"

Results saved to result/ablation_minerals.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import pandas as pd

from config import OUTPUT_PATH
from run_ablation import _DEFAULT_SEEDS, run_ablation_sweep

_RESULTS_PATH = "result/ablation_minerals.json"
_MODE = "full"
_CONDITIONS = {
    "with_minerals": {"exclude_minerals": False},
    "no_minerals":   {"exclude_minerals": True},
}


def run_minerals_comparison(
    processed_df: pd.DataFrame,
    seeds: list[int],
    regions: list[str] | None = None,
) -> dict:
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
        description="Comment (2)-B: with_minerals (current) vs no_minerals "
                     "(6 mineral-derived columns excluded everywhere, "
                     "compensated with the 9 is_group indicators)"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", type=str, default=None,
                         help="Restrict to a subset of regions (default: all)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    results = run_minerals_comparison(processed_df, args.seeds, args.regions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "ablation_minerals",
        "description": (
            "Comment (2)-B: with_minerals (current, 6 mineral-derived "
            "columns available to SHAP selection/ensemble/graph) vs "
            "no_minerals (excluded from the SHAP candidate pool, ensemble "
            "refit without them, graph rebuilt without them, compensated "
            "with the 9 is_group indicators for dimension parity) on "
            "Exp 1 (leave-one-region-out inductive), mode=full, SPIRE."
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

    print("\n── Comment (2)-B summary (avg ± std across seeds) ──────────")
    print(f"{'Condition':<16} {'F1 (avg±std)':>18}")
    for cond in _CONDITIONS:
        avg = results[cond]["summary"]["avg_metrics"].get("f1")
        std = results[cond]["summary"]["std_metrics"].get("f1")
        print(f"{cond:<16} {avg:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
