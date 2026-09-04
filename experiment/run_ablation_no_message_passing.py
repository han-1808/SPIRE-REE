"""
experiment/run_ablation_no_message_passing.py

Comment (6)-C / (17)-B: no-message-passing control.

Compares SPIRE (GraphSAGE, with message passing) against the exact same
13-dim node-feature vector (mode="full": 10 SHAP-selected + ensemble_score
+ w_mean + w_max) fed through a plain MLP that never reads edge_index/
edge_weight (`gnn_training.MLP`, added for this purpose). Isolates the
contribution of message passing itself, independent of the ensemble/
edge-stat input features -- which (6)-A's mode ablation already varies
separately. Directly answers (6)-C ("compare SPIRE against the same node
vector fed through a plain MLP") and (17)-B ("no-message-passing control"),
which the reviewer comments explicitly note are the same experiment.

Reuses `run_ablation.py`'s existing `model_class` parameter (already
generalized for the Comment 13-B GCN comparison) -- no new training/eval
logic, just `model_class=MLP` instead of the default `GraphSAGE`.

Usage:
    python experiment/run_ablation_no_message_passing.py
    python experiment/run_ablation_no_message_passing.py --seeds 42 1 2
    python experiment/run_ablation_no_message_passing.py --regions Oceania "South and Central Asia"

Results saved to result/ablation_no_message_passing.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import pandas as pd

from config import OUTPUT_PATH
from gnn_training import GraphSAGE, MLP
from run_ablation import _DEFAULT_SEEDS, run_ablation_sweep

_RESULTS_PATH = "result/ablation_no_message_passing.json"
_MODE = "full"
_CONDITIONS = {
    "spire": GraphSAGE,
    "mlp_no_message_passing": MLP,
}


def run_no_message_passing_comparison(
    processed_df: pd.DataFrame,
    seeds: list[int],
    regions: list[str] | None = None,
) -> dict:
    results: dict = {}
    for cond_name, model_class in _CONDITIONS.items():
        print(f"\n{'#'*60}\n  Condition: {cond_name}  (model_class={model_class.__name__})\n{'#'*60}")
        cond_results, cond_summary = run_ablation_sweep(
            processed_df, modes=[_MODE], seeds=seeds, regions=regions, model_class=model_class,
        )
        results[cond_name] = {
            "runs": cond_results[_MODE],
            "summary": cond_summary[_MODE],
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comment (6)-C / (17)-B: SPIRE (message passing) vs MLP "
                     "(same 13-dim node vector, no graph)"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", type=str, default=None,
                         help="Restrict to a subset of regions (default: all)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    results = run_no_message_passing_comparison(processed_df, args.seeds, args.regions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "ablation_no_message_passing",
        "description": (
            "Comment (6)-C / (17)-B: SPIRE (GraphSAGE, mode=full 13-dim) vs "
            "the same 13-dim node vector fed through a plain MLP that never "
            "reads edge_index/edge_weight -- isolates the contribution of "
            "message passing itself, independent of the ensemble/edge-stat "
            "input features."
        ),
        "conditions": {k: v.__name__ for k, v in _CONDITIONS.items()},
        "seeds": args.seeds,
        "regions": args.regions,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print("\n── Comment (6)-C / (17)-B summary (avg ± std across seeds) ──")
    print(f"{'Condition':<26} {'F1 (avg±std)':>18}")
    for cond in _CONDITIONS:
        avg = results[cond]["summary"]["avg_metrics"].get("f1")
        std = results[cond]["summary"]["std_metrics"].get("f1")
        print(f"{cond:<26} {avg:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
