"""
experiment/run_ablation_ensemble_only.py

Comment (17)-C: Ensemble-only baseline. Thresholds the already-fitted
`ensemble_score` (mean of OOF-fit XGBoost/CatBoost/RandomForest
probabilities, Comment 12's leakage-free fix) directly at 0.5, with NO
graph/message-passing involved, and compares its F1 against SPIRE on the
same Exp1 (leave-one-region-out) split. Isolates how much of SPIRE's
performance the ensemble prior alone already accounts for, independent of
message passing or edge-derived features (w_mean/w_max) -- directly
addresses the reviewer's "ensemble-only model" control request.

SPIRE's numbers come from `run_ablation.run_ablation_sweep` unmodified (no
duplicated training/eval logic). The ensemble-only baseline is read
straight off `_build_graph_artifacts`'s cached `ensemble_models`/`X_test`
-- the same non-graph-baseline pattern already used in run_lodto.py's
(1)-C comparison. Since the split/ensemble fit is built once per region
(mode="full", `run_ablation.py`'s existing design: only the GNN's own
training varies across "seeds", not the split or the ensemble models --
see `_build_graph_artifacts`'s module docstring), the ensemble-only
baseline is deterministic per region and does NOT vary across seeds, so
it's computed once per region rather than once per (region, seed).

Usage:
    python experiment/run_ablation_ensemble_only.py
    python experiment/run_ablation_ensemble_only.py --seeds 42 1 2
    python experiment/run_ablation_ensemble_only.py --regions Oceania "South and Central Asia"

Results saved to result/ablation_ensemble_only.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import numpy as np
import pandas as pd

from config import OUTPUT_PATH
from run_ablation import (
    _DEFAULT_SEEDS,
    _build_graph_artifacts,
    _compute_metrics,
    _region_avg,
    run_ablation_sweep,
)

_RESULTS_PATH = "result/ablation_ensemble_only.json"
_MODE = "full"


def run_ensemble_only_baseline(
    processed_df: pd.DataFrame,
    regions: list[str] | None = None,
) -> dict:
    if regions is None:
        regions = sorted(processed_df["Region"].dropna().unique().tolist())

    per_region: dict = {}
    for region in regions:
        print(f"    [ensemble_only] {region}...")
        graph_artifacts = _build_graph_artifacts(processed_df, region, _MODE)
        y_true = graph_artifacts["y_test"].to_numpy()
        X_test = graph_artifacts["X_test"]
        ensemble_models = graph_artifacts["ensemble_models"]

        ens_prob = np.mean([m.predict_proba(X_test)[:, 1] for m in ensemble_models], axis=0)
        ens_pred = (ens_prob >= 0.5).astype(int)
        metrics = _compute_metrics(y_true, ens_pred, ens_prob)
        print(f"      f1={metrics['f1']:.4f}")

        per_region[region] = {"n_test": int(len(y_true)), "metrics": metrics}

    return {"avg_metrics": _region_avg(per_region), "per_region": per_region}


def main():
    parser = argparse.ArgumentParser(
        description="Comment (17)-C: ensemble_score (thresholded, no graph) vs SPIRE on Exp 1"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", type=str, default=None,
                         help="Restrict to a subset of regions (default: all)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)
    regions = args.regions or sorted(processed_df["Region"].dropna().unique().tolist())

    print("\n── SPIRE (mode=full, Exp 1 leave-one-region-out) ──")
    spire_results, spire_summary = run_ablation_sweep(
        processed_df, modes=[_MODE], seeds=args.seeds, regions=args.regions,
    )

    print("\n── Ensemble-only baseline (no graph, threshold ensemble_score @ 0.5) ──")
    ensemble_only = run_ensemble_only_baseline(processed_df, regions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "ablation_ensemble_only",
        "description": (
            "Comment (17)-C: ensemble_score (mean of OOF-fit XGBoost/CatBoost/"
            "RandomForest, thresholded at 0.5, no graph/message-passing) vs "
            "SPIRE (mode=full, 13-dim) on Exp 1 (leave-one-region-out). "
            "Ensemble-only is deterministic per region (no seed variance); "
            "SPIRE is averaged across seeds (GNN training variance only, "
            "same split/ensemble per region across seeds -- run_ablation.py's "
            "existing design)."
        ),
        "regions": regions,
        "seeds": args.seeds,
        "spire": {"runs": spire_results[_MODE], "summary": spire_summary[_MODE]},
        "ensemble_only": ensemble_only,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print("\n── Comment (17)-C summary ──")
    spire_f1 = spire_summary[_MODE]["avg_metrics"].get("f1")
    spire_f1_std = spire_summary[_MODE]["std_metrics"].get("f1")
    ens_f1 = ensemble_only["avg_metrics"].get("f1")
    print(f"SPIRE          F1 = {spire_f1:.4f} ± {spire_f1_std:.4f}")
    print(f"Ensemble-only  F1 = {ens_f1:.4f}")
    print(f"Delta (SPIRE - ensemble-only) = {spire_f1 - ens_f1:+.4f}")


if __name__ == "__main__":
    main()
