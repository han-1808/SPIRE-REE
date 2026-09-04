"""
experiment/run_lodto.py

Comment (1)-C: Leave-One-Deposit-Type-Out (LODTO).

Analogous to Exp 1's leave-one-region-out, but held out by deposit-type
group (one of the 9 `is_*` columns) instead of `Region`: train on records
NOT carrying the held-out group's flag, test on every record that does
carry it (create_deposit_type_split_indices, preprocessing/feature_engineering.py).
Since the 9 groups are not mutually exclusive, test sets for different
held-out groups can overlap -- see that function's docstring.

Reuses run_ablation.py's `_build_graph_artifacts`/`_train_and_predict` via
the `split=` hook added for this purpose, rather than duplicating the graph
build/train/predict logic -- only the split *indices* differ from Exp 1's
ablation sweep, everything downstream (graph construction, GNN training,
inductive test-node prediction) is identical.

Non-graph baseline: the ensemble_score itself (mean of XGBoost/CatBoost/RF
probabilities, already fit as part of the leakage-free node-feature
pipeline -- Comment 12), thresholded at 0.5. Reused as-is rather than
building a separate model, since it's already computed for every LODTO run
and is a genuine non-graph (no message passing) prediction.

Motivated by (1)-A/(1)-B: (1)-A found the model prefers a geographic
shortcut over deposit-type-specific mineral signal when both are available
(see run_ablation_coordinates.py); (1)-B found a strong negative
correlation between per-region deposit-type entropy and F1. LODTO is the
direct test of whether SPIRE can actually generalize to a *completely
unseen* deposit type, or whether performance collapses without in-group
training examples.

Usage:
    python experiment/run_lodto.py
    python experiment/run_lodto.py --seeds 42 1 2
    python experiment/run_lodto.py --groups is_carbonatite_system is_placer

Results saved to result/lodto.json.
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
from feature_engineering import create_deposit_type_split_indices
from gnn_training import GraphSAGE
from run_ablation import _build_graph_artifacts, _compute_metrics, _train_and_predict
from run_deposit_type_analysis import IS_GROUP_COLS

_RESULTS_PATH = "result/lodto.json"
_MODE = "full"
_DEFAULT_SEEDS = [42, 1, 2]


def _run_one_group(processed_df: pd.DataFrame, held_out_group: str, seed: int) -> dict:
    split = create_deposit_type_split_indices(
        processed_df, held_out_group, val_ratio=0.2, random_seed=seed
    )
    n_test = len(split["test_idx"])
    print(f"    {held_out_group}: n_train={len(split['train_idx'])} "
          f"n_val={len(split['val_idx'])} n_test={n_test}")

    graph_artifacts = _build_graph_artifacts(
        processed_df, held_out_group, _MODE, split=split
    )
    y_true = graph_artifacts["y_test"].to_numpy()

    y_prob_spire, y_pred_spire, best_config, best_val_f1 = _train_and_predict(
        graph_artifacts, processed_df, _MODE, model_class=GraphSAGE
    )
    spire_metrics = _compute_metrics(y_true, y_pred_spire, y_prob_spire)

    # Non-graph baseline: threshold the already-fitted ensemble_score directly.
    X_test = graph_artifacts["X_test"]
    ensemble_models = graph_artifacts["ensemble_models"]
    ens_prob = np.mean([m.predict_proba(X_test)[:, 1] for m in ensemble_models], axis=0)
    ens_pred = (ens_prob >= 0.5).astype(int)
    baseline_metrics = _compute_metrics(y_true, ens_pred, ens_prob)

    return {
        "n_test": n_test,
        "best_config": best_config,
        "best_val_f1": best_val_f1,
        "spire": spire_metrics,
        "ensemble_baseline": baseline_metrics,
    }


def run_lodto_sweep(
    processed_df: pd.DataFrame,
    seeds: list[int],
    groups: list[str] | None = None,
) -> dict:
    if groups is None:
        groups = IS_GROUP_COLS
    print(f"Deposit-type groups ({len(groups)}): {groups}")
    print(f"Seeds: {seeds}")

    runs: dict = {}
    for seed in seeds:
        print(f"\n{'#'*60}\n  RUN seed={seed}\n{'#'*60}")
        per_group: dict = {}
        for g in groups:
            per_group[g] = _run_one_group(processed_df, g, seed)
            spire_f1 = per_group[g]["spire"]["f1"]
            base_f1 = per_group[g]["ensemble_baseline"]["f1"]
            print(f"    [{g}] spire_f1={spire_f1:.4f}  ensemble_baseline_f1={base_f1:.4f}")
        runs[str(seed)] = per_group

    return runs


def _avg_across_seeds(runs: dict, groups: list[str]) -> dict:
    avg: dict = {}
    for g in groups:
        spire_f1s = [runs[s][g]["spire"]["f1"] for s in runs]
        base_f1s = [runs[s][g]["ensemble_baseline"]["f1"] for s in runs]
        avg[g] = {
            "spire_f1_mean": float(np.mean(spire_f1s)),
            "spire_f1_std": float(np.std(spire_f1s)),
            "ensemble_baseline_f1_mean": float(np.mean(base_f1s)),
            "ensemble_baseline_f1_std": float(np.std(base_f1s)),
            "n_test": runs[next(iter(runs))][g]["n_test"],
        }
    return avg


def main():
    parser = argparse.ArgumentParser(description="Comment (1)-C: Leave-One-Deposit-Type-Out")
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--groups", nargs="+", type=str, default=None,
                         choices=IS_GROUP_COLS, help="Restrict to a subset of groups (default: all 9)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    groups = args.groups or IS_GROUP_COLS
    runs = run_lodto_sweep(processed_df, args.seeds, groups)
    avg = _avg_across_seeds(runs, groups)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "lodto",
        "description": (
            "Comment (1)-C: leave-one-deposit-type-out. Train on 8 groups, "
            "test on the held-out group's records (multi-label -- test sets "
            "for different held-out groups can overlap). SPIRE (mode=full, "
            "13-dim) vs a non-graph baseline (ensemble_score thresholded at 0.5)."
        ),
        "groups": groups,
        "seeds": args.seeds,
        "avg_across_seeds": avg,
        "runs": runs,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print("\n── Comment (1)-C LODTO summary (avg ± std across seeds) ──")
    print(f"{'Group':<28} {'n_test':>7} {'SPIRE F1':>16} {'Baseline F1':>16} {'Delta':>8}")
    for g in groups:
        a = avg[g]
        delta = a["spire_f1_mean"] - a["ensemble_baseline_f1_mean"]
        print(f"{g:<28} {a['n_test']:>7} "
              f"{a['spire_f1_mean']:.4f}±{a['spire_f1_std']:.4f}  "
              f"{a['ensemble_baseline_f1_mean']:.4f}±{a['ensemble_baseline_f1_std']:.4f}  "
              f"{delta:>+.4f}")


if __name__ == "__main__":
    main()
