"""
experiment/run_class_counts_per_fold.py

Comment (4)-C: report class counts per fold.

Unlike (4)-D, this needs no model at all -- just the split itself. Computes
the exact same split each real experiment script will produce (same
`test_ratio`/`k_for_buffer`/seed, via the same shared
`create_buffered_region_split_indices` / `create_buffered_sample_split_indices`
functions those scripts call), for every region x every seed the paper
actually uses (from README.md's documented seed lists), and reports the
has_ree=0/1 count in each fold's test set.

This directly quantifies the reviewer's "one flipped case can materially
change F1" concern: a fold with e.g. 8 positive / 1 negative test nodes
has very little room for the minority class to be estimated reliably, no
matter how large ROC-AUC/PR-AUC come out.

Exp2 (`run_gnn_per_region.py` / `run_baseline_per_region.py`, identical
split) and Exp5 (`run_exp5_proportion_per_region.py`) use
`create_buffered_region_split_indices` (train/val confined to the region).
Exp3 (`run_inductive_sample.py`) and Exp6 (`run_exp6_proportion_inductive_sample.py`,
imports Exp3's split function) use `create_buffered_sample_split_indices`
(train/val spans the whole dataset) -- note this uses one `rng` shared
*across* the region loop within a seed (matching the real scripts exactly,
not a fresh RNG per region), so results here only match the real run if
regions are iterated in the same sorted order.

Usage:
    python experiment/run_class_counts_per_fold.py

Results saved to result/class_counts_per_fold.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import numpy as np
import pandas as pd

from config import OUTPUT_PATH
from feature_engineering import (
    create_buffered_region_split_indices,
    create_buffered_sample_split_indices,
)

_TEST_RATIO = 0.15
_OUT_PATH = "result/class_counts_per_fold.json"

# Exact seed lists used for each experiment's "5-run average" table in README.md.
_EXP_SEEDS = {
    "exp2": [42, 50, 32, 17, 21],
    "exp3": [42, 23, 10, 16, 49],
    "exp5": [42, 18, 20, 48, 50],
    "exp6": [42, 20, 11, 60, 48],
}
_EXP_K_FOR_BUFFER = {"exp2": 10, "exp3": 15, "exp5": 10, "exp6": 15}
_EXP_KIND = {"exp2": "region", "exp3": "sample", "exp5": "region", "exp6": "sample"}


def _region_fold_counts(processed_df: pd.DataFrame, region: str, seed: int, k_for_buffer: int) -> dict:
    region_global_idx = np.where(processed_df["Region"].values == region)[0]
    coords_region = processed_df[["Latitude", "Longitude"]].values[region_global_idx]
    split = create_buffered_region_split_indices(
        region_global_idx, coords_region, test_ratio=_TEST_RATIO,
        k_for_buffer=k_for_buffer, random_seed=seed,
    )
    test_global = region_global_idx[split["test_local"]]
    y_test = processed_df["has_ree"].values[test_global]
    return {
        "n_test": int(len(y_test)),
        "n_positive": int(y_test.sum()),
        "n_negative": int(len(y_test) - y_test.sum()),
    }


def _sample_fold_counts_for_seed(processed_df: pd.DataFrame, regions: list, seed: int, k_for_buffer: int) -> dict:
    """Mirrors run_inductive_sample.py's `_run_one_seed`: one shared `rng`
    advanced across the region loop within this seed, in sorted region order."""
    rng = np.random.default_rng(seed)
    out = {}
    for i, region in enumerate(regions):
        split = create_buffered_sample_split_indices(
            processed_df, region, rng, test_ratio=_TEST_RATIO,
            k_for_buffer=k_for_buffer, val_seed=seed + i,
        )
        y_test = processed_df["has_ree"].values[split["test_idx"]]
        out[region] = {
            "n_test": int(len(y_test)),
            "n_positive": int(y_test.sum()),
            "n_negative": int(len(y_test) - y_test.sum()),
        }
    return out


def run_class_counts(processed_df: pd.DataFrame) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    results: dict = {}

    for exp_name, seeds in _EXP_SEEDS.items():
        kind = _EXP_KIND[exp_name]
        k_for_buffer = _EXP_K_FOR_BUFFER[exp_name]
        print(f"\n[{exp_name}] kind={kind} k_for_buffer={k_for_buffer} seeds={seeds}")
        exp_result: dict = {}

        for seed in seeds:
            if kind == "region":
                exp_result[str(seed)] = {
                    region: _region_fold_counts(processed_df, region, seed, k_for_buffer)
                    for region in regions
                }
            else:
                exp_result[str(seed)] = _sample_fold_counts_for_seed(
                    processed_df, regions, seed, k_for_buffer
                )

        # Flag folds where the minority class has very little room (<=1 node)
        single_class_or_near_folds = []
        for seed, per_region in exp_result.items():
            for region, counts in per_region.items():
                if min(counts["n_positive"], counts["n_negative"]) <= 1:
                    single_class_or_near_folds.append(
                        f"{region}/seed={seed} (pos={counts['n_positive']}, neg={counts['n_negative']})"
                    )
        print(f"  {len(single_class_or_near_folds)} / {len(seeds) * len(regions)} folds have "
              f"<=1 minority-class test node:")
        for s in single_class_or_near_folds[:10]:
            print(f"    {s}")
        if len(single_class_or_near_folds) > 10:
            print(f"    ... and {len(single_class_or_near_folds) - 10} more")

        results[exp_name] = {
            "seeds": seeds, "k_for_buffer": k_for_buffer,
            "per_seed_per_region": exp_result,
            "n_folds_with_minority_leq_1": len(single_class_or_near_folds),
            "n_folds_total": len(seeds) * len(regions),
        }

    return results


def main():
    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    results = run_class_counts(processed_df)

    Path("result").mkdir(exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {_OUT_PATH}")


if __name__ == "__main__":
    main()
