"""
experiment/run_check_exp5_splits.py

Reviewer #2, second round, Remark 5: verify that the Experiment 5 folds can be
re-created from their seeds.

`exp5_proportion_per_region.json` stores each fold's test labels (`y_true`) and
`split_info`, but not the record indices. This script needs no model: it
re-creates every reported fold with the same split call as
`run_exp5_proportion_per_region.py` (`create_buffered_region_split_indices`,
test ratio 0.15, buffer k = 10, one seed per region) and checks the stored test
labels (element by element) and every `split_info` field against the
re-created split. It exits with an error if any fold differs.

Usage:
    python experiment/run_check_exp5_splits.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import OUTPUT_PATH
from feature_engineering import create_buffered_region_split_indices
from run_exp5_proportion_per_region import _GRAPH_K, _TEST_RATIO

_RESULTS_PATH = "result/exp5_proportion_per_region.json"


def _same(a, b) -> bool:
    return bool(np.isclose(a, b)) if isinstance(b, float) else a == b


def check_exp5_splits(processed_df: pd.DataFrame) -> list:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    labels = processed_df["has_ree"].to_numpy(dtype=int)
    coords = processed_df[["Latitude", "Longitude"]].to_numpy()

    with open(_RESULTS_PATH, encoding="utf-8") as f:
        runs = json.load(f)["runs"]

    mismatches = []
    for run in runs:
        seed = run["seed"]
        for region in regions:
            stored = run["per_region"][region]
            region_idx = np.where(processed_df["Region"].values == region)[0]
            split = create_buffered_region_split_indices(
                region_idx, coords[region_idx], test_ratio=_TEST_RATIO,
                k_for_buffer=_GRAPH_K, random_seed=seed,
            )
            labels_match = labels[region_idx][split["test_local"]].tolist() == stored["y_true"]
            info_match = all(_same(split["split_info"][k], v) for k, v in stored["split_info"].items())
            if not (labels_match and info_match):
                mismatches.append((seed, region, labels_match, info_match))
        print(f"  seed {seed}: checked {len(regions)} folds")

    n_folds = len(runs) * len(regions)
    print(f"\n{n_folds - len(mismatches)}/{n_folds} folds reproduce the stored labels and split_info")
    return mismatches


def main():
    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    mismatches = check_exp5_splits(processed_df)
    if mismatches:
        raise SystemExit(f"mismatched folds: {mismatches}")


if __name__ == "__main__":
    main()
