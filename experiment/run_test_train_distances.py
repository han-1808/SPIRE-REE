"""
experiment/run_test_train_distances.py

Reviewer #1, Remark 4 (first round): realised test-to-training distances for
the buffered spatial-block experiments (2, 3, 5 and 6).

This script needs no model. For every reported fold it obtains the held-out
test set and the retained training/validation pool, then computes the
haversine distance from each test record to its nearest train/val record:

  - Exp 2: positional indices read from `split_manifest_per_region.json`.
  - Exp 5: folds re-created from the seeds with
    `create_buffered_region_split_indices` (as in run_check_exp5_splits.py);
    the stored test labels and `split_info` are checked fold by fold.
  - Exp 3 / Exp 6: folds re-created with `_make_sample_split` (as in
    run_buffer_counts_exp3_exp6.py); the stored test indices are checked.

The script aborts if any re-created fold differs from the stored one.
Train and validation records together form the training graph, so the
reported minimum is the closest any test record comes to a graph node.

Usage:
    python experiment/run_test_train_distances.py

Results saved to result/test_train_distances.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from config import OUTPUT_PATH
from feature_engineering import _EARTH_RADIUS_KM, create_buffered_region_split_indices
from run_buffer_counts_exp3_exp6 import _stored_test_idx
from run_check_exp5_splits import _same
from run_exp5_proportion_per_region import _GRAPH_K as _EXP5_GRAPH_K
from run_exp5_proportion_per_region import _TEST_RATIO as _EXP5_TEST_RATIO
from run_inductive_sample import _make_sample_split

_MANIFEST_PATH = "result/split_manifest_per_region.json"
_EXP5_PATH = "result/exp5_proportion_per_region.json"
_EXP2_PATH = "result/gnn_per_region.json"
_SAMPLE_FILES = {
    "exp3": "result/inductive_sample.json",
    "exp6": "result/exp6_proportion_inductive_sample.json",
}
_OUT_PATH = "result/test_train_distances.json"


def _nearest_distances_km(coords_test: np.ndarray, coords_pool: np.ndarray) -> np.ndarray:
    """Haversine distance (km) from each test record to its nearest pool record."""
    nbrs = NearestNeighbors(n_neighbors=1, metric="haversine")
    nbrs.fit(np.radians(coords_pool))
    dist_rad, _ = nbrs.kneighbors(np.radians(coords_test))
    return dist_rad[:, 0] * _EARTH_RADIUS_KM


def _fold_record(coords: np.ndarray, test_idx, pool_idx, split_info: dict) -> dict:
    dist = _nearest_distances_km(coords[np.asarray(test_idx)], coords[np.asarray(pool_idx)])
    min_km = float(dist.min())
    buffer_km = float(split_info["buffer_km"])
    return {
        "n_test": int(len(test_idx)),
        "n_train_val": int(len(pool_idx)),
        "buffer_km": buffer_km,
        "buffer_fallback": bool(split_info.get("buffer_fallback", False)),
        "min_test_to_train_km": min_km,
        "median_test_to_train_km": float(np.median(dist)),
        "min_exceeds_buffer": bool(min_km > buffer_km),
    }


def _exp2_folds(coords: np.ndarray) -> dict:
    with open(_MANIFEST_PATH, encoding="utf-8") as f:
        splits = json.load(f)["splits"]
    with open(_EXP2_PATH, encoding="utf-8") as f:
        runs = {str(r["seed"]): r for r in json.load(f)["runs"]}

    per_seed: dict = {}
    for seed, regions in splits.items():
        per_seed[seed] = {}
        for region, split in regions.items():
            info = runs[seed]["per_region"][region]["split_info"]
            per_seed[seed][region] = _fold_record(
                coords, split["test"], split["train"] + split["val"], info,
            )
    return per_seed


def _exp5_folds(processed_df: pd.DataFrame, coords: np.ndarray) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    labels = processed_df["has_ree"].to_numpy(dtype=int)
    with open(_EXP5_PATH, encoding="utf-8") as f:
        runs = json.load(f)["runs"]

    per_seed: dict = {}
    for run in runs:
        seed = run["seed"]
        per_seed[str(seed)] = {}
        for region in regions:
            stored = run["per_region"][region]
            region_idx = np.where(processed_df["Region"].values == region)[0]
            split = create_buffered_region_split_indices(
                region_idx, coords[region_idx], test_ratio=_EXP5_TEST_RATIO,
                k_for_buffer=_EXP5_GRAPH_K, random_seed=seed,
            )
            labels_match = labels[region_idx][split["test_local"]].tolist() == stored["y_true"]
            info_match = all(_same(split["split_info"][k], v) for k, v in stored["split_info"].items())
            if not (labels_match and info_match):
                raise SystemExit(f"[exp5] seed {seed} {region}: re-created split "
                                 f"does not match the stored fold")
            pool = np.concatenate([split["train_local"], split["val_local"]])
            per_seed[str(seed)][region] = _fold_record(
                coords, region_idx[split["test_local"]], region_idx[pool], split["split_info"],
            )
    return per_seed


def _sample_folds(processed_df: pd.DataFrame, coords: np.ndarray, exp: str, path: str) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    with open(path, encoding="utf-8") as f:
        runs = json.load(f)["runs"]

    per_seed: dict = {}
    for run in runs:
        seed = run["seed"]
        rng = np.random.default_rng(seed)
        per_seed[str(seed)] = {}
        for i, region in enumerate(regions):
            # n_sample is unused by the split (kept for call-site compatibility).
            split = _make_sample_split(processed_df, region, 0, rng, val_seed=seed + i)
            if sorted(_stored_test_idx(exp, run, region)) != sorted(split["test_idx"].tolist()):
                raise SystemExit(f"[{exp}] seed {seed} {region}: re-created split "
                                 f"does not match the stored test set")
            pool = np.concatenate([split["train_idx"], split["val_idx"]])
            per_seed[str(seed)][region] = _fold_record(
                coords, split["test_idx"], pool, split["split_info"],
            )
    return per_seed


def _summarise(per_seed: dict) -> dict:
    folds = [f for regions in per_seed.values() for f in regions.values()]
    mins = np.array([f["min_test_to_train_km"] for f in folds])
    ratios = mins / np.array([f["buffer_km"] for f in folds])
    return {
        "n_folds": len(folds),
        "min_test_to_train_km": {"min": float(mins.min()), "median": float(np.median(mins)),
                                 "max": float(mins.max())},
        "min_over_buffer_ratio": {"min": float(ratios.min()), "median": float(np.median(ratios))},
        "n_folds_min_exceeds_buffer": int(sum(f["min_exceeds_buffer"] for f in folds)),
        "n_folds_buffer_fallback": int(sum(f["buffer_fallback"] for f in folds)),
    }


def main():
    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)
    coords = processed_df[["Latitude", "Longitude"]].to_numpy()

    folds = {
        "exp2": (_MANIFEST_PATH, _exp2_folds(coords)),
        "exp5": (_EXP5_PATH, _exp5_folds(processed_df, coords)),
        **{exp: (path, _sample_folds(processed_df, coords, exp, path))
           for exp, path in _SAMPLE_FILES.items()},
    }

    results = {}
    for exp, (source, per_seed) in folds.items():
        summary = _summarise(per_seed)
        results[exp] = {"source": source, "summary": summary, "per_seed_per_region": per_seed}
        d = summary["min_test_to_train_km"]
        print(f"  [{exp}] {summary['n_folds']} folds: min test-to-train "
              f"{d['min']:.1f}-{d['max']:.1f} km (median {d['median']:.1f}); "
              f"min > buffer in {summary['n_folds_min_exceeds_buffer']}/{summary['n_folds']}; "
              f"fallback {summary['n_folds_buffer_fallback']}")

    Path("result").mkdir(exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {_OUT_PATH}")


if __name__ == "__main__":
    main()
