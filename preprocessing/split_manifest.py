"""
preprocessing/split_manifest.py

A persisted, shared definition of the held-out splits, so that every model in an
experiment is evaluated on identical test sets by construction rather than by
coincidence.

Why this exists
---------------
`run_gnn_per_region.py` and `run_baseline_per_region.py` each chose their seeds with

    seeds = [42] + random.sample([s for s in range(61) if s != 42], N_RUNS - 1)

and neither called `random.seed`. The four non-default seeds were therefore drawn from
an unseeded global generator, independently in each process: the graph run recorded
[42, 60, 22, 38, 48] and the baseline run [42, 59, 29, 3, 9]. Since the spatial block
identity depends on the seed, the two scripts evaluated on different held-out blocks for
four of five runs, leaving exactly one shared seed and reducing the paired significance
test on Experiment 2 to n = 1.

Pinning the seed list is necessary but NOT sufficient, and the response letter should not
claim otherwise. Identical seeds only give identical splits while every script keeps
calling the same split function with the same arguments; that is true today (both use
`create_buffered_region_split_indices` with test_ratio=0.15 and k_for_buffer=10) but it
is an invariant no one is checking, and it silently breaks the moment one script changes
a constant. Persisting the resulting record identifiers and having every script read them
makes the agreement explicit and auditable.

What is stored
--------------
Positional indices into the processed dataframe, NOT `ID_No`. `ID_No` is not a unique key
in this dataset -- rows 689 and 3101 both carry ID_No 3352.0 (two East Asia occurrences
about 150 m apart), and one row has a missing ID_No -- so it cannot identify a record.
`ID_No` is stored alongside for human inspection only. A content hash of the processed
dataframe is recorded so a manifest built against a different data snapshot is rejected
rather than silently reused.

Usage
-----
    from split_manifest import get_region_splits

    splits = get_region_splits(processed_df, seeds=[42, 1, 2, 4, 7])
    train_g, val_g, test_g = splits[seed][region]
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import create_buffered_region_split_indices

_MANIFEST_PATH = "result/split_manifest_per_region.json"

# Must match the callers. Both run_gnn_per_region.py and run_baseline_per_region.py
# use these values; the manifest is what makes that agreement checkable.
TEST_RATIO = 0.15
K_FOR_BUFFER = 10
MIN_REGION_NODES = 20

# Pinned, explicit, and shared. Replaces the unseeded random.sample in both scripts.
DEFAULT_SEEDS = [42, 1, 2, 4, 7]


def _frame_fingerprint(processed_df: pd.DataFrame) -> str:
    """Cheap content hash: row count plus the coordinate/region columns that drive splits."""
    cols = ["Latitude", "Longitude", "Region"]
    payload = f"{len(processed_df)}|" + "|".join(
        processed_df[c].astype(str).str.cat(sep=",") for c in cols if c in processed_df
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _build(processed_df: pd.DataFrame, seeds: list[int]) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    out: dict = {}
    for seed in seeds:
        per_region: dict = {}
        for region in regions:
            region_global_idx = np.where(processed_df["Region"] == region)[0]
            if len(region_global_idx) < MIN_REGION_NODES:
                continue
            coords_region = processed_df[["Latitude", "Longitude"]].values[region_global_idx]
            split = create_buffered_region_split_indices(
                region_global_idx, coords_region, test_ratio=TEST_RATIO,
                k_for_buffer=K_FOR_BUFFER, random_seed=seed,
            )
            per_region[region] = {
                "train": region_global_idx[split["train_local"]].tolist(),
                "val": region_global_idx[split["val_local"]].tolist(),
                "test": region_global_idx[split["test_local"]].tolist(),
                "split_info": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in split["split_info"].items()},
            }
        out[str(seed)] = per_region
    return out


def get_region_splits(processed_df: pd.DataFrame,
                      seeds: list[int] | None = None,
                      path: str = _MANIFEST_PATH,
                      rebuild: bool = False) -> dict:
    """
    Return {seed(str): {region: {"train": [...], "val": [...], "test": [...]}}}.

    Builds and persists the manifest on first use; afterwards every caller reads the
    same file. Raises if an existing manifest was built from a different dataframe or
    does not cover the requested seeds, rather than silently regenerating.
    """
    seeds = list(seeds or DEFAULT_SEEDS)
    fp = _frame_fingerprint(processed_df)
    p = Path(path)

    if p.exists() and not rebuild:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("fingerprint") != fp:
            raise SystemExit(
                f"{path} was built from a different processed dataframe "
                f"(fingerprint {doc.get('fingerprint')} vs {fp}). Delete it and rerun "
                f"every model in this experiment, or pass rebuild=True -- do not mix "
                f"splits across data snapshots."
            )
        missing = [s for s in seeds if str(s) not in doc["splits"]]
        if missing:
            raise SystemExit(
                f"{path} does not contain seed(s) {missing}. Rebuild the manifest and "
                f"rerun all models so they share one definition."
            )
        return doc["splits"]

    splits = _build(processed_df, seeds)
    p.parent.mkdir(parents=True, exist_ok=True)
    id_lookup = (processed_df["ID_No"].tolist() if "ID_No" in processed_df else None)
    doc = {
        "description": (
            "Shared held-out splits for the per-region (Experiment 2) protocol. Values "
            "are POSITIONAL indices into the processed dataframe; ID_No is not unique in "
            "this dataset and cannot be used as a key. Every model in Experiment 2 reads "
            "this file so the comparison is like-for-like by construction."
        ),
        "fingerprint": fp,
        "n_records": int(len(processed_df)),
        "test_ratio": TEST_RATIO,
        "k_for_buffer": K_FOR_BUFFER,
        "min_region_nodes": MIN_REGION_NODES,
        "seeds": seeds,
        "id_no_by_position": id_lookup,
        "splits": splits,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"  [split manifest] built {len(seeds)} seed(s) x "
          f"{len(next(iter(splits.values())))} regions -> {path}")
    return splits


def assert_split_matches(splits: dict, seed: int, region: str,
                         train_g, val_g, test_g) -> None:
    """Defensive check for callers that still compute their own split."""
    ref = splits[str(seed)][region]
    for name, got, want in (("train", train_g, ref["train"]),
                            ("val", val_g, ref["val"]),
                            ("test", test_g, ref["test"])):
        if sorted(int(i) for i in got) != sorted(int(i) for i in want):
            raise AssertionError(
                f"split mismatch for seed={seed} region={region} ({name}): this script "
                f"produced {len(got)} indices that differ from the shared manifest. The "
                f"Experiment 2 comparison would not be like-for-like."
            )
