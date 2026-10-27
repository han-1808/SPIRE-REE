"""
experiment/run_buffer_counts_exp3_exp6.py

Reviewer #2, second round, Remark 5: post-buffer sample counts for
Experiments 3 and 6.

Experiments 2 and 5 already store each fold's `split_info` in their result
files; Experiments 3 and 6 do not. This script needs no model: it re-creates
the exact held-out splits of the reported Exp 3 / Exp 6 runs (seeds read from
the result files, regions in sorted order, one `rng` shared across the region
loop -- as in `run_inductive_sample.py` and
`run_exp6_proportion_inductive_sample.py`) and exports each fold's
`split_info` plus its test class counts.

Every re-created test set is checked against the `sample_idx` / `test_idx`
stored in the result file; the script aborts if any fold does not match.

Usage:
    python experiment/run_buffer_counts_exp3_exp6.py

Results saved to result/buffer_counts_exp3_exp6.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import OUTPUT_PATH
from run_inductive_sample import _make_sample_split

_RESULT_FILES = {
    "exp3": "result/inductive_sample.json",
    "exp6": "result/exp6_proportion_inductive_sample.json",
}
_OUT_PATH = "result/buffer_counts_exp3_exp6.json"


def _stored_test_idx(exp: str, run: dict, region: str) -> list:
    if exp == "exp3":
        return run["spire"]["per_region"][region]["sample_idx"]
    region_data = run["per_region"][region]
    return region_data.get("test_idx") or region_data.get("sample_idx")


def _to_builtin(value):
    return value.item() if hasattr(value, "item") else value


def run_buffer_counts(processed_df: pd.DataFrame) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    labels = processed_df["has_ree"].to_numpy()
    results: dict = {}

    for exp, path in _RESULT_FILES.items():
        with open(path, encoding="utf-8") as f:
            runs = json.load(f)["runs"]

        per_seed: dict = {}
        for run in runs:
            seed = run["seed"]
            rng = np.random.default_rng(seed)
            per_region: dict = {}
            for i, region in enumerate(regions):
                # n_sample is unused by the split (kept for call-site compatibility).
                split = _make_sample_split(processed_df, region, 0, rng, val_seed=seed + i)
                test_idx = split["test_idx"]
                if sorted(_stored_test_idx(exp, run, region)) != sorted(test_idx.tolist()):
                    raise SystemExit(f"[{exp}] seed {seed} {region}: re-created split "
                                     f"does not match the stored test set")
                y = labels[test_idx]
                per_region[region] = {
                    "n_test_positive": int(y.sum()),
                    "n_test_negative": int(len(y) - y.sum()),
                    **{k: _to_builtin(v) for k, v in split["split_info"].items()},
                }
            per_seed[str(seed)] = per_region
            print(f"  [{exp}] seed {seed}: {len(regions)} folds match the stored splits")

        results[exp] = {"source": path, "seeds": [run["seed"] for run in runs],
                        "per_seed_per_region": per_seed}

    return results


def main():
    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    results = run_buffer_counts(processed_df)

    Path("result").mkdir(exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {_OUT_PATH}")


if __name__ == "__main__":
    main()
