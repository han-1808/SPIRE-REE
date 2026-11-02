"""
experiment/run_geology_rationale.py

Reviewer #1, second round, Remark 1: descriptive statistics behind the
geological rationale for a single cross-type model.

Needs no model. Reads the processed records, the raw compilation (for
`Dep_Note` and the sparse fields dropped during preprocessing) and the stored
Experiment 1 SPIRE predictions (`gnn_inductive.json`), and reports:

  - REE minerals by lithological group: for the positive records of each
    group, the most frequently listed REE mineral and the share of records
    listing monazite;
  - the `Dep_Type` composition of the carbonatite group (residual enrichment);
  - ion-adsorption records, identified by the keyword "adsorp" in `Dep_Type`
    or `Dep_Note`: record IDs, regions, group flags, positive rate, SPIRE
    Experiment 1 error rate by class and their share of China's errors;
  - region x group cell counts, multi-label and unflagged records;
  - fill rates of `Alteration`, `Assoc_Rock` and `Host_Age` in the raw data;
  - per-region deposit-type entropy and SPIRE F1 (copied from
    `deposit_type_entropy_vs_f1.json`).

Raw records are matched to processed records by `ID_No`; the raw file holds
one duplicated ID (3352) with identical rows, which is dropped.

Usage:
    python experiment/run_geology_rationale.py

Results saved to result/geology_rationale.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import numpy as np
import pandas as pd

from config import DATA_PATH, OUTPUT_PATH

_GNN_PATH = "result/gnn_inductive.json"
_ENTROPY_PATH = "result/deposit_type_entropy_vs_f1.json"
_OUT_PATH = "result/geology_rationale.json"

_ION_ADSORPTION_KEYWORD = "adsorp"
_SPARSE_FIELDS = ["Alteration", "Assoc_Rock", "Host_Age"]
_SMALL_CELL_THRESHOLDS = [30, 10]


def _minerals_by_group(df: pd.DataFrame, groups: list) -> dict:
    out = {}
    for group in groups:
        lists = df.loc[(df[group] == 1) & df["REE_Mins"].notna(), "REE_Mins"].str.lower()
        minerals = lists.str.split(r"[,;]").explode().str.strip()
        counts = minerals[minerals != ""].value_counts()
        top = counts.index[0]
        out[group] = {
            "n_positive_records": int(len(lists)),
            "top_mineral": top,
            "top_mineral_record_share": float(lists.str.contains(top, regex=False).mean()),
            "monazite_record_share": float(lists.str.contains("monazite", regex=False).mean()),
            "top3_mineral_mentions": {k: int(v) for k, v in counts.head(3).items()},
        }
    n_monazite_top = sum(v["top_mineral"] == "monazite" for v in out.values())
    print(f"  monazite is the most listed REE mineral in {n_monazite_top}/{len(groups)} groups")
    return out


def _spire_exp1_errors(y: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-record error rate of SPIRE over the Experiment 1 runs."""
    with open(_GNN_PATH, encoding="utf-8") as f:
        runs = json.load(f)["runs"]
    n_err = np.zeros(len(y))
    n_pred = np.zeros(len(y))
    for run in runs.values():
        for region_data in run["per_region"].values():
            idx = np.asarray(region_data["test_idx"])
            if not np.array_equal(np.asarray(region_data["y_true"]), y[idx]):
                raise SystemExit("Stored y_true does not match REE_Mins-derived labels")
            y_pred = np.asarray(region_data["predictions"]["spire"]["y_pred"])
            n_err[idx] += y_pred != y[idx]
            n_pred[idx] += 1
    return n_err / n_pred, len(runs)


def _ion_adsorption(df: pd.DataFrame, dep_note: pd.Series, groups: list, y: np.ndarray) -> dict:
    text = (df["Dep_Type"].fillna("") + " " + dep_note.fillna("")).str.lower()
    is_ia = text.str.contains(_ION_ADSORPTION_KEYWORD, regex=False).to_numpy()
    err, n_runs = _spire_exp1_errors(y)
    china = (df["Region"] == "China").to_numpy()

    result = {
        "rule": f"'{_ION_ADSORPTION_KEYWORD}' in Dep_Type or Dep_Note (case-insensitive)",
        "n_records": int(is_ia.sum()),
        "record_ids": [int(i) for i in df.loc[is_ia, "ID_No"]],
        "by_region": {k: int(v) for k, v in df.loc[is_ia, "Region"].value_counts().items()},
        "group_flags": {g: int(df.loc[is_ia, g].sum()) for g in groups},
        "positive_rate": float(y[is_ia].mean()),
        "spire_exp1": {
            "n_runs": n_runs,
            "error_rate": float(err[is_ia].mean()),
            "error_rate_other_records": float(err[~is_ia].mean()),
            "error_rate_negatives": float(err[is_ia & (y == 0)].mean()),
            "error_rate_positives": float(err[is_ia & (y == 1)].mean()),
            "share_of_china_errors": float(err[is_ia & china].sum() / err[china].sum()),
        },
    }
    s = result["spire_exp1"]
    print(f"  ion-adsorption: {result['n_records']} records, "
          f"error {s['error_rate']:.3f} vs {s['error_rate_other_records']:.3f}, "
          f"negatives {s['error_rate_negatives']:.3f}, "
          f"share of China errors {s['share_of_china_errors']:.3f}")
    return result


def _region_group_cells(df: pd.DataFrame, groups: list) -> dict:
    cells = df.groupby("Region")[groups].sum().astype(int)
    n_flags = df[groups].sum(axis=1)
    out = {
        "counts": {r: row.to_dict() for r, row in cells.iterrows()},
        "n_cells": int(cells.size),
        "n_multi_label_records": int((n_flags > 1).sum()),
        "n_unflagged_records": int((n_flags == 0).sum()),
    }
    for t in _SMALL_CELL_THRESHOLDS:
        out[f"n_cells_below_{t}"] = int((cells.to_numpy() < t).sum())
    print(f"  region x group: {out['n_cells_below_30']}/{out['n_cells']} cells < 30, "
          f"{out['n_cells_below_10']} < 10")
    return out


def run_geology_rationale() -> dict:
    df = pd.read_excel(OUTPUT_PATH)
    raw = pd.read_excel(DATA_PATH).drop_duplicates("ID_No").set_index("ID_No")
    groups = [c for c in df.columns if c.startswith("is_")]
    y = df["REE_Mins"].notna().astype(int).to_numpy()

    with open(_ENTROPY_PATH, encoding="utf-8") as f:
        entropy = json.load(f)

    carbonatite_types = df.loc[df["is_carbonatite_system"] == 1, "Dep_Type"].str.lower()

    results = {
        "experiment": "geology_rationale",
        "description": (
            "Descriptive statistics for the geological rationale (Reviewer 1, "
            "Remark 1). Labels: has_ree = REE_Mins documented. Errors are SPIRE "
            "Experiment 1 predictions averaged over runs per record."
        ),
        "n_records": int(len(df)),
        "n_positive": int(y.sum()),
        "minerals_by_group": _minerals_by_group(df, groups),
        "carbonatite_dep_type": {k: int(v) for k, v in carbonatite_types.value_counts().items()},
        "ion_adsorption": _ion_adsorption(df, df["ID_No"].map(raw["Dep_Note"]), groups, y),
        "region_group_cells": _region_group_cells(df, groups),
        "raw_fill_rates": {c: float(raw[c].notna().mean()) for c in _SPARSE_FIELDS},
        "entropy_vs_f1": {
            "source": _ENTROPY_PATH,
            "per_region": entropy["per_region"],
            **{k: entropy[k] for k in ["spearman_r", "spearman_p", "pearson_r", "pearson_p"]},
        },
    }

    Path(_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {_OUT_PATH}")
    return results


if __name__ == "__main__":
    run_geology_rationale()
