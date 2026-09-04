"""
experiment/run_region_error_decomposition.py

Comment 7-A: region-specific error decomposition.

For each region, gets SPIRE's per-node TP/FP/FN/TN on Exp 1
(leave-one-region-out) from the per-node predictions `run_gnn.py` now
saves (`y_true`/`test_idx` per region, `y_prob`/`y_pred` per model under
`predictions`), then breaks FN/FP down by:
  (i)   deposit-type (`is_group`, Comment 1) -- read directly from the
        processed dataset via `test_idx`.
  (ii)  documentation completeness -- reuses `result/documentation_completeness_scores.json`
        (Comment 2-C.a's bucket), joined via `ID_No` (avoids duplicating
        that logic).
  (iii) mean k-NN distance of the misclassified nodes specifically --
        a fresh haversine k-NN (k=15, matching the graph's own `_GRAPH_K`)
        computed within each region's own node set (not the full training
        graph, which excludes the region entirely under leave-one-region-out).

Requires `result/gnn_inductive.json` to have been (re)generated with the
Comment 7-A per-node instrumentation added to `run_gnn.py` -- an older
result file (missing `test_idx`/`predictions`) is handled gracefully: the
region is skipped and reported, not a crash.

Usage:
    python experiment/run_region_error_decomposition.py

Results saved to result/region_error_decomposition.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from config import OUTPUT_PATH

_GNN_RESULTS_PATH = "result/gnn_inductive.json"
_COMPLETENESS_SCORES_PATH = "result/documentation_completeness_scores.json"
_OUT_PATH = "result/region_error_decomposition.json"
_GRAPH_K = 15
_EARTH_RADIUS_KM = 6371.0
_IS_GROUP_COLS = [
    "is_carbonatite_system", "is_alkaline_intrusive", "is_mafic_ultramafic",
    "is_felsic_pegmatite", "is_metamorphic_metasomatic", "is_sedimentary",
    "is_placer", "is_supergene_lateritic", "is_unclassified",
]


def _mean_knn_distance_km(coords: np.ndarray, k: int = _GRAPH_K) -> np.ndarray:
    """Mean haversine distance (km) to each node's k nearest neighbours
    within `coords` (excluding itself)."""
    n_neighbors = min(k + 1, len(coords))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="haversine")
    nbrs.fit(np.radians(coords))
    dist_rad, _ = nbrs.kneighbors(np.radians(coords))
    dist_km = dist_rad[:, 1:] * _EARTH_RADIUS_KM  # drop self (col 0)
    return dist_km.mean(axis=1)


def _confusion_label(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def decompose_one_seed_run(processed_df: pd.DataFrame, run: dict,
                            completeness: pd.DataFrame, model: str = "spire") -> dict:
    per_region = run.get("per_region", {})
    result: dict = {}
    n_skipped = 0

    for region, region_data in per_region.items():
        test_idx = region_data.get("test_idx")
        y_true = region_data.get("y_true")
        predictions = region_data.get("predictions", {})
        if test_idx is None or y_true is None or model not in predictions:
            n_skipped += 1
            continue
        y_pred = predictions[model]["y_pred"]

        region_rows = processed_df.iloc[test_idx].copy()
        region_rows["y_true"] = y_true
        region_rows["y_pred"] = y_pred
        region_rows["confusion"] = [
            _confusion_label(t, p) for t, p in zip(y_true, y_pred)
        ]

        coords = region_rows[["Latitude", "Longitude"]].to_numpy()
        region_rows["mean_knn_dist_km"] = _mean_knn_distance_km(coords)

        region_rows = region_rows.merge(
            completeness[["ID_No", "completeness_bucket"]], on="ID_No", how="left"
        )

        misclassified = region_rows[region_rows["confusion"].isin(["FN", "FP"])]

        result[region] = {
            "n_test": len(region_rows),
            "confusion_counts": region_rows["confusion"].value_counts().to_dict(),
            "misclassified": {
                "n": len(misclassified),
                "by_deposit_type": {
                    g: int(misclassified[g].sum()) for g in _IS_GROUP_COLS
                },
                "by_completeness_bucket": misclassified["completeness_bucket"]
                    .value_counts().to_dict(),
                "mean_knn_dist_km": float(misclassified["mean_knn_dist_km"].mean())
                    if len(misclassified) else None,
            },
            "correctly_classified": {
                "mean_knn_dist_km": float(
                    region_rows[~region_rows["confusion"].isin(["FN", "FP"])]["mean_knn_dist_km"].mean()
                ) if len(region_rows) > len(misclassified) else None,
            },
        }

    if n_skipped:
        print(f"  {n_skipped} region(s) skipped (missing per-node data -- "
              f"result file predates the Comment 7-A instrumentation)")
    return result


def run_error_decomposition(model: str = "spire") -> dict:
    gnn_path = Path(_GNN_RESULTS_PATH)
    completeness_path = Path(_COMPLETENESS_SCORES_PATH)
    if not gnn_path.exists():
        print(f"{_GNN_RESULTS_PATH} not found -- run run_gnn.py first")
        return {}
    if not completeness_path.exists():
        print(f"{_COMPLETENESS_SCORES_PATH} not found -- run "
              f"run_documentation_completeness.py first")
        return {}

    print("Loading processed data + Exp1 results + completeness scores...")
    processed_df = pd.read_excel(OUTPUT_PATH)
    with open(gnn_path, encoding="utf-8") as f:
        gnn_payload = json.load(f)
    completeness = pd.read_json(completeness_path)

    results: dict = {}
    for seed, run in gnn_payload.get("runs", {}).items():
        print(f"  seed={seed}...")
        results[seed] = decompose_one_seed_run(processed_df, run, completeness, model=model)

    Path("result").mkdir(exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": model, "runs": results}, f, indent=2)
    print(f"Saved → {_OUT_PATH}")
    return results


if __name__ == "__main__":
    run_error_decomposition()
