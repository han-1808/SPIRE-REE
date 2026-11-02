"""
experiment/run_pooled_vs_stratified.py

Reviewer #1, second round, Remark 1: does stratifying by lithological group
help? Compares a pooled tabular ensemble with group-specific ensembles under
Experiment 1 (leave-one-region-out).

This compares TABULAR ENSEMBLES, not a stratified SPIRE. Group-specific SPIRE
models would need one graph per group, and the per-group graphs would be too
small under leave-one-region-out splits.

For each seed and held-out region, the split, the full candidate feature set
and the XGBoost + CatBoost + Random Forest ensemble are those of SPIRE's prior
(`create_spatial_split_indices`, `prepare_xgboost_inputs(fit_idx=train_idx)`,
`_make_ensemble_models`, refit on the training split as in `run_gnn.py`):

  - pooled     : one ensemble fitted on all training records;
  - stratified : for each group, an ensemble fitted only on the training
                 records carrying that group's flag.

Both are scored on the same held-out records of that group (mean of the
three predicted probabilities, threshold 0.5). A (seed, region, group) fold
is skipped if the group has fewer than 30 training records, a single class
in training, or a single class in the held-out region; skipped folds are
reported. Per seed, predictions are pooled over the remaining regions and
F1, ROC-AUC and PR-AUC are computed; the five seeds are compared with a
two-sided Wilcoxon signed-rank test (stratified minus pooled). With five
pairs the smallest attainable p-value is 0.0625.

Usage:
    python experiment/run_pooled_vs_stratified.py
    python experiment/run_pooled_vs_stratified.py --seeds 42 1

Results saved to result/pooled_vs_stratified.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from config import OUTPUT_PATH
from feature_engineering import (
    _make_ensemble_models,
    create_spatial_split_indices,
    prepare_xgboost_inputs,
)

_OUT_PATH = "result/pooled_vs_stratified.json"
_SEEDS = [42, 1, 2, 4, 7]
_GROUPS = [
    "is_placer",
    "is_felsic_pegmatite",
    "is_carbonatite_system",
    "is_alkaline_intrusive",
    "is_sedimentary",
]
_MIN_TRAIN = 30
_METRICS = ["f1", "roc_auc", "pr_auc"]


def _ensemble_prob(X: pd.DataFrame, y: pd.Series, fit_idx, score_idx, seed: int) -> np.ndarray:
    models = _make_ensemble_models(random_state=seed)
    for m in models:
        m.fit(X.iloc[fit_idx], y.iloc[fit_idx])
    return np.mean([m.predict_proba(X.iloc[score_idx])[:, 1] for m in models], axis=0)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "f1": float(f1_score(y_true, (y_prob >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def _run_one_seed(df: pd.DataFrame, regions: list, seed: int) -> tuple[dict, list]:
    """Pooled per-group predictions of both ensembles for one seed."""
    flags = df[_GROUPS].to_numpy().astype(bool)
    collected = {g: {"y": [], "pooled": [], "stratified": []} for g in _GROUPS}
    skipped = []

    for region in regions:
        split = create_spatial_split_indices(df, test_region=region, val_ratio=0.2,
                                             random_seed=seed)
        train_idx, test_idx = split["train_idx"], split["test_idx"]
        _, X, y = prepare_xgboost_inputs(df, fit_idx=train_idx)
        y_arr = y.to_numpy()

        pooled_prob = _ensemble_prob(X, y, train_idx, test_idx, seed)

        for j, group in enumerate(_GROUPS):
            g_train = train_idx[flags[train_idx, j]]
            in_test = flags[test_idx, j]
            g_test = test_idx[in_test]
            reason = None
            if len(g_test) == 0:
                continue
            if len(g_train) < _MIN_TRAIN:
                reason = f"{len(g_train)} training records"
            elif len(np.unique(y_arr[g_train])) < 2:
                reason = "single class in training"
            elif len(np.unique(y_arr[g_test])) < 2:
                reason = "single class in held-out region"
            if reason:
                skipped.append({"seed": seed, "region": region, "group": group,
                                "n_test": int(len(g_test)), "reason": reason})
                continue

            collected[group]["y"].append(y_arr[g_test])
            collected[group]["pooled"].append(pooled_prob[in_test])
            collected[group]["stratified"].append(
                _ensemble_prob(X, y, g_train, g_test, seed))
        print(f"  seed {seed} {region}: done")

    per_group = {}
    for group, c in collected.items():
        y_true = np.concatenate(c["y"])
        per_group[group] = {
            "n_test_predictions": int(len(y_true)),
            "n_regions": len(c["y"]),
            "pooled": _metrics(y_true, np.concatenate(c["pooled"])),
            "stratified": _metrics(y_true, np.concatenate(c["stratified"])),
        }
    return per_group, skipped


def run_pooled_vs_stratified(seeds: list) -> dict:
    df = pd.read_excel(OUTPUT_PATH)
    regions = sorted(df["Region"].dropna().unique().tolist())

    per_seed, skipped = {}, []
    for seed in seeds:
        per_seed[str(seed)], seed_skipped = _run_one_seed(df, regions, seed)
        skipped.extend(seed_skipped)

    summary = {}
    for group in _GROUPS:
        rows = [per_seed[str(s)][group] for s in seeds]
        summary[group] = {"n_test_predictions_per_seed": [r["n_test_predictions"] for r in rows]}
        for m in _METRICS:
            pooled = np.array([r["pooled"][m] for r in rows])
            strat = np.array([r["stratified"][m] for r in rows])
            diff = strat - pooled
            entry = {
                "pooled_mean": float(pooled.mean()),
                "stratified_mean": float(strat.mean()),
                "mean_diff_stratified_minus_pooled": float(diff.mean()),
                "n_seeds_stratified_higher": int((diff > 0).sum()),
            }
            if len(seeds) > 1 and np.any(diff != 0):
                w_stat, w_p = wilcoxon(strat, pooled)
                entry.update({"wilcoxon_statistic": float(w_stat), "wilcoxon_p": float(w_p)})
            summary[group][m] = entry

    print(f"\n{'group':<24}" + "".join(f"{m:>30}" for m in _METRICS))
    for group, s in summary.items():
        print(f"{group:<24}" + "".join(
            f"  {s[m]['pooled_mean']:.3f}->{s[m]['stratified_mean']:.3f} "
            f"({s[m]['n_seeds_stratified_higher']}/{len(seeds)})"
            f" p={s[m].get('wilcoxon_p', float('nan')):.3f}" for m in _METRICS))
    print(f"\nSkipped folds: {len(skipped)}")

    results = {
        "experiment": "pooled_vs_stratified",
        "description": (
            "Experiment 1 (leave-one-region-out). Tabular XGBoost+CatBoost+RF "
            "ensemble (SPIRE's prior configuration) fitted on all training records "
            "(pooled) or only on the training records of one lithological group "
            "(stratified), scored on the same held-out records of that group. Not a "
            "stratified SPIRE."
        ),
        "seeds": seeds,
        "groups": _GROUPS,
        "min_train_records": _MIN_TRAIN,
        "summary": summary,
        "per_seed": per_seed,
        "skipped_folds": skipped,
    }
    Path(_OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {_OUT_PATH}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=_SEEDS)
    args = parser.parse_args()
    run_pooled_vs_stratified(args.seeds)


if __name__ == "__main__":
    main()
