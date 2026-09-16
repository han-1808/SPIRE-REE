"""
experiment/run_completeness_stratified.py

Partial answer to: does the model learn geology, or documentation completeness?

The documentation probe shows that completeness meta-features alone predict `has_ree`
at AUC 0.753. The mineral/lithology ablations show the geological features are
predictive. Neither separates the two, because the mineral indicators are themselves
derived from the same descriptive fields whose completeness the probe exploits.

This script gives a partial separation that needs no retraining. The idea: hold
documentation completeness approximately fixed by stratifying the held-out predictions
into completeness buckets, and ask whether SPIRE still separates the classes *within* a
stratum. Inside a stratum a documentation-only model has little left to exploit, so
discrimination that survives stratification is not attributable to completeness alone.

Both models are held to the SAME protocol. The documentation-only probe is fitted on
exactly the records SPIRE trained on for each leave-one-region-out fold and predicts
exactly the records SPIRE was tested on, over unique processed records rather than
repeated prediction rows. An earlier version cross-validated the probe inside each
stratum, which differed from SPIRE in evaluation protocol as well as in features and
therefore confounded the two; matching the folds lowers the probe's AUC substantially,
because it must now generalise to an unseen region as SPIRE does.

Reported per stratum: SPIRE's ROC-AUC, the matched probe's ROC-AUC, and the class
prevalence (a near single-class stratum supports no conclusion).

What this does and does not establish. A clear gap indicates SPIRE uses information a
completeness-only model does not have, with completeness approximately held fixed. It
does NOT establish that the residual information is *geological* rather than some other
correlate of the descriptive text, and the four meta-features are not a complete
description of documentation practice, so residual documentation effects are constrained
rather than eliminated.

Usage:
    python experiment/run_completeness_stratified.py

Results saved to result/completeness_stratified.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from config import OUTPUT_PATH

_GNN_RESULTS = "result/gnn_inductive.json"
_COMPLETENESS = "result/documentation_completeness_scores.json"
_OUT = "result/completeness_stratified.json"
_META_COLS = ["n_nonempty_fields", "comments_len", "loc_note_len", "ref_list_len"]


def _spire_predictions_by_position(doc: dict) -> dict:
    """position -> list of (y_true, y_prob) over all seeds/regions."""
    out: dict = {}
    runs = doc.get("runs", [])
    runs = list(runs.values()) if isinstance(runs, dict) else runs
    for run in runs:
        for region_data in (run.get("per_region") or {}).values():
            idx = (region_data or {}).get("test_idx")
            # y_true is stored at region level; `predictions` holds only y_prob/y_pred.
            y_true = (region_data or {}).get("y_true")
            preds = (region_data or {}).get("predictions", {}).get("spire")
            if idx is None or y_true is None or not preds or preds.get("y_prob") is None:
                continue
            for pos, yt, yp in zip(idx, y_true, preds["y_prob"]):
                out.setdefault(int(pos), []).append((int(yt), float(yp)))
    return out


def _doc_probe_matched(doc: dict, comp: pd.DataFrame, processed: pd.DataFrame) -> dict:
    """Documentation-only probe evaluated on SPIRE's own leave-one-region-out folds.

    For every (seed, region) fold recorded in the Experiment 1 result file, the probe is
    fitted on exactly the records SPIRE trained on -- all regions except the held-out one
    -- and predicts exactly the records SPIRE was tested on. Fitting is over UNIQUE
    processed records, never over repeated prediction rows, so a record appearing under
    five seeds cannot leak between a probe's own train and test sets.

    Returns position -> list of (y_true, y_prob), the same shape as the SPIRE side.
    """
    X_all = comp[_META_COLS].to_numpy()
    y_all = comp["has_ree"].to_numpy().astype(int)
    region_of = processed["Region"].to_numpy()
    out: dict = {}
    runs = doc.get("runs", [])
    runs = list(runs.values()) if isinstance(runs, dict) else runs
    seen_folds = set()
    for run in runs:
        for region, region_data in (run.get("per_region") or {}).items():
            idx = (region_data or {}).get("test_idx")
            if idx is None:
                continue
            # The probe is deterministic given the fold, and the LORO fold is identical
            # across seeds, so fit once per region rather than once per (seed, region).
            if region in seen_folds:
                continue
            seen_folds.add(region)
            test_pos = np.asarray(idx, dtype=int)
            train_mask = region_of != region
            train_mask[test_pos] = False
            m = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
            m.fit(X_all[train_mask], y_all[train_mask])
            probs = m.predict_proba(X_all[test_pos])[:, 1]
            for pos, q in zip(test_pos, probs):
                out.setdefault(int(pos), []).append((int(y_all[pos]), float(q)))
    return out


def main() -> int:
    for f in (_GNN_RESULTS, _COMPLETENESS):
        if not Path(f).exists():
            print(f"{f} not found -- run the upstream script first")
            return 1

    processed = pd.read_excel(OUTPUT_PATH)
    comp = pd.read_json(_COMPLETENESS)
    assert len(comp) == len(processed), "completeness frame is not aligned with processed_df"

    with open(_GNN_RESULTS, encoding="utf-8") as f:
        preds_by_pos = _spire_predictions_by_position(json.load(f))
    if not preds_by_pos:
        print("no stored SPIRE per-node predictions in", _GNN_RESULTS)
        return 1

    with open(_GNN_RESULTS, encoding="utf-8") as f:
        doc_preds = _doc_probe_matched(json.load(f), comp, processed)
    print(f"documentation-only probe fitted on SPIRE's own folds: "
          f"{len(doc_preds)} records predicted\n")

    buckets = comp["completeness_bucket"].to_numpy()
    result = {"strata": {}, "n_positions_with_predictions": len(preds_by_pos),
              "probe_protocol": "matched leave-one-region-out folds, fitted on unique records"}

    print(f"SPIRE held-out predictions available for {len(preds_by_pos)} of "
          f"{len(processed)} records\n")
    print(f"{'stratum':<22}{'n_pred':>8}{'prev':>8}{'SPIRE AUC':>12}{'doc-only AUC':>14}{'gap':>9}")

    for bucket in sorted(set(buckets)):
        pos_in = [p for p in preds_by_pos if buckets[p] == bucket]
        yt, yp = [], []
        for p in pos_in:
            for t, q in preds_by_pos[p]:
                yt.append(t); yp.append(q)
        yt_a, yp_a = np.asarray(yt), np.asarray(yp)
        if len(np.unique(yt_a)) < 2:
            print(f"{bucket:<22}{len(yt_a):>8}   single-class -- skipped")
            continue
        spire_auc = float(roc_auc_score(yt_a, yp_a))

        # documentation-only probe on MATCHED outer splits (see _doc_probe_matched):
        # same leave-one-region-out folds, same training pools, same test records as
        # SPIRE, then stratified identically. A probe cross-validated inside the stratum
        # would differ from SPIRE in protocol as well as in features, and the gap would
        # confound the two.
        mask = np.array([b == bucket for b in buckets])
        yt_d, yp_d = [], []
        for p in pos_in:
            if p in doc_preds:
                for t, q in doc_preds[p]:
                    yt_d.append(t); yp_d.append(q)
        doc_auc = (float(roc_auc_score(yt_d, yp_d))
                   if len(yt_d) and len(np.unique(yt_d)) >= 2 else None)

        prev = float(yt_a.mean())
        gap = (spire_auc - doc_auc) if doc_auc is not None else float("nan")
        result["strata"][str(bucket)] = {
            "n_predictions": int(len(yt_a)), "n_records": int(mask.sum()),
            "prevalence": prev, "spire_roc_auc": spire_auc,
            "documentation_only_roc_auc": doc_auc, "gap": gap,
        }
        print(f"{bucket:<22}{len(yt_a):>8}{prev:>8.3f}{spire_auc:>12.3f}"
              f"{(doc_auc if doc_auc is not None else float('nan')):>14.3f}{gap:>+9.3f}")

    # ── finer strata ────────────────────────────────────────────────────────────
    # The two-bucket split controls for very little: the documentation-only probe
    # scores almost as well inside a bucket as it does overall, which means
    # completeness still varies widely within a bucket. Repeat the comparison on
    # quintiles of a continuous completeness score, so the control tightens. If
    # SPIRE's advantage survives as the probe's own AUC collapses toward 0.5, the
    # residual signal is not completeness.
    score = comp[_META_COLS].rank(pct=True).mean(axis=1).to_numpy()
    qs = np.quantile(score, np.linspace(0, 1, 6))
    qs[-1] += 1e-9
    print(f"\n{'quintile':<22}{'n_pred':>8}{'prev':>8}{'SPIRE AUC':>12}{'doc-only AUC':>14}{'gap':>9}")
    result["quintiles"] = {}
    for i in range(5):
        mask = (score >= qs[i]) & (score < qs[i + 1])
        pos_in = [p for p in preds_by_pos if mask[p]]
        yt, yp = [], []
        for p in pos_in:
            for t, q in preds_by_pos[p]:
                yt.append(t); yp.append(q)
        yt_a, yp_a = np.asarray(yt), np.asarray(yp)
        if len(yt_a) == 0 or len(np.unique(yt_a)) < 2:
            print(f"Q{i+1:<21}{len(yt_a):>8}   single-class -- skipped")
            continue
        spire_auc = float(roc_auc_score(yt_a, yp_a))
        yt_d, yp_d = [], []
        for p in pos_in:
            if p in doc_preds:
                for t, q in doc_preds[p]:
                    yt_d.append(t); yp_d.append(q)
        doc_auc = (float(roc_auc_score(yt_d, yp_d))
                   if len(yt_d) and len(np.unique(yt_d)) >= 2 else None)
        gap = (spire_auc - doc_auc) if doc_auc is not None else float("nan")
        result["quintiles"][f"Q{i+1}"] = {
            "n_predictions": int(len(yt_a)), "n_records": int(mask.sum()),
            "prevalence": float(yt_a.mean()), "spire_roc_auc": spire_auc,
            "documentation_only_roc_auc": doc_auc, "gap": gap,
        }
        print(f"Q{i+1:<21}{len(yt_a):>8}{float(yt_a.mean()):>8.3f}{spire_auc:>12.3f}"
              f"{(doc_auc if doc_auc is not None else float('nan')):>14.3f}{gap:>+9.3f}")

    Path(_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {_OUT}")
    print("\nA positive gap inside a stratum means SPIRE separates the classes using "
          "information a completeness-only model does not have, with completeness held "
          "approximately fixed. It does not identify that information as geological.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
