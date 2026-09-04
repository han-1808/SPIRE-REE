"""
experiment/run_documentation_completeness.py

Comment (2)-C: documentation-completeness / artifact probe.

Builds a documentation-completeness feature set per record from the RAW
database (data/Global_REE_occurrence_database.xlsx) -- not the processed
feature file, since these are administrative/free-text fields, not model
inputs -- joined onto the processed dataset via ID_No (same de-dupe
pattern as the one duplicated ID_No used elsewhere: raw has one ID
(3352.0) split across 2 rows with identical content).

Completeness signals used:
  - n_nonempty_fields : count of non-null among 8 free-text/administrative
    fields (Comments, Loc_Note, Ref_List, Rec_Note, Dep_Note, Stat_Note,
    Expl_Note, P_Note), 0-8.
  - comments_len, loc_note_len, ref_list_len : character length of each
    field (0 if null). Ref_List is populated for every record (0% null,
    checked directly), so its *presence* has no variance -- length is
    used instead, as the more informative completeness proxy for that field.

(2)-C.b -- documentation-artifact probe (run now, fully independent of
the GNN pipeline): trains a classifier using ONLY these completeness
features (no geological/mineral features at all) to predict `has_ree`,
cross-validated AUC. AUC ~0.5 would refute "the model rides on
documentation completeness rather than genuine REE signal"; a clearly
high AUC would support that concern and should be disclosed in
Limitations.

(2)-C.a -- split F1 by well-documented vs poorly-documented (needs
per-node TP/FP/FN/TN from Exp1, not yet saved -- same gap as Comment 7's
error decomposition): this script only builds and saves the
completeness score/bucket per record here; the actual F1-by-bucket split
is computed later, once Exp1 is instrumented to save per-node
predictions (shared with Comment 7-A, built together to avoid
duplicating the completeness-feature logic).

Usage:
    python experiment/run_documentation_completeness.py

Results: result/documentation_completeness.json (features + C.b AUC),
result/documentation_completeness_scores.json (per-record score/bucket,
for later reuse by (2)-C.a / Comment 7-A).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import OUTPUT_PATH

RAW_PATH = "data/Global_REE_occurrence_database.xlsx"
_TEXT_FIELDS = [
    "Comments", "Loc_Note", "Ref_List", "Rec_Note",
    "Dep_Note", "Stat_Note", "Expl_Note", "P_Note",
]
_LEN_FIELDS = ["Comments", "Loc_Note", "Ref_List"]


def _load_raw_text_fields() -> pd.DataFrame:
    raw = pd.read_excel(RAW_PATH, usecols=["ID_No"] + _TEXT_FIELDS)
    return raw.drop_duplicates(subset="ID_No", keep="first")


def build_completeness_features(processed_df: pd.DataFrame) -> pd.DataFrame:
    raw = _load_raw_text_fields()
    merged = processed_df[["ID_No", "has_ree"]].merge(raw, on="ID_No", how="left")
    if merged[_TEXT_FIELDS].isna().all(axis=1).any():
        n_bad = int(merged[_TEXT_FIELDS].isna().all(axis=1).sum())
        raise ValueError(
            f"{n_bad} processed row(s) had no matching raw record via ID_No "
            "(all 8 text fields NaN simultaneously) -- join is likely broken."
        )

    out = pd.DataFrame(index=merged.index)
    out["ID_No"] = merged["ID_No"]
    out["has_ree"] = merged["has_ree"]
    out["n_nonempty_fields"] = merged[_TEXT_FIELDS].notna().sum(axis=1)
    for field in _LEN_FIELDS:
        col = f"{field.lower()}_len"
        out[col] = merged[field].fillna("").astype(str).str.len()
    return out


_COMPLETENESS_FEATURE_COLS = ["n_nonempty_fields", "comments_len", "loc_note_len", "ref_list_len"]


def run_completeness_probe(features: pd.DataFrame, y: pd.Series | None = None, seed: int = 42) -> dict:
    """(2)-C.b: classifier using ONLY completeness features to predict a
    binary target (default: has_ree). `y` lets the same probe be reused
    against other has_* mineral columns as a specificity check (is the
    completeness->label correlation an REE-specific artifact, or a
    property of the compiled database that affects any mineral flag?)."""
    X = features[_COMPLETENESS_FEATURE_COLS].astype(float)
    if y is None:
        y = features["has_ree"].astype(int)
    else:
        y = y.astype(int)

    model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    return {
        "feature_cols": _COMPLETENESS_FEATURE_COLS,
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_auc_per_fold": [float(a) for a in aucs],
        "n": int(len(X)),
        "positive_rate": float(y.mean()),
    }


def run_specificity_check(features: pd.DataFrame, processed_df: pd.DataFrame,
                           seed: int = 42, n_minerals: int = 5) -> dict:
    """(2)-C.b follow-up: is the completeness->has_ree correlation specific
    to REE, or does documentation completeness predict the mention of ANY
    mineral about equally well (a general property of the compiled
    database, not an REE-specific artifact)? Runs the same probe against
    a handful of non-REE-diagnostic `has_*` columns for comparison."""
    has_cols = [c for c in processed_df.columns if c.startswith("has_") and c != "has_ree"]
    prevalence = processed_df[has_cols].mean().sort_values(ascending=False)
    # Skip REE-diagnostic minerals (already implicated in the REE label
    # itself, e.g. via Host_Lith inference) so the comparison set is
    # unambiguously non-REE.
    ree_diagnostic = {"has_zircon", "has_ilmenite", "has_pyrochlore", "has_fluorite", "has_magnetite"}
    candidates = [c for c in prevalence.index if c not in ree_diagnostic][:n_minerals]

    results: dict = {}
    for col in candidates:
        y = processed_df[col]
        if y.sum() < 20:
            continue
        probe = run_completeness_probe(features, y=y, seed=seed)
        results[col] = probe
    return results


def build_completeness_bucket(features: pd.DataFrame) -> pd.DataFrame:
    """(2)-C.a groundwork: well-documented vs poorly-documented, by median
    split on n_nonempty_fields (the most directly interpretable signal)."""
    median = features["n_nonempty_fields"].median()
    features = features.copy()
    features["completeness_bucket"] = np.where(
        features["n_nonempty_fields"] > median, "well_documented", "poorly_documented"
    )
    return features


def main():
    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    print("Building documentation-completeness features...")
    features = build_completeness_features(processed_df)
    print(features[["n_nonempty_fields", "comments_len", "loc_note_len", "ref_list_len"]].describe())

    print("\n[(2)-C.b] Documentation-artifact probe (completeness-only classifier)...")
    probe_result = run_completeness_probe(features)
    print(f"  CV AUC = {probe_result['cv_auc_mean']:.4f} ± {probe_result['cv_auc_std']:.4f}  "
          f"(n={probe_result['n']}, positive_rate={probe_result['positive_rate']:.3f})")

    print("\n[(2)-C.b specificity check] Is this REE-specific, or does completeness "
          "predict ANY mineral about equally well?...")
    specificity_result = run_specificity_check(features, processed_df)
    for col, info in specificity_result.items():
        print(f"  {col:20s} prevalence={info['positive_rate']:.3f}  "
              f"AUC={info['cv_auc_mean']:.3f} ± {info['cv_auc_std']:.3f}")

    Path("result").mkdir(exist_ok=True)
    with open("result/documentation_completeness.json", "w", encoding="utf-8") as f:
        json.dump({"c_b_probe": probe_result, "c_b_specificity_check": specificity_result}, f, indent=2)
    print("  Saved -> result/documentation_completeness.json")

    print("\n[(2)-C.a groundwork] Building well/poorly-documented bucket...")
    bucketed = build_completeness_bucket(features)
    print(bucketed["completeness_bucket"].value_counts())
    bucketed.to_json("result/documentation_completeness_scores.json", orient="records", indent=2)
    print("  Saved -> result/documentation_completeness_scores.json "
          "(per-record scores, for (2)-C.a / Comment 7-A once Exp1 per-node "
          "predictions are available)")


if __name__ == "__main__":
    main()
