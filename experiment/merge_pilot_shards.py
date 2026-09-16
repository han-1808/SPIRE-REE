"""
experiment/merge_pilot_shards.py

Combines the per-region shards of run_spire_weighted_pilot.py into one verdict.

The pilot was sharded by region so the 12 paired folds could run concurrently;
each shard writes its own JSON. This merges them and applies the decision rule
stated in the pilot: the comparison is made on VALIDATION F1, with test
metrics reported but never used to choose. A Wilcoxon signed-rank test over
the paired folds is included when scipy is available, since a mean difference
of a thousandth over twelve folds needs a significance statement rather than a
bare number.

Usage:
    python experiment/merge_pilot_shards.py
    python experiment/merge_pilot_shards.py --shards result/pilot_shards --out result/spire_weighted_pilot.json
"""

import argparse
import glob
import json
import statistics as st
from pathlib import Path

_VARIANTS = ["spire", "spire_weighted"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="result/pilot_shards")
    ap.add_argument("--out", default="result/spire_weighted_pilot.json")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.shards}/*.json"))
    if not files:
        raise SystemExit(f"no shard files under {args.shards}")

    folds = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            folds.extend(json.load(fh).get("folds", []))
    if not folds:
        raise SystemExit("shards contained no folds")

    def col(variant, key):
        return [f[variant][key] for f in folds if variant in f]

    summary = {v: {k: st.fmean(col(v, k)) for k in ("val_f1", "f1", "roc_auc", "pr_auc")}
               for v in _VARIANTS}

    val_d = [a - b for a, b in zip(col("spire_weighted", "val_f1"), col("spire", "val_f1"))]
    test_d = [a - b for a, b in zip(col("spire_weighted", "f1"), col("spire", "f1"))]
    wins_val = sum(1 for d in val_d if d > 0)
    wins_test = sum(1 for d in test_d if d > 0)

    p_val = p_test = None
    try:
        from scipy.stats import wilcoxon
        if len(set(val_d)) > 1:
            p_val = float(wilcoxon(val_d).pvalue)
        if len(set(test_d)) > 1:
            p_test = float(wilcoxon(test_d).pvalue)
    except Exception:
        pass

    n = len(folds)
    print(f"\nWEIGHTED-SPIRE PILOT — {n} paired folds "
          f"({len(files)} shards, {len({f['region'] for f in folds})} regions)")
    print("=" * 68)
    print(f"{'variant':<18}{'val F1':>10}{'test F1':>10}{'ROC-AUC':>10}{'PR-AUC':>10}")
    for v in _VARIANTS:
        s = summary[v]
        print(f"{v:<18}{s['val_f1']:>10.3f}{s['f1']:>10.3f}{s['roc_auc']:>10.3f}{s['pr_auc']:>10.3f}")
    print(f"{'delta':<18}"
          f"{summary['spire_weighted']['val_f1'] - summary['spire']['val_f1']:>+10.3f}"
          f"{summary['spire_weighted']['f1'] - summary['spire']['f1']:>+10.3f}"
          f"{summary['spire_weighted']['roc_auc'] - summary['spire']['roc_auc']:>+10.3f}"
          f"{summary['spire_weighted']['pr_auc'] - summary['spire']['pr_auc']:>+10.3f}")
    print()
    print(f"  weighted wins on VALIDATION (selection criterion): {wins_val}/{n}"
          + (f"   Wilcoxon p={p_val:.3f}" if p_val is not None else ""))
    print(f"  weighted wins on TEST:                             {wins_test}/{n}"
          + (f"   Wilcoxon p={p_test:.3f}" if p_test is not None else ""))

    verdict = ("ADOPT — weighted variant wins on validation"
               if (p_val is not None and p_val < 0.05 and wins_val > n / 2)
               else "DO NOT ADOPT — no validation-level improvement; "
                    "edge weights do not help GraphSAGE mean aggregation on this graph")
    print(f"\n  VERDICT: {verdict}")

    payload = {"n_folds": n, "n_regions": len({f["region"] for f in folds}),
               "folds": folds, "summary": summary,
               "val_delta": summary["spire_weighted"]["val_f1"] - summary["spire"]["val_f1"],
               "test_delta": summary["spire_weighted"]["f1"] - summary["spire"]["f1"],
               "weighted_wins_val": wins_val, "weighted_wins_test": wins_test,
               "wilcoxon_p_val": p_val, "wilcoxon_p_test": p_test, "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
