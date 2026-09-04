"""
experiment/run_paired_significance_test.py

Comment (4)-E: paired comparison (t-test / Wilcoxon signed-rank) between
SPIRE and the strongest baseline (MLP) across runs (seeds).

Needs only the per-seed aggregate metric already saved in each result
file's `runs` list -- no new instrumentation required, unlike (4)-D. Uses
each experiment's own headline metric: F1 for Exp2/Exp3 (classification),
MAE for Exp5/Exp6 (region-level proportion prediction, lower is better --
doesn't change how the paired test itself works, only the sign convention
for interpreting `mean_diff`).

Exp2's SPIRE and MLP numbers live in two separate result files
(`gnn_per_region.json` / `baseline_per_region.json`) and are paired by
matching `seed`; Exp3/5/6 keep SPIRE and MLP in the same file/run entry
(combined baseline+GNN scripts), no merge needed.

n_pairs = number of seeds with both a SPIRE and an MLP value available
(typically 5, the paper's own "5-run average" seed count) -- small by
construction, so the paired test here should be read as indicative, not
as a substitute for the much larger sample (4)-D's pooled bootstrap CI
already gives.

Usage:
    python experiment/run_paired_significance_test.py

Results saved to result/paired_significance_test.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

_OUT_PATH = "result/paired_significance_test.json"

_EXP2_GNN_PATH = "result/gnn_per_region.json"
_EXP2_BASELINE_PATH = "result/baseline_per_region.json"
_EXP3_PATH = "result/inductive_sample.json"
_EXP5_PATH = "result/exp5_proportion_per_region.json"
_EXP6_PATH = "result/exp6_proportion_inductive_sample.json"


def _load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        print(f"  {path} not found -- skipped")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _extract_exp2() -> tuple[list, list, list, str] | None:
    gnn = _load(_EXP2_GNN_PATH)
    bl = _load(_EXP2_BASELINE_PATH)
    if gnn is None or bl is None:
        return None
    gnn_by_seed = {r["seed"]: (r.get("avg_metrics") or {}).get("spire", {}).get("f1")
                   for r in gnn.get("runs", [])}
    mlp_by_seed = {r["seed"]: (r.get("avg_metrics") or {}).get("mlp", {}).get("f1")
                   for r in bl.get("runs", [])}
    seeds = sorted(set(gnn_by_seed) & set(mlp_by_seed))
    spire_vals, mlp_vals, used_seeds = [], [], []
    for s in seeds:
        gv, mv = gnn_by_seed[s], mlp_by_seed[s]
        if gv is not None and mv is not None:
            used_seeds.append(s); spire_vals.append(gv); mlp_vals.append(mv)
    return used_seeds, spire_vals, mlp_vals, "f1"


def _extract_exp3() -> tuple[list, list, list, str] | None:
    payload = _load(_EXP3_PATH)
    if payload is None:
        return None
    seeds, spire_vals, mlp_vals = [], [], []
    for r in payload.get("runs", []):
        sv = (r.get("spire", {}).get("avg_metrics") or {}).get("f1")
        mv = ((r.get("baselines", {}).get("avg_metrics") or {}).get("mlp") or {}).get("f1")
        if sv is not None and mv is not None:
            seeds.append(r["seed"]); spire_vals.append(sv); mlp_vals.append(mv)
    return seeds, spire_vals, mlp_vals, "f1"


def _extract_exp5_or_6(path: str) -> tuple[list, list, list, str] | None:
    payload = _load(path)
    if payload is None:
        return None
    seeds, spire_vals, mlp_vals = [], [], []
    for r in payload.get("runs", []):
        agg = r.get("aggregated_metrics", {}) or {}
        s = agg.get("spire")
        m = agg.get("mlp")
        if s and m and s.get("mae") is not None and m.get("mae") is not None:
            seeds.append(r["seed"]); spire_vals.append(s["mae"]); mlp_vals.append(m["mae"])
    return seeds, spire_vals, mlp_vals, "mae"


_EXTRACTORS = {
    "exp2": _extract_exp2,
    "exp3": _extract_exp3,
    "exp5": lambda: _extract_exp5_or_6(_EXP5_PATH),
    "exp6": lambda: _extract_exp5_or_6(_EXP6_PATH),
}


def _paired_test(spire_vals: list, mlp_vals: list) -> dict:
    diffs = np.array(spire_vals) - np.array(mlp_vals)
    result = {
        "n_pairs": len(diffs),
        "spire_mean": float(np.mean(spire_vals)) if len(spire_vals) else None,
        "mlp_mean": float(np.mean(mlp_vals)) if len(mlp_vals) else None,
        "mean_diff_spire_minus_mlp": float(diffs.mean()) if len(diffs) else None,
    }
    if len(diffs) < 2:
        result["ttest"] = None
        result["wilcoxon"] = None
        result["note"] = "fewer than 2 pairs -- test not meaningful"
        return result

    if np.allclose(diffs, 0.0):
        # SPIRE == MLP on every seed -- the only genuinely degenerate case
        # (zero effect AND zero variance); a *constant non-zero* difference
        # is NOT degenerate -- it's the strongest possible paired effect and
        # scipy handles it correctly (huge t, tiny p), so it must NOT be
        # short-circuited here (a bug caught by testing this exact case:
        # ttest_rel([.90,.92,.89,.91,.93],[.80,.82,.79,.81,.83]) with a
        # constant +0.10 gap gives t=3.7e15, p=3.3e-62, correctly significant).
        result["ttest"] = None
        result["wilcoxon"] = None
        result["note"] = "SPIRE and MLP identical on every paired seed -- no difference to test"
        return result

    t_stat, t_p = ttest_rel(spire_vals, mlp_vals)
    result["ttest"] = {"statistic": float(t_stat), "p_value": float(t_p)}
    try:
        w_stat, w_p = wilcoxon(spire_vals, mlp_vals)
        result["wilcoxon"] = {"statistic": float(w_stat), "p_value": float(w_p)}
    except ValueError as e:
        # e.g. all-zero-rank ties, can happen with very small n
        result["wilcoxon"] = {"error": str(e)}
    return result


def run_paired_significance_test() -> dict:
    results: dict = {}
    for exp_name, extractor in _EXTRACTORS.items():
        print(f"[{exp_name}]")
        extracted = extractor()
        if extracted is None:
            continue
        seeds, spire_vals, mlp_vals, metric = extracted
        if not seeds:
            print(f"  no paired (spire, mlp) seeds found")
            continue
        test_result = _paired_test(spire_vals, mlp_vals)
        test_result["metric"] = metric
        test_result["seeds"] = seeds
        results[exp_name] = test_result
        print(f"  n_pairs={test_result['n_pairs']}  metric={metric}  "
              f"spire_mean={test_result['spire_mean']:.4f}  mlp_mean={test_result['mlp_mean']:.4f}")
        if test_result.get("ttest"):
            print(f"  paired t-test:  t={test_result['ttest']['statistic']:.3f}  "
                  f"p={test_result['ttest']['p_value']:.4f}")
        if test_result.get("wilcoxon") and "statistic" in test_result["wilcoxon"]:
            print(f"  Wilcoxon:       W={test_result['wilcoxon']['statistic']:.3f}  "
                  f"p={test_result['wilcoxon']['p_value']:.4f}")
        elif test_result.get("note"):
            print(f"  {test_result['note']}")

    Path("result").mkdir(exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {_OUT_PATH}")
    return results


if __name__ == "__main__":
    run_paired_significance_test()
