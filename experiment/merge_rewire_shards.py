"""
experiment/merge_rewire_shards.py

Recombines per-region shards of run_ablation_random_rewire.py into
result/ablation_random_rewire.json.

Why sharded: the control is 2 conditions x 10 regions x 5 seeds = 100 leave-one-region-out
folds, each a 36-point grid, and the unsharded script writes its JSON only at the end.

Pairing note. run_ablation_random_rewire seeds torch/numpy at the start of each
(seed, condition) block, so within a shard the real and rewired arms begin from the
same RNG state and are paired on initialisation as well as on split and node features.
Because each shard runs a single region, the RNG stream no longer depends on how many
regions preceded it, which makes every fold's initialisation a function of its own seed
alone. Both arms are sharded identically, so the real-vs-rewired comparison -- the only
thing this control claims -- stays like-for-like.

Usage:
    python experiment/merge_rewire_shards.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import glob
import json

import numpy as np

from run_ablation import _region_avg

_CONDITIONS = ["real_graph", "random_rewired"]
_EXPECTED_REGIONS = 10
_EXPECTED_SEEDS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="result/rewire_shards")
    ap.add_argument("--out", default="result/ablation_random_rewire.json")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.shards}/*.json"))
    if not files:
        raise SystemExit(f"no shard files under {args.shards}")

    merged: dict = {c: {} for c in _CONDITIONS}
    seeds_seen: set = set()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            payload = json.load(fh)
        for cond in _CONDITIONS:
            for seed, run in (payload.get("results", {}).get(cond) or {}).items():
                seeds_seen.add(seed)
                slot = merged[cond].setdefault(seed, {})
                for region, fold in (run.get("per_region") or {}).items():
                    if region in slot:
                        raise SystemExit(
                            f"duplicate fold {cond}/seed={seed}/{region} in {f} -- "
                            f"clear {args.shards} and rerun"
                        )
                    slot[region] = fold

    problems = []
    for cond in _CONDITIONS:
        if len(merged[cond]) != _EXPECTED_SEEDS:
            problems.append(f"{cond}: {len(merged[cond])}/{_EXPECTED_SEEDS} seeds")
        for seed, per_region in sorted(merged[cond].items()):
            if len(per_region) != _EXPECTED_REGIONS:
                problems.append(f"{cond}/seed={seed}: {len(per_region)}/{_EXPECTED_REGIONS} regions")
    if problems:
        head = "INCOMPLETE — " + "; ".join(problems[:8])
        if not args.allow_partial:
            raise SystemExit(head + "\n(pass --allow-partial to merge anyway)")
        print("WARNING: " + head)

    results: dict = {c: {} for c in _CONDITIONS}
    for cond in _CONDITIONS:
        for seed, per_region in merged[cond].items():
            results[cond][seed] = {
                "avg_metrics": _region_avg(per_region),
                "per_region": per_region,
            }

    seeds = sorted(seeds_seen, key=int)
    summary = {}
    for cond in _CONDITIONS:
        per_seed = [results[cond][s]["avg_metrics"] for s in seeds if s in results[cond]]
        keys = {k for m in per_seed for k in m}
        block = {}
        for k in sorted(keys):
            vals = [m[k] for m in per_seed if k in m]
            block[f"{k}_mean"] = float(np.mean(vals))
            block[f"{k}_std"] = float(np.std(vals))
        summary[cond] = block

    payload = {
        "experiment": "ablation_random_rewire",
        "description": (
            "Comment (17)-D: SPIRE (mode=full) on the real spatial/feature graph vs a "
            "degree-preserving randomly rewired graph (networkx double_edge_swap). Node "
            "features are held at REAL-graph values in both arms: _apply_rewire replaces "
            "edge_index/edge_weight after _build_graph_artifacts has computed x, so "
            "w_mean/w_max are identical across conditions and only the message-passing "
            "topology differs. The rewired arm therefore still receives real-graph "
            "information through its node features, which makes the measured gap a LOWER "
            "BOUND on the contribution of topology. Network initialisation is seeded, so "
            "the arms are paired on split, node features and initialisation. Merged from "
            "per-region shards; see merge_rewire_shards.py."
        ),
        "regions": sorted({r for c in _CONDITIONS for s in merged[c] for r in merged[c][s]}),
        "seeds": [int(s) for s in seeds],
        "sharded": True,
        "n_shards": len(files),
        "summary": summary,
        "results": results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nmerged {len(files)} shards -> {args.out}")
    print(f"{'condition':<16}{'F1':>18}{'ROC-AUC':>18}{'PR-AUC':>18}")
    for cond in _CONDITIONS:
        row = f"{cond:<16}"
        for k in ("f1", "roc_auc", "pr_auc"):
            row += f"{summary[cond][f'{k}_mean']:.4f} ± {summary[cond][f'{k}_std']:.4f}".rjust(18)
        print(row)

    print("\nper-seed paired differences (real - rewired):")
    for k in ("f1", "roc_auc", "pr_auc"):
        diffs = [results["real_graph"][s]["avg_metrics"][k]
                 - results["random_rewired"][s]["avg_metrics"][k] for s in seeds]
        wins = sum(1 for d in diffs if d > 0)
        line = f"  {k:<9} mean {np.mean(diffs):+.4f}  real ahead {wins}/{len(diffs)}"
        try:
            from scipy.stats import wilcoxon, ttest_rel
            if len(set(diffs)) > 1:
                line += f"  wilcoxon p={wilcoxon(diffs).pvalue:.4f}"
                a = [results["real_graph"][s]["avg_metrics"][k] for s in seeds]
                b = [results["random_rewired"][s]["avg_metrics"][k] for s in seeds]
                line += f"  t p={ttest_rel(a, b).pvalue:.4f}"
        except Exception:
            pass
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
