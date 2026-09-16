"""
experiment/merge_graph_weighting_shards.py

Recombines the shards of run_ablation_graph_weighting.py into the single
result/ablation_graph_weighting.json that the response letter and figures read.

Why shard at all: the sweep is 4 conditions x 10 held-out regions x 5 seeds =
200 leave-one-region-out folds, each a 36-point hyperparameter grid. Run as one
process it takes ~9h on one core and writes its JSON only at the very end, so a
crash at hour 8 loses everything. Sharded by (condition, region) it is 40
independent processes that each write their own file.

Aggregation is done with run_ablation's own `_region_avg` / `_avg_metrics` /
`_std_metrics` rather than reimplemented here, so a merged file is arithmetically
identical to what the monolithic run would have produced:
`_region_avg` is an unweighted mean over the per-region dict, so splitting the
region loop across processes and reuniting the dicts changes nothing.

One documented difference from the monolithic run: it seeded once per seed and
then looped regions, so each region inherited the RNG state left by the previous
one. A shard seeds fresh and runs its single region, making every fold's stream
depend only on its own seed and not on region ordering. Values therefore differ
from an unsharded run at the usual seed-to-seed magnitude; all four conditions
are sharded identically, so the comparison between them -- the only thing this
experiment claims -- stays like-for-like.

Usage:
    python experiment/merge_graph_weighting_shards.py
    python experiment/merge_graph_weighting_shards.py --shards result/gw_shards
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import glob
import json

from run_ablation import _INPUT_DIMS, _avg_metrics, _region_avg, _std_metrics
from run_ablation_graph_weighting import _CONDITIONS, _MODE

_EXPECTED_SEEDS = 5
_EXPECTED_REGIONS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="result/gw_shards")
    ap.add_argument("--out", default="result/ablation_graph_weighting.json")
    ap.add_argument("--allow-partial", action="store_true",
                    help="merge even if some (condition, region) shards are missing")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.shards}/*.json"))
    if not files:
        raise SystemExit(f"no shard files under {args.shards}")

    # merged[cond][seed] = {region: fold}
    merged: dict = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            payload = json.load(fh)
        for cond, block in (payload.get("results") or {}).items():
            for seed, run in (block.get("runs") or {}).items():
                slot = merged.setdefault(cond, {}).setdefault(seed, {})
                for region, fold in (run.get("per_region") or {}).items():
                    if region in slot:
                        raise SystemExit(
                            f"duplicate fold {cond}/seed={seed}/{region} -- "
                            f"two shards cover the same cell ({f}). Clear "
                            f"{args.shards} and rerun rather than merging both."
                        )
                    slot[region] = fold

    # Coverage check before anything is written: a silently partial merge would
    # average over fewer regions and look like a real effect.
    problems = []
    for cond in _CONDITIONS:
        if cond not in merged:
            problems.append(f"condition {cond}: entirely missing")
            continue
        if len(merged[cond]) != _EXPECTED_SEEDS:
            problems.append(f"{cond}: {len(merged[cond])}/{_EXPECTED_SEEDS} seeds")
        for seed, per_region in sorted(merged[cond].items()):
            if len(per_region) != _EXPECTED_REGIONS:
                problems.append(
                    f"{cond}/seed={seed}: {len(per_region)}/{_EXPECTED_REGIONS} regions"
                )
    if problems:
        head = "INCOMPLETE — " + "; ".join(problems[:8])
        if not args.allow_partial:
            raise SystemExit(head + "\n(pass --allow-partial to merge anyway)")
        print("WARNING: " + head)

    results: dict = {}
    for cond, per_seed in merged.items():
        runs = {}
        for seed, per_region in per_seed.items():
            runs[seed] = {"avg_metrics": _region_avg(per_region),
                          "per_region": per_region}
        seed_avgs = [r["avg_metrics"] for r in runs.values() if r["avg_metrics"]]
        results[cond] = {
            "runs": runs,
            "summary": {
                "input_dim": _INPUT_DIMS[_MODE],
                "n_seeds": len(seed_avgs),
                "avg_metrics": _avg_metrics(seed_avgs),
                "std_metrics": _std_metrics(seed_avgs),
            },
        }

    payload = {
        "experiment": "ablation_graph_weighting",
        "description": (
            "Comment 13-B: graph edge-weighting comparison on Exp 1 "
            "(leave-one-region-out inductive), mode=no_edge_stats (11-dim). "
            "spatial_only=haversine distance only; geo_only=cosine similarity on "
            "the geo-only vector (Comment 13-A fix); mixed_current=alpha=0.5 "
            "combination with cosine similarity on the FULL node vector "
            "(pre-fix double-counting, reproduced only for this comparison); "
            "mixed_fixed=alpha=0.5 combination with the Comment 13-A fix applied "
            "(the default used everywhere else in the codebase). Trained on GCN, "
            "not SPIRE: SAGEConv/GATConv do not consume edge_weight in message "
            "passing (only GCNConv does). Merged from per-(condition, region) "
            "shards; see merge_graph_weighting_shards.py."
        ),
        "model": "gcn",
        "conditions": dict(_CONDITIONS),
        "seeds": sorted({int(s) for c in merged for s in merged[c]}),
        "regions": sorted({r for c in merged for s in merged[c] for r in merged[c][s]}),
        "sharded": True,
        "n_shards": len(files),
        "results": results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nmerged {len(files)} shards -> {args.out}")
    print("\n── Comment 13-B summary (avg ± std across seeds) ──────────")
    print(f"{'Condition':<16}{'F1':>16}{'ROC-AUC':>16}{'PR-AUC':>16}")
    for cond in _CONDITIONS:
        if cond not in results:
            continue
        a, sd = results[cond]["summary"]["avg_metrics"], results[cond]["summary"]["std_metrics"]
        row = f"{cond:<16}"
        for k in ("f1", "roc_auc", "pr_auc"):
            row += f"{a.get(k, float('nan')):.4f} ± {sd.get(k, 0.0):.4f}".rjust(16)
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
