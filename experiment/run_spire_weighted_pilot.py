"""
experiment/run_spire_weighted_pilot.py

Paired pilot: does making SPIRE edge-weight aware actually help?

SPIRE's `SAGEConv` layers take only (x, edge_index), so every edge weight this
pipeline computes -- the 0.5/0.5 haversine x cosine-similarity blend -- is
discarded before it reaches the encoder. `GraphSAGEWeighted` swaps `SAGEConv`
for `GraphConv(aggr="mean")`, which is the same functional form with the
weights restored (identical parameter count), so any difference is attributable
to the weighting and not to capacity.

Design:
  - Both variants share ONE graph/split per region, so the comparison is
    paired: same nodes, same edges, same ensemble scores, same test set.
  - Each variant is tuned INDEPENDENTLY on validation F1 over the same grid.
    run_gnn.py tunes only its first model and makes the rest inherit that
    config (Reviewer #2 Remark #9); reusing SPIRE's config here would bias the
    comparison toward whichever variant went first.
  - Model choice is made on validation. Test metrics are reported but never
    used for selection.

This is a pilot over a subset of regions/seeds, meant only to decide whether a
full six-experiment rerun is justified.

Usage:
    python experiment/run_spire_weighted_pilot.py
    python experiment/run_spire_weighted_pilot.py --seeds 42 1 --regions Africa Oceania

Results saved to result/spire_weighted_pilot.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import statistics as st

import pandas as pd

from config import OUTPUT_PATH
from gnn_training import GraphSAGE, GraphSAGEWeighted
from run_gnn import (
    _build_graph_artifacts,
    _compute_metrics,
    _predict_test_nodes,
    _train_model_on_artifacts,
)

_VARIANTS = [("spire", GraphSAGE), ("spire_weighted", GraphSAGEWeighted)]
_RESULTS_PATH = "result/spire_weighted_pilot.json"
_DEFAULT_REGIONS = ["Africa", "China", "Europe", "Oceania", "South America"]
_DEFAULT_SEEDS = [42, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", default=_DEFAULT_REGIONS)
    parser.add_argument("--out", default=_RESULTS_PATH,
                        help="output path; set per-shard when running regions in parallel")
    args = parser.parse_args()

    processed_df = pd.read_excel(OUTPUT_PATH)
    rows = []

    for seed in args.seeds:
        import torch
        torch.manual_seed(seed)
        for region in args.regions:
            print(f"\n{'='*64}\n  {region}  (seed={seed})\n{'='*64}", flush=True)
            artifacts = _build_graph_artifacts(processed_df, region, random_seed=seed)
            y_true = artifacts["y_test"].to_numpy()

            row = {"seed": seed, "region": region, "n_test": int(len(artifacts["test_idx"]))}
            for name, model_class in _VARIANTS:
                # best_config=None -> tune this variant on its own
                trained, config, val_f1 = _train_model_on_artifacts(
                    model_class, artifacts, None
                )
                y_prob, y_pred = _predict_test_nodes(trained, processed_df)
                m = _compute_metrics(y_true, y_pred, y_prob)
                row[name] = {"val_f1": val_f1, "config": list(config), **m}
                print(f"  [{name}] val_f1={val_f1:.3f}  test_f1={m['f1']:.3f} "
                      f"roc={m['roc_auc']:.3f} pr={m['pr_auc']:.3f}", flush=True)
            rows.append(row)

    def mean(name, key):
        return st.fmean([r[name][key] for r in rows])

    summary = {}
    for name, _ in _VARIANTS:
        summary[name] = {k: mean(name, k) for k in ("val_f1", "f1", "roc_auc", "pr_auc")}
    wins = sum(1 for r in rows if r["spire_weighted"]["f1"] > r["spire"]["f1"])

    print(f"\n{'='*64}\nPILOT SUMMARY over {len(rows)} region-seed folds\n{'='*64}")
    print(f"{'variant':<18}{'val F1':>9}{'test F1':>9}{'ROC-AUC':>9}{'PR-AUC':>9}")
    for name, _ in _VARIANTS:
        s = summary[name]
        print(f"{name:<18}{s['val_f1']:>9.3f}{s['f1']:>9.3f}{s['roc_auc']:>9.3f}{s['pr_auc']:>9.3f}")
    print(f"\nspire_weighted better on test F1 in {wins}/{len(rows)} folds")
    print(f"val-F1 delta (the selection criterion): "
          f"{summary['spire_weighted']['val_f1'] - summary['spire']['val_f1']:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"folds": rows, "summary": summary, "n_folds": len(rows),
                   "weighted_wins": wins}, f, indent=2)
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
