"""
experiment/run_ablation_random_rewire.py

Comment (17)-D: random-rewire control.

Keeps the exact same node features and the exact same degree distribution,
but randomly rewires *which* nodes are connected -- via `networkx`'s
`double_edge_swap` (a sequence of degree-preserving edge swaps: pick 2
edges (a,b)/(c,d), replace with (a,d)/(c,b) if that doesn't create a
self-loop or duplicate edge). This tests whether SPIRE benefits from the
*real* spatial/feature-similarity graph structure specifically, or merely
from *having some graph* (any degree-preserving topology) to pass through
GraphSAGE's aggregation.

Edge weights are preserved as a value distribution but reassigned to the
new (rewired) edges via a random permutation -- the spec only asks to
"randomly shuffle the connections", not to change the weight distribution,
so `w_mean`/`w_max` (mode="full"'s edge-stat augmentation) stay on a
comparable scale to the real-graph condition; only which pairs of nodes
they connect is randomized.

Both `tv_data.edge_index`/`.edge_weight` (used for training) AND
`graph_artifacts["edge_index_tv"]`/`["edge_weight_tv"]` (reused directly as
the base graph for test-time inductive augmentation, see `_predict_node` in
run_ablation.py) are rewired together, so training and inductive test-node
prediction operate on the same rewired base graph consistently -- rewiring
only `tv_data` and leaving `edge_index_tv` untouched would silently train
on one graph and predict on another. A new query node's own edges to its
k nearest neighbours are still built from its real coordinates/features (an
inductive node's own connections can't be meaningfully "rewired" in
isolation) -- only the *existing* training-graph topology it attaches to
is randomized.

Usage:
    python experiment/run_ablation_random_rewire.py
    python experiment/run_ablation_random_rewire.py --seeds 42 1 2
    python experiment/run_ablation_random_rewire.py --regions Oceania "South and Central Asia"

Results saved to result/ablation_random_rewire.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import networkx as nx
import numpy as np
import pandas as pd
import torch

from config import OUTPUT_PATH
from gnn_training import GraphSAGE
from run_ablation import (
    _DEFAULT_SEEDS,
    _build_graph_artifacts,
    _compute_metrics,
    _region_avg,
    _train_and_predict,
)

_RESULTS_PATH = "result/ablation_random_rewire.json"
_MODE = "full"


def _rewire_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                   n_nodes: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    ei = edge_index.cpu().numpy()
    ew = edge_weight.cpu().numpy()
    src, dst = ei[0], ei[1]
    undirected_mask = src < dst  # edge_index is bidirectional; keep one direction per pair

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(zip(src[undirected_mask].tolist(), dst[undirected_mask].tolist()))
    n_edges = G.number_of_edges()

    nx.double_edge_swap(G, nswap=max(1, n_edges * 2), max_tries=max(10, n_edges * 40), seed=seed)

    new_edges = np.array(list(G.edges()), dtype=np.int64).T  # (2, n_undirected)
    new_edge_index = np.concatenate([new_edges, new_edges[::-1]], axis=1)

    rng = np.random.default_rng(seed)
    shuffled_w = rng.permutation(ew[undirected_mask])
    new_edge_weight = np.concatenate([shuffled_w, shuffled_w])

    return (
        torch.tensor(new_edge_index, dtype=edge_index.dtype, device=edge_index.device),
        torch.tensor(new_edge_weight, dtype=edge_weight.dtype, device=edge_weight.device),
    )


def _apply_rewire(graph_artifacts: dict, seed: int) -> dict:
    tv_data = graph_artifacts["tv_data"]
    n_nodes = tv_data.x.shape[0]
    new_ei, new_ew = _rewire_edges(tv_data.edge_index, tv_data.edge_weight, n_nodes, seed)
    tv_data.edge_index = new_ei
    tv_data.edge_weight = new_ew
    graph_artifacts["edge_index_tv"] = new_ei
    graph_artifacts["edge_weight_tv"] = new_ew
    return graph_artifacts


def run_random_rewire_comparison(
    processed_df: pd.DataFrame,
    seeds: list[int],
    regions: list[str] | None = None,
) -> dict:
    if regions is None:
        regions = sorted(processed_df["Region"].dropna().unique().tolist())

    results: dict = {"real_graph": {}, "random_rewired": {}}
    for seed in seeds:
        print(f"\n{'#'*60}\n  RUN seed={seed}\n{'#'*60}")
        for cond_name, rewire in [("real_graph", False), ("random_rewired", True)]:
            print(f"\n  ── {cond_name} ──")
            # Seed the network initialisation, not just the rewiring RNG.
            #
            # Previously `seed` reached only _apply_rewire (nx.double_edge_swap and
            # the weight permutation), so the GNN weights were drawn from whatever
            # global torch state happened to be current. The five "seeds" were
            # therefore five uncontrolled repeats of one configuration: the reported
            # f1_std measured run-to-run jitter rather than seed variance, the run was
            # not reproducible, and the real_graph arm did not reproduce the value the
            # feature-ablation harness obtains for the identical configuration
            # (0.8367 against 0.8374). Seeding here restores both properties and makes
            # the two arms paired on initialisation as well as on split.
            torch.manual_seed(seed)
            np.random.seed(seed)
            per_region: dict = {}
            for region in regions:
                graph_artifacts = _build_graph_artifacts(processed_df, region, _MODE)
                if rewire:
                    graph_artifacts = _apply_rewire(graph_artifacts, seed)

                y_true = graph_artifacts["y_test"].to_numpy()
                y_prob, y_pred, best_config, best_val_f1 = _train_and_predict(
                    graph_artifacts, processed_df, _MODE, model_class=GraphSAGE,
                )
                metrics = _compute_metrics(y_true, y_pred, y_prob)
                print(f"    [{cond_name}/seed={seed}] {region}: f1={metrics['f1']:.4f}")
                per_region[region] = {
                    "n_test": int(len(y_true)), "best_config": best_config,
                    "best_val_f1": best_val_f1, "metrics": metrics,
                }
            results[cond_name][str(seed)] = {
                "avg_metrics": _region_avg(per_region), "per_region": per_region,
            }

    return results


def _summarize(results: dict, seeds: list[int]) -> dict:
    """Store every metric, not F1 alone.

    The previous version kept only f1_mean/f1_std, which forced the ROC-AUC and
    PR-AUC values quoted in the response to be averaged by hand from the per-seed
    records. Those are precisely the metrics on which the real graph separates from
    the rewired one, so they belong in the stored summary.
    """
    summary = {}
    for cond in ["real_graph", "random_rewired"]:
        per_seed = [results[cond][str(s)]["avg_metrics"] for s in seeds]
        keys = {k for m in per_seed for k in m}
        block = {}
        for k in sorted(keys):
            vals = [m[k] for m in per_seed if k in m]
            block[f"{k}_mean"] = float(np.mean(vals))
            block[f"{k}_std"] = float(np.std(vals))  # population SD (ddof=0)
        summary[cond] = block
    return summary


def main():
    parser = argparse.ArgumentParser(description="Comment (17)-D: real graph vs randomly rewired graph")
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", type=str, default=None,
                         help="Restrict to a subset of regions (default: all)")
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)
    regions = args.regions or sorted(processed_df["Region"].dropna().unique().tolist())

    results = run_random_rewire_comparison(processed_df, args.seeds, regions)
    summary = _summarize(results, args.seeds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "ablation_random_rewire",
        "description": (
            "Comment (17)-D: SPIRE (mode=full) on the real spatial/feature "
            "graph vs a degree-preserving randomly rewired graph (networkx "
            "double_edge_swap) -- same node features, same degree "
            "distribution, randomized connectivity."
        ),
        "regions": regions,
        "seeds": args.seeds,
        "summary": summary,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print("\n── Comment (17)-D summary ──")
    print(f"{'condition':<16}{'F1':>18}{'ROC-AUC':>18}{'PR-AUC':>18}")
    for cond in ("real_graph", "random_rewired"):
        row = f"{cond:<16}"
        for k in ("f1", "roc_auc", "pr_auc"):
            row += f"{summary[cond][f'{k}_mean']:.4f} ± {summary[cond][f'{k}_std']:.4f}".rjust(18)
        print(row)


if __name__ == "__main__":
    main()
