"""
experiment/run_ablation.py

Feature ablation study for SPIRE — Exp 1 (leave-one-region-out inductive).

Three ablation conditions:
  full          : 13-dim (10 SHAP + 1 ensemble + 2 edge-stat) — full pipeline
  no_edge_stats : 11-dim (10 SHAP + 1 ensemble, edge-stat augmentation removed)
  no_ensemble   : 10-dim (10 SHAP only, ensemble score and edge-stats both removed)

Runs each mode n_runs times with different torch seeds (default: 5 seeds matching
Exp 1 multi-run in gnn_inductive.json). Graph construction is deterministic and
built once per (mode, region); only GNN weight initialisation varies across seeds.

Usage:
    python experiment/run_ablation.py                              # all 3 modes, 5 seeds
    python experiment/run_ablation.py --modes no_edge_stats no_ensemble  # skip full re-run
    python experiment/run_ablation.py --seeds 42 1 2              # custom seeds

Results saved to result/ablation_feature.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from config import OUTPUT_PATH
from feature_engineering import (
    build_node_features_no_leakage,
    build_weighted_graph,
    create_spatial_split_indices,
    prepare_xgboost_inputs,
    select_geo_similarity_columns,
    train_xgboost_and_select_features,
)
from gnn_training import (
    GCN,
    GraphSAGE,
    augment_with_edge_stats,
    build_pyg_data,
    build_train_val_subgraph,
    train_gnn_model,
    tune_gnn_model,
)

_EARTH_RADIUS_KM = 6371.0
_GRAPH_K = 15
_GRAPH_ALPHA = 0.5
_GRAPH_MAX_DIST_KM = 2000.0
_PREDICT_K = 10
_RESULTS_PATH = "result/ablation_feature.json"

_ALL_MODES = ["full", "no_edge_stats", "no_ensemble", "no_ensemble_keep_edge_stats"]
_INPUT_DIMS = {"full": 13, "no_edge_stats": 11, "no_ensemble": 10, "no_ensemble_keep_edge_stats": 12}
# Comment (6)-A: `no_ensemble_keep_edge_stats` (12-dim) drops ensemble_score
# but keeps edge-stat augmentation (w_mean/w_max) -- the missing intermediate
# condition needed to fairly isolate the ensemble-prior effect from the
# edge-stat effect, since the original nested ablation (13 -> 11 -> 10)
# confounds the two when comparing 13-dim directly to 10-dim.
_MODES_WITHOUT_ENSEMBLE = {"no_ensemble", "no_ensemble_keep_edge_stats"}
_MODES_WITH_EDGE_STATS = {"full", "no_ensemble_keep_edge_stats"}
_DEFAULT_SEEDS = [42, 1, 2, 4, 7]

# Comment (2)-B: 6 mineral-derived columns dropped from the SHAP candidate
# pool for the mineral/inferred-feature ablation, and the 9 is_group
# deposit-type indicators (Comment 1) used to compensate the lost
# dimensions.
_MINERAL_DERIVED_COLS = [
    "has_ilmenite", "has_magnetite", "has_barite",
    "has_zircon", "has_pyrochlore", "has_fluorite",
]
_IS_GROUP_COLS = [
    "is_carbonatite_system", "is_alkaline_intrusive", "is_mafic_ultramafic",
    "is_felsic_pegmatite", "is_metamorphic_metasomatic", "is_sedimentary",
    "is_placer", "is_supergene_lateritic", "is_unclassified",
]


# ── Graph + feature construction ──────────────────────────────────────────────
# Built once per (mode, region); graph is deterministic given fixed val_split seed.

def _build_graph_artifacts(
    processed_df: pd.DataFrame,
    region: str,
    mode: str,
    train_scoring: str = "oof",
    weighting_strategy: str = "mixed",
    geo_fix: bool = True,
    unify_edge_rule: bool = True,
    exclude_coords: bool = False,
    exclude_minerals: bool = False,
    exclude_lithology: bool = False,
    split: dict | None = None,
) -> dict:
    """
    exclude_lithology : Comment (2)-D lithology-inference ablation
        (run_lithology_ablation.py). `True` drops the 9 `is_group`
        deposit-type/host-lithology indicators (Comment 1) from the SHAP
        candidate pool -- these are the features directly downstream of
        `infer_host_lith`'s rule-based inference, so this quantifies how
        much the model relies on them vs. the reviewer's "circularity
        risk" concern (host lithology itself partly rule-inferred, not
        directly measured). No dimension compensation here (unlike
        `exclude_minerals`) -- the solution text doesn't call for it, and
        `is_group` isn't the same kind of "just happens to also carry the
        signal" feature the 6 mineral columns were.
    exclude_minerals : Comment (2)-B mineral/inferred-feature ablation
        (run_ablation_minerals.py). `True` drops the 6 mineral-derived
        columns (`has_ilmenite`, `has_magnetite`, `has_barite`, `has_zircon`,
        `has_pyrochlore`, `has_fluorite`) from the SHAP *candidate* pool
        before top-10 selection -- so the ensemble (fit on the same
        stripped `X`) and the graph's cosine-similarity term (built from
        `X_selected`, which can no longer contain them) are both leak-free
        too, not just the node vector. Per the solution's own dimension-
        compensation note, the 9 `is_group` deposit-type indicators
        (Comment 1) are then appended to `X_selected` explicitly (skipping
        any already SHAP-selected) so mode="full" stays a fair ~13-dim
        comparison rather than penalizing Model B purely for having fewer
        input dimensions.
    split : Comment (1)-C (LODTO) reuse hook. When given, this pre-computed
        {"train_idx", "val_idx", "test_idx"} dict is used directly instead
        of calling `create_spatial_split_indices(processed_df, test_region=region, ...)`
        internally -- lets `run_lodto.py` reuse this exact function with a
        leave-one-deposit-type-out split instead of leave-one-region-out,
        with `region` then used only as a label for logging (it can be any
        string, e.g. a deposit-type group name).
    exclude_coords : Comment (1)-A follow-up ablation
        (run_ablation_coordinates.py). `False` (default) is the normal
        pipeline. `True` drops Latitude/Longitude from the SHAP *candidate*
        pool before top-10 selection (reuses `prepare_xgboost_inputs`'s
        existing `drop_coordinate_features` flag) -- so the freed slots go
        to the next-best mineral/lithology features and the node vector
        keeps the same dimensionality (13-dim in mode="full") either way,
        making it a fair like-for-like comparison rather than just fewer
        dims. Graph *topology* (spatial k-NN via `coords_tv`, still built
        from real Lat/Long) is untouched -- this only tests reliance on
        Lat/Long as a node *feature*, not on the spatial graph structure
        itself (a separate, already-settled design decision from Comment 13/14).
    weighting_strategy, geo_fix : Comment 13-B 4-condition comparison
        (run_ablation_graph_weighting.py). `weighting_strategy` is passed
        straight through to `build_weighted_graph` ("distance" | "feature_similarity"
        | "mixed"). `geo_fix=True` (default) uses the Comment 13-A fix
        (geo-only vector, excludes Latitude/Longitude/ensemble_score) for the
        cosine-similarity term; `geo_fix=False` reproduces the pre-fix
        double-counting (cosine similarity computed on the full node vector)
        for comparison only — never the default in normal runs.
    unify_edge_rule : Comment 14 — decided: Solution A (unified rule) is now
        the default. The combined formula applies to every edge regardless
        of train/val membership (safe: w_geo depends only on observable
        features x, never the target y). `False` reproduces the old
        asymmetric behaviour (train-train edges get the combined weight,
        edges touching a val node fall back to spatial-only) — kept only
        for run_ablation_edge_rule.py's Comment 14-B comparison, never the
        default in normal runs.
    """
    if split is None:
        split = create_spatial_split_indices(processed_df, test_region=region, val_ratio=0.2)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    graph_df, X, y = prepare_xgboost_inputs(processed_df, fit_idx=train_idx,
                                             drop_coordinate_features=exclude_coords)
    if exclude_minerals:
        X = X.drop(columns=[c for c in _MINERAL_DERIVED_COLS if c in X.columns])
    if exclude_lithology:
        X = X.drop(columns=[c for c in _IS_GROUP_COLS if c in X.columns])
    _, _, X_selected = train_xgboost_and_select_features(X, y, fit_idx=train_idx)
    if exclude_minerals:
        missing_groups = [c for c in _IS_GROUP_COLS if c not in X_selected.columns]
        if missing_groups:
            X_selected = pd.concat([X_selected, X[missing_groups]], axis=1)

    tv_pos = np.sort(np.concatenate([train_idx, val_idx]))
    old_to_new = {int(old): new for new, old in enumerate(tv_pos)}
    new_train = np.array([old_to_new[i] for i in train_idx], dtype=np.int64)
    new_val = np.array([old_to_new[i] for i in val_idx], dtype=np.int64)

    coords_tv = processed_df[["Latitude", "Longitude"]].values[tv_pos]

    if mode in _MODES_WITHOUT_ENSEMBLE:
        # 10-dim (no_ensemble) or 12-dim (no_ensemble_keep_edge_stats):
        # SHAP features only, no ensemble score
        X_node_base = torch.tensor(
            X_selected.iloc[tv_pos].astype(float).values, dtype=torch.float32
        )
        ensemble_models = None
    else:
        # 11-dim: SHAP + ensemble score. `train_scoring="oof"` (default) is
        # leakage-free (Comment 12-A): training nodes get K-fold
        # out-of-fold scores, val nodes get scores from the ensemble refit
        # on the full train_idx (also reused for test-time inference in
        # _predict_test_nodes below). `train_scoring="in_sample"` reproduces
        # the pre-fix leaky behaviour, kept only for the Comment 12-C
        # OOF-vs-in-sample-vs-no-ensemble comparison (run_ablation_ensemble_scoring.py).
        X_node_base, ensemble_models = build_node_features_no_leakage(
            X_selected.iloc[tv_pos], X, y, train_idx, val_idx, new_train, new_val,
            train_scoring=train_scoring,
        )

    # Geo-only vector for cosine similarity (Comment 13): excludes
    # Latitude/Longitude (already used directly as the spatial edge-weight
    # term) and ensemble_score (never part of X_selected) to avoid
    # double-counting them into the "geological similarity" edge weight.
    # geo_fix=False deliberately reuses X_node_base itself (the pre-fix
    # behaviour) for the Comment 13-B comparison.
    if geo_fix:
        X_geo_tv = torch.tensor(
            select_geo_similarity_columns(X_selected.iloc[tv_pos]).astype(float).values,
            dtype=torch.float32,
        )
    else:
        X_geo_tv = X_node_base

    edge_index, edge_weight, sigma = build_weighted_graph(
        coords_tv, X_node_base,
        alpha=_GRAPH_ALPHA, k=_GRAPH_K,
        max_distance_km=_GRAPH_MAX_DIST_KM,
        fit_idx=new_train,
        return_sigma=True,
        weighting_strategy=weighting_strategy,
        X_geo=X_geo_tv,
        unify_edge_rule=unify_edge_rule,
    )

    X_node_tv_base = X_node_base.clone().cpu()
    X_geo_tv_base = X_geo_tv.clone().cpu()

    if mode in _MODES_WITH_EDGE_STATS:
        X_node_base = augment_with_edge_stats(X_node_base, edge_index, edge_weight)

    N_tv = len(tv_pos)
    data = build_pyg_data(X_node_base, edge_index, edge_weight, y.iloc[tv_pos])
    train_mask = torch.zeros(N_tv, dtype=torch.bool)
    val_mask = torch.zeros(N_tv, dtype=torch.bool)
    train_mask[torch.tensor(new_train)] = True
    val_mask[torch.tensor(new_val)] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = torch.zeros(N_tv, dtype=torch.bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    data.x = data.x.detach()

    train_idx_t = torch.tensor(new_train, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(new_val, dtype=torch.long, device=device)
    tv_data, sub_train, sub_val = build_train_val_subgraph(data, train_idx_t, val_idx_t)

    return {
        "tv_data": tv_data,
        "sub_train": sub_train,
        "sub_val": sub_val,
        "device": device,
        "coords_tv": coords_tv,
        "X_node_tv_base": X_node_tv_base,
        "X_geo_tv_base": X_geo_tv_base,
        "weighting_strategy": weighting_strategy,
        "geo_fix": geo_fix,
        "edge_index_tv": tv_data.edge_index,
        "edge_weight_tv": tv_data.edge_weight,
        "graph_sigma": sigma,
        "ensemble_models": ensemble_models,
        "test_idx": test_idx,
        "X_test": X.iloc[test_idx].reset_index(drop=True),
        "X_selected_test": X_selected.iloc[test_idx].reset_index(drop=True),
        "y_test": y.iloc[test_idx].reset_index(drop=True),
    }


# ── GNN training (called once per seed per region) ────────────────────────────

def _train_and_predict(
    graph_artifacts: dict,
    processed_df: pd.DataFrame,
    mode: str,
    model_class=GraphSAGE,
) -> tuple[dict, tuple, float]:
    """
    model_class : GNN architecture to train (default GraphSAGE = SPIRE).
        NOTE (found while building the Comment 13-B comparison): SAGEConv
        (GraphSAGE/SPIRE) and GATConv-as-used-here (GAT) do NOT consume
        `edge_weight` in message passing — only GCNConv (GCN) does. So the
        graph-weighting conditions compared in run_ablation_graph_weighting.py
        only produce different predictions when `model_class=GCN`; for SPIRE
        they differ only through `augment_with_edge_stats` (mode="full").
    """
    tv_data = graph_artifacts["tv_data"]
    sub_train = graph_artifacts["sub_train"]
    sub_val = graph_artifacts["sub_val"]
    device = graph_artifacts["device"]

    best_config, best_val_f1 = tune_gnn_model(model_class, tv_data, sub_train, sub_val, device)
    hidden_dim, num_layers, dropout, lr = best_config

    model, _, _ = train_gnn_model(
        model_class=model_class,
        data=tv_data,
        train_idx=sub_train,
        val_idx=sub_val,
        test_idx=None,
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        epochs=200,
        patience=20,
        device=device,
        criterion=torch.nn.CrossEntropyLoss(),
    )

    inference_artifacts = {
        **{k: v for k, v in graph_artifacts.items()
           if k not in {"tv_data", "sub_train", "sub_val"}},
        "model": model,
        "X_node_tv": tv_data.x,
    }

    y_prob, y_pred = _predict_test_nodes(inference_artifacts, processed_df, mode)
    return y_prob, y_pred, list(best_config), float(best_val_f1)


# ── Inductive prediction ──────────────────────────────────────────────────────

def _predict_node(
    artifacts: dict,
    lat: float,
    lon: float,
    x_new_base: torch.Tensor,
    mode: str,
    x_new_geo: torch.Tensor,
) -> float:
    device = artifacts["device"]
    model = artifacts["model"]
    coords_tv = artifacts["coords_tv"]
    X_node_tv = artifacts["X_node_tv"]
    X_node_tv_base = artifacts["X_node_tv_base"]
    X_geo_tv_base = artifacts["X_geo_tv_base"]
    weighting_strategy = artifacts["weighting_strategy"]
    edge_index_tv = artifacts["edge_index_tv"]
    edge_weight_tv = artifacts["edge_weight_tv"]
    sigma = artifacts["graph_sigma"]

    nbrs = NearestNeighbors(n_neighbors=_PREDICT_K, metric="haversine")
    nbrs.fit(np.radians(coords_tv))
    dist_rad, indices = nbrs.kneighbors(np.radians([[lat, lon]]))
    dist_km = dist_rad[0] * _EARTH_RADIUS_KM
    indices = indices[0]

    within = dist_km <= _GRAPH_MAX_DIST_KM
    neighbour_idx = indices[within] if within.sum() >= 1 else indices[:1]

    mini_coords = np.vstack([[lat, lon], coords_tv[neighbour_idx]])
    mini_X_base = torch.cat([x_new_base.cpu(), X_node_tv_base[neighbour_idx].cpu()], dim=0)
    # Geo-only vector for the mini-graph's cosine similarity (Comment 13) —
    # same fix (or, for the Comment 13-B comparison, same deliberate
    # non-fix) as the TV graph, kept consistent between train and test time.
    mini_X_geo = torch.cat([x_new_geo.cpu(), X_geo_tv_base[neighbour_idx].cpu()], dim=0)

    ei_mini, ew_mini = build_weighted_graph(
        mini_coords, mini_X_base,
        alpha=_GRAPH_ALPHA, k=len(neighbour_idx),
        weighting_strategy=weighting_strategy, sigma=sigma,
        X_geo=mini_X_geo,
    )

    mask0 = (ei_mini[0] == 0) | (ei_mini[1] == 0)
    ei_local = ei_mini[:, mask0]
    ew_local = ew_mini[mask0]

    if mode in _MODES_WITH_EDGE_STATS:
        incoming = ei_local[1] == 0
        if int(incoming.sum()) > 0:
            ew_in = ew_local[incoming]
            mean_ew, max_ew = float(ew_in.mean()), float(ew_in.max())
        else:
            mean_ew = max_ew = 0.0
        edge_stats = torch.tensor([[mean_ew, max_ew]], dtype=torch.float32)
        x_new_aug = torch.cat([x_new_base.cpu(), edge_stats], dim=1).to(device)
    else:
        x_new_aug = x_new_base.to(device)

    N = X_node_tv.shape[0]
    l2g = torch.full((1 + len(neighbour_idx),), -1, dtype=torch.long)
    l2g[0] = N
    l2g[1:] = torch.tensor(neighbour_idx, dtype=torch.long)
    ei_global = l2g[ei_local].to(device)
    ew_global = ew_local.to(device)

    X_aug = torch.cat([X_node_tv, x_new_aug], dim=0)
    ei_aug = torch.cat([edge_index_tv, ei_global], dim=1)
    ew_aug = torch.cat([edge_weight_tv, ew_global], dim=0)

    y_dummy = torch.zeros(N + 1, dtype=torch.long, device=device)
    aug_data = Data(x=X_aug, edge_index=ei_aug, edge_weight=ew_aug, y=y_dummy)

    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(aug_data), dim=1)[N, 1].item()
    return prob


def _predict_test_nodes(
    artifacts: dict, processed_df: pd.DataFrame, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    test_idx = artifacts["test_idx"]
    X_test = artifacts["X_test"]
    X_selected_test = artifacts["X_selected_test"]
    ensemble_models = artifacts["ensemble_models"]
    geo_fix = artifacts["geo_fix"]

    test_coords = processed_df[["Latitude", "Longitude"]].values[test_idx]
    y_probs = []

    for i in range(len(test_idx)):
        x_sel = torch.tensor(
            X_selected_test.iloc[i].astype(float).values, dtype=torch.float
        ).unsqueeze(0)

        if mode in _MODES_WITHOUT_ENSEMBLE:
            x_new_base = x_sel                                       # (1, 10)
        else:
            ens_score = float(np.mean([
                m.predict_proba(X_test.iloc[[i]])[0, 1] for m in ensemble_models
            ]))
            x_ens = torch.tensor([[ens_score]], dtype=torch.float)
            x_new_base = torch.cat([x_sel, x_ens], dim=1)           # (1, 11)

        # geo_fix=False deliberately reuses x_new_base itself (pre-fix
        # behaviour) for the Comment 13-B comparison.
        if geo_fix:
            x_new_geo = torch.tensor(
                select_geo_similarity_columns(X_selected_test.iloc[[i]]).astype(float).values,
                dtype=torch.float,
            )
        else:
            x_new_geo = x_new_base

        lat, lon = test_coords[i]
        prob = _predict_node(artifacts, lat, lon, x_new_base, mode, x_new_geo)
        y_probs.append(prob)

    y_prob = np.array(y_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_prob, y_pred


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    try:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc":  float(roc_auc_score(y_true, y_prob)),
            "pr_auc":   float(average_precision_score(y_true, y_prob)),
        }
    except Exception:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        }


def _avg_metrics(metrics_list: list[dict]) -> dict:
    all_keys = {k for m in metrics_list for k in m}
    return {k: float(np.mean([m[k] for m in metrics_list if k in m])) for k in sorted(all_keys)}


def _std_metrics(metrics_list: list[dict]) -> dict:
    all_keys = {k for m in metrics_list for k in m}
    return {k: float(np.std([m[k] for m in metrics_list if k in m])) for k in sorted(all_keys)}


def _region_avg(per_region: dict) -> dict:
    valid = [m["metrics"] for m in per_region.values() if m.get("metrics")]
    if not valid:
        return {}
    all_keys = {k for m in valid for k in m}
    return {k: float(np.mean([m[k] for m in valid if k in m])) for k in sorted(all_keys)}


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_ablation_sweep(
    processed_df: pd.DataFrame,
    modes: list[str],
    seeds: list[int],
    train_scoring: str = "oof",
    regions: list[str] | None = None,
    weighting_strategy: str = "mixed",
    geo_fix: bool = True,
    model_class=GraphSAGE,
    unify_edge_rule: bool = True,
    exclude_coords: bool = False,
    exclude_minerals: bool = False,
    exclude_lithology: bool = False,
) -> dict:
    # `regions` restricts which held-out test regions are swept; `processed_df`
    # must still contain ALL regions (every other region is training data for
    # whichever region is held out) — never pre-filter processed_df itself.
    if regions is None:
        regions = sorted(processed_df["Region"].dropna().unique().tolist())
    print(f"Regions ({len(regions)}): {regions}")
    print(f"Modes  : {modes}")
    print(f"Seeds  : {seeds}")
    print(f"Train scoring (ensemble modes only): {train_scoring}")

    # results[mode][str(seed)] = {avg_metrics, per_region}
    results: dict = {m: {} for m in modes}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}  ({_INPUT_DIMS[mode]}-dim input)")
        print(f"{'='*60}")

        # Build graph once per region for this mode (deterministic)
        print(f"\n  Building graph artifacts for all regions (mode={mode})...")
        graph_cache: dict = {}
        for region in regions:
            print(f"    {region}...")
            graph_cache[region] = _build_graph_artifacts(
                processed_df, region, mode, train_scoring=train_scoring,
                weighting_strategy=weighting_strategy, geo_fix=geo_fix,
                unify_edge_rule=unify_edge_rule, exclude_coords=exclude_coords,
                exclude_minerals=exclude_minerals, exclude_lithology=exclude_lithology,
            )

        # Train and predict for each seed
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"\n  ── seed={seed} ──")

            per_region: dict = {}
            for region in regions:
                n_test = int(len(graph_cache[region]["test_idx"]))
                y_true = graph_cache[region]["y_test"].to_numpy()

                y_prob, y_pred, best_config, best_val_f1 = _train_and_predict(
                    graph_cache[region], processed_df, mode, model_class=model_class,
                )
                metrics = _compute_metrics(y_true, y_pred, y_prob)
                print(f"    [{mode}/seed={seed}] {region}: f1={metrics['f1']:.4f}")

                per_region[region] = {
                    "n_test":      n_test,
                    "best_config": best_config,
                    "best_val_f1": best_val_f1,
                    "metrics":     metrics,
                }

            seed_avg = _region_avg(per_region)
            print(f"  [{mode}/seed={seed}] avg: {seed_avg}")
            results[mode][str(seed)] = {"avg_metrics": seed_avg, "per_region": per_region}

    # Compute avg ± std across seeds per mode
    summary: dict = {}
    for mode in modes:
        seed_avgs = [results[mode][str(s)]["avg_metrics"] for s in seeds
                     if str(s) in results[mode]]
        if not seed_avgs:
            continue
        avg = _avg_metrics(seed_avgs)
        std = _std_metrics(seed_avgs)
        summary[mode] = {
            "input_dim":  _INPUT_DIMS[mode],
            "n_seeds":    len(seed_avgs),
            "avg_metrics": avg,
            "std_metrics": std,
        }

    return results, summary


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature ablation study (Exp 1 SPIRE)")
    parser.add_argument(
        "--modes", nargs="+", choices=_ALL_MODES, default=_ALL_MODES,
        help="Ablation modes to run (default: all three)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS,
        help=f"Random seeds for GNN initialisation (default: {_DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--train-scoring", choices=["oof", "in_sample"], default="oof",
        help="Ensemble score for training nodes (Comment 12): 'oof' (default, "
             "leakage-free) or 'in_sample' (pre-fix behaviour, for comparison only). "
             "Ignored for mode=no_ensemble.",
    )
    parser.add_argument("--out", type=str, default=_RESULTS_PATH)
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    new_results, new_summary = run_ablation_sweep(processed_df, args.modes, args.seeds, args.train_scoring)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing file if partial run
    existing_results = {}
    existing_summary = {}
    if out_path.exists() and set(args.modes) != set(_ALL_MODES):
        with open(out_path, encoding="utf-8") as f:
            prev = json.load(f)
        existing_results = prev.get("runs", {})
        existing_summary = prev.get("summary", {})

    merged_results = {**existing_results, **new_results}
    merged_summary = {**existing_summary, **new_summary}

    payload = {
        "experiment": "ablation_feature",
        "description": (
            "Feature ablation for SPIRE on Exp 1 (leave-one-region-out inductive). "
            "full=13-dim (SHAP+ensemble+edge-stat), "
            "no_edge_stats=11-dim (SHAP+ensemble), "
            "no_ensemble=10-dim (SHAP only)."
        ),
        "model":   "spire",
        "modes":   list(merged_results.keys()),
        "seeds":   args.seeds,
        "runs":    merged_results,
        "summary": merged_summary,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Ablation summary table ───────────────────────────────────────────────
    print("\n── Ablation summary (avg ± std across seeds) ──────────")
    print(f"{'Mode':<18} {'Dim':>4}  {'F1 (avg±std)':>18}  {'ΔF1 vs full':>12}")
    print("-" * 58)
    full_avg_f1 = merged_summary.get("full", {}).get("avg_metrics", {}).get("f1")
    for mode in _ALL_MODES:
        if mode not in merged_summary:
            continue
        s = merged_summary[mode]
        avg_f1 = s["avg_metrics"]["f1"]
        std_f1 = s["std_metrics"]["f1"]
        dim    = s["input_dim"]
        if full_avg_f1 is not None and mode != "full":
            delta = f"{avg_f1 - full_avg_f1:+.4f}"
        else:
            delta = "—"
        print(f"{mode:<18} {dim:>4}  {avg_f1:.4f} ± {std_f1:.4f}  {delta:>12}")


if __name__ == "__main__":
    main()
