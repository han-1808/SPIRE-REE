"""
experiment/run_inductive_sample.py

Inductive evaluation — sample-based split.

For each region:
  - N nodes are randomly sampled from that region as the test set.
  - ALL remaining nodes (including the other nodes in the same region) form train+val.
  - Feature pipeline (scaler, imputer, XGBoost, SHAP selection) fitted on train_idx only.

Both baselines and GNN are evaluated inductively on those N test nodes.

Results saved to result/inductive_sample.json.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import OUTPUT_PATH, SAMPLE_INDUCTIVE_METRICS_PATH
from feature_engineering import (
    _make_single_xgb_model,
    build_node_features_no_leakage,
    build_weighted_graph,
    compute_ensemble_scores_no_leakage,
    create_buffered_sample_split_indices,
    prepare_xgboost_inputs,
    select_geo_similarity_columns,
    train_xgboost_and_select_features,
)
from gnn_training import (
    GAT,
    GCN,
    GraphSAGE,
    augment_with_edge_stats,
    build_pyg_data,
    build_train_val_subgraph,
    train_gnn_model,
    tune_gnn_model,
)
from baseline import SimpleAutoencoder
from run_gnn_per_region import _predict_node_aug

N_SAMPLE = 5  # kept for backward-compat with existing --n-sample CLI default; unused by the split itself now
_TEST_RATIO = 0.15  # Comment (4)-A: was a fixed n_sample=5, now 10-20%/region (mid-point)
SEED     = 42
VAL_RATIO = 0.2
N_RUNS = 5

_GRAPH_K          = 15
_GRAPH_ALPHA      = 0.5
_GRAPH_MAX_DIST_KM = 2000.0


# ── Split ─────────────────────────────────────────────────────────────────────

def _make_sample_split(
    processed_df: pd.DataFrame,
    region: str,
    n_sample: int,
    rng: np.random.Generator,
    val_seed: int,
) -> dict:
    """
    Comment (4)-A/B: thin wrapper around
    `feature_engineering.create_buffered_sample_split_indices` -- test =
    `_TEST_RATIO` of `region`'s own nodes (was a fixed n_sample, the
    `n_sample` parameter is kept only for call-site/CLI backward
    compatibility and is no longer used to size the split), train+val =
    every other node in the entire dataset, with train/val candidates
    within the region's own mean k-NN distance of a test node excluded
    (spatial buffering).
    """
    return create_buffered_sample_split_indices(
        processed_df, region, rng, test_ratio=_TEST_RATIO, val_ratio=VAL_RATIO,
        k_for_buffer=_GRAPH_K, val_seed=val_seed,
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    try:
        return {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_true, y_prob)),
            "pr_auc":    float(average_precision_score(y_true, y_prob)),
        }
    except Exception:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        }


def _avg_metrics(metrics_per_region: dict) -> dict:
    valid = {r: m for r, m in metrics_per_region.items() if m}
    if not valid:
        return {}
    # Average only over entries that contain each key (some regions may return
    # a reduced dict when roc_auc/pr_auc cannot be computed).
    all_keys = {k for m in valid.values() for k in m}
    return {
        k: sum(v[k] for v in valid.values() if k in v)
           / sum(1 for v in valid.values() if k in v)
        for k in sorted(all_keys)
    }


# ── Baseline inductive prediction ─────────────────────────────────────────────

def _run_baseline_for_region(
    processed_df: pd.DataFrame,
    split: dict,
    skip_kriging: bool = False,
    seed: int = 42,
) -> dict:
    """
    Fit all baseline models on train_idx, predict on test_idx (N sampled nodes).
    train_idx includes all non-test nodes — even the rest of the same region.
    """
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]

    _, X, y = prepare_xgboost_inputs(processed_df, fit_idx=train_idx)
    _, _, X_selected = train_xgboost_and_select_features(X, y, fit_idx=train_idx)
    # Leakage-free xgb_score (Comment 17 fairness fix, same OOF machinery as
    # Comment 12-A): train rows get a K-fold out-of-fold score, val/test
    # rows get the score from the model refit on the full train_idx.
    other_idx = np.concatenate([val_idx, test_idx])
    train_scores, other_scores, _ = compute_ensemble_scores_no_leakage(
        X, y, train_idx, other_idx, random_state=seed,
        models_factory=_make_single_xgb_model,
    )
    xgb_score = np.empty(len(X), dtype=float)
    xgb_score[train_idx] = train_scores
    xgb_score[other_idx] = other_scores
    X_bl = X_selected.copy()
    X_bl["xgb_score"] = xgb_score

    X_train = X_bl.iloc[train_idx].astype(float)
    y_train = y.iloc[train_idx]
    X_test  = X_bl.iloc[test_idx].astype(float)
    y_true  = y.iloc[test_idx].to_numpy()

    coords_train = processed_df[["Latitude", "Longitude"]].values[train_idx]
    coords_test  = processed_df[["Latitude", "Longitude"]].values[test_idx]

    results: dict = {}
    predictions: dict = {}  # Comment (4)-D: kept separate, see _avg_metrics

    # ── Tabular models ────────────────────────────────────────────────────────
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    tabular_models = {
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss", random_state=seed,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=seed, verbose=-1, n_jobs=1,
        ),
        "catboost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            loss_function="Logloss", eval_metric="Logloss",
            random_seed=seed, verbose=False,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_split=2, min_samples_leaf=1,
            random_state=seed, n_jobs=-1,
        ),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=True, class_weight="balanced", random_state=seed)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64), activation="relu",
                alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
                max_iter=500, early_stopping=True, random_state=seed,
            )),
        ]),
    }

    for name, model in tabular_models.items():
        try:
            model.fit(X_train, y_train)
            y_prob = np.clip(model.predict_proba(X_test)[:, 1], 0.0, 1.0)
            y_pred = (y_prob >= 0.5).astype(int)
            results[name] = _compute_metrics(y_true, y_pred, y_prob)
            predictions[name] = {"y_true": y_true.tolist(), "y_prob": y_prob.tolist(), "y_pred": y_pred.tolist()}
        except Exception as e:
            print(f"    [WARN] {name}: {e}")
            results[name] = None

    # ── Autoencoder + LogisticRegression ─────────────────────────────────────
    try:
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        ae  = SimpleAutoencoder(X_tr_sc.shape[1], latent_dim=32).to(device)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        X_tr_t = torch.tensor(X_tr_sc, dtype=torch.float32, device=device)

        ae.train()
        for _ in range(50):
            perm = torch.randperm(X_tr_t.size(0), device=device)
            for i in range(0, X_tr_t.size(0), 128):
                b = X_tr_t[perm[i : i + 128]]
                opt.zero_grad()
                recon, _ = ae(b)
                nn.MSELoss()(recon, b).backward()
                opt.step()

        ae.eval()
        with torch.no_grad():
            tr_lat = ae.encoder(X_tr_t).cpu().numpy()
            te_lat = ae.encoder(
                torch.tensor(X_te_sc, dtype=torch.float32, device=device)
            ).cpu().numpy()

        pred = LogisticRegression(max_iter=1000, random_state=42)
        pred.fit(tr_lat, y_train)
        y_prob = pred.predict_proba(te_lat)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results["autoencoder_predictor"] = _compute_metrics(y_true, y_pred, y_prob)
        predictions["autoencoder_predictor"] = {"y_true": y_true.tolist(), "y_prob": y_prob.tolist(), "y_pred": y_pred.tolist()}
    except Exception as e:
        print(f"    [WARN] autoencoder_predictor: {e}")
        results["autoencoder_predictor"] = None

    # ── Spatial KNN ───────────────────────────────────────────────────────────
    try:
        knn_inner = KNeighborsRegressor(n_neighbors=15, weights="distance", metric="haversine")
        knn_inner.fit(np.radians(coords_train), y_train.to_numpy())
        y_prob = np.clip(knn_inner.predict(np.radians(coords_test)), 0.0, 1.0)
        y_pred = (y_prob >= 0.5).astype(int)
        results["spatial_knn_regression"] = _compute_metrics(y_true, y_pred, y_prob)
        predictions["spatial_knn_regression"] = {"y_true": y_true.tolist(), "y_prob": y_prob.tolist(), "y_pred": y_pred.tolist()}
    except Exception as e:
        print(f"    [WARN] spatial_knn_regression: {e}")
        results["spatial_knn_regression"] = None

    # ── Kriging ───────────────────────────────────────────────────────────────
    if not skip_kriging:
        try:
            from pykrige.ok import OrdinaryKriging
            kriging = OrdinaryKriging(
                coords_train[:, 1], coords_train[:, 0],
                y_train.to_numpy(dtype=float),
                variogram_model="linear", coordinates_type="geographic",
                verbose=False, enable_plotting=False,
            )
            raw, _ = kriging.execute("points", coords_test[:, 1], coords_test[:, 0])
            y_prob = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
            y_pred = (y_prob >= 0.5).astype(int)
            results["kriging"] = _compute_metrics(y_true, y_pred, y_prob)
            predictions["kriging"] = {"y_true": y_true.tolist(), "y_prob": y_prob.tolist(), "y_pred": y_pred.tolist()}
        except Exception as e:
            print(f"    [WARN] kriging: {e}")
            results["kriging"] = None

    return results, predictions


# ── GNN inductive prediction ──────────────────────────────────────────────────

def _build_graph_artifacts(processed_df: pd.DataFrame, split: dict, seed: int = 42) -> dict:
    """
    Build the train+val graph and fit ensemble models. Does NOT train any GNN.
    Returns a dict of graph data and prediction artifacts shared by all GNN models.
    """
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]

    _, X, y = prepare_xgboost_inputs(processed_df, fit_idx=train_idx)
    _, _, X_selected = train_xgboost_and_select_features(X, y, fit_idx=train_idx)

    tv_pos    = np.sort(np.concatenate([train_idx, val_idx]))
    old_to_new = {int(old): new for new, old in enumerate(tv_pos)}
    new_train  = np.array([old_to_new[i] for i in train_idx], dtype=np.int64)
    new_val    = np.array([old_to_new[i] for i in val_idx],   dtype=np.int64)

    coords_tv   = processed_df[["Latitude", "Longitude"]].values[tv_pos]
    # Leakage-free (Comment 12): train nodes get K-fold OOF scores, val nodes
    # get scores from the ensemble refit on the full train_idx (also reused
    # for test-time inference below).
    X_node_base, ensemble_models = build_node_features_no_leakage(
        X_selected.iloc[tv_pos], X, y, train_idx, val_idx, new_train, new_val,
        random_state=seed,
    )
    # Geo-only vector for cosine similarity (Comment 13): excludes
    # Latitude/Longitude and ensemble_score to avoid double-counting them
    # into the "geological similarity" edge weight.
    X_geo_tv = torch.tensor(
        select_geo_similarity_columns(X_selected.iloc[tv_pos]).astype(float).values,
        dtype=torch.float32,
    )
    edge_index, edge_weight, sigma = build_weighted_graph(
        coords_tv, X_node_base,
        alpha=_GRAPH_ALPHA, k=_GRAPH_K,
        max_distance_km=_GRAPH_MAX_DIST_KM,
        fit_idx=new_train,
        return_sigma=True,
        X_geo=X_geo_tv,
        unify_edge_rule=True,  # Comment 14 Solution A (decided)
    )

    X_node_tv_base = X_node_base.clone().cpu()
    X_geo_tv_base = X_geo_tv.clone().cpu()
    X_node_base = augment_with_edge_stats(X_node_base, edge_index, edge_weight)

    N_tv = len(tv_pos)
    data = build_pyg_data(X_node_base, edge_index, edge_weight, y.iloc[tv_pos])
    train_mask = torch.zeros(N_tv, dtype=torch.bool)
    val_mask   = torch.zeros(N_tv, dtype=torch.bool)
    train_mask[torch.tensor(new_train)] = True
    val_mask[torch.tensor(new_val)]     = True
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = torch.zeros(N_tv, dtype=torch.bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = data.to(device)
    data.x = data.x.detach()

    train_idx_t = torch.tensor(new_train, dtype=torch.long, device=device)
    val_idx_t   = torch.tensor(new_val,   dtype=torch.long, device=device)
    tv_data, sub_train, sub_val = build_train_val_subgraph(data, train_idx_t, val_idx_t)

    return {
        # For GNN training
        "tv_data":   tv_data,
        "sub_train": sub_train,
        "sub_val":   sub_val,
        "device":    device,
        # For inductive prediction (_predict_node_aug)
        "coords_tv":       coords_tv,
        "X_node_tv":       tv_data.x,
        "X_node_tv_base":  X_node_tv_base,
        "X_geo_tv_base":   X_geo_tv_base,
        "edge_index_tv":   tv_data.edge_index,
        "edge_weight_tv":  tv_data.edge_weight,
        "graph_sigma":     sigma,
        # For test node feature construction
        "ensemble_models": ensemble_models,
        "test_idx":        test_idx,
        "X_test":          X.iloc[test_idx].reset_index(drop=True),
        "X_selected_test": X_selected.iloc[test_idx].reset_index(drop=True),
        "y_test":          y.iloc[test_idx].reset_index(drop=True),
    }


def _train_gnn_model_on_artifacts(
    model_class,
    graph_artifacts: dict,
    best_config: tuple | None = None,
) -> tuple[dict, tuple]:
    """
    Train model_class on pre-built graph artifacts.
    Tunes with grid search if best_config is None; otherwise uses the given config.
    Returns (prediction_artifacts_with_model, best_config).
    """
    tv_data   = graph_artifacts["tv_data"]
    sub_train = graph_artifacts["sub_train"]
    sub_val   = graph_artifacts["sub_val"]
    device    = graph_artifacts["device"]

    if best_config is None:
        best_config, _ = tune_gnn_model(model_class, tv_data, sub_train, sub_val, device)

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

    artifacts = {k: v for k, v in graph_artifacts.items()
                 if k not in {"tv_data", "sub_train", "sub_val"}}
    artifacts["model"] = model
    return artifacts, best_config


def _predict_with_artifacts(
    artifacts: dict,
    test_idx: np.ndarray,
    test_coords: np.ndarray,
) -> dict:
    """Inductively predict test nodes using the trained model in artifacts."""
    X_selected_test = artifacts["X_selected_test"]
    X_test          = artifacts["X_test"]
    ensemble_models = artifacts["ensemble_models"]

    y_probs = []
    for i in range(len(test_idx)):
        ens_score = float(np.mean([
            m.predict_proba(X_test.iloc[[i]])[0, 1] for m in ensemble_models
        ]))
        x_sel = torch.tensor(
            X_selected_test.iloc[i].astype(float).values, dtype=torch.float
        ).unsqueeze(0)
        x_new_geo = torch.tensor(
            select_geo_similarity_columns(X_selected_test.iloc[[i]]).astype(float).values,
            dtype=torch.float,
        )
        x_ens = torch.tensor([[ens_score]], dtype=torch.float)
        x_new_base = torch.cat([x_sel, x_ens], dim=1)

        lat, lon = test_coords[i]
        prob = _predict_node_aug(artifacts, lat, lon, x_new_base, x_new_geo)
        y_probs.append(prob)

    y_prob = np.array(y_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = artifacts["y_test"].to_numpy()
    metrics = _compute_metrics(y_true, y_pred, y_prob)
    predictions = {"y_true": y_true.tolist(), "y_prob": y_prob.tolist(), "y_pred": y_pred.tolist()}
    return metrics, predictions  # Comment (4)-D: predictions kept separate, see _avg_metrics


def _run_gnn_for_region(processed_df: pd.DataFrame, split: dict, seed: int = 42) -> dict:
    """
    Build graph once, tune SPIRE, then train GCN and GAT with the same config.
    Returns dict mapping model name → test metrics.
    """
    graph_artifacts = _build_graph_artifacts(processed_df, split, seed=seed)
    test_idx    = split["test_idx"]
    test_coords = processed_df[["Latitude", "Longitude"]].values[test_idx]

    best_config = None
    metrics = {}
    predictions = {}  # Comment (4)-D

    for name, model_class in [("spire", GraphSAGE), ("gcn", GCN), ("gat", GAT)]:
        label = "Tuning+Training" if best_config is None else f"Training (config from SPIRE={best_config})"
        print(f"    [{name.upper()}] {label}")
        arts, found_config = _train_gnn_model_on_artifacts(model_class, graph_artifacts, best_config)
        if best_config is None:
            best_config = found_config
        metrics[name], predictions[name] = _predict_with_artifacts(arts, test_idx, test_coords)
        print(f"    {name:<12}: {metrics[name]}")

    return metrics, predictions


# ── Main sweep ─────────────────────────────────────────────────────────────────

def _run_one_seed(
    processed_df: pd.DataFrame,
    regions: list,
    n_sample: int,
    seed: int,
    skip_kriging: bool = False,
) -> dict:
    """Run one full sweep (all regions) for a single seed. Returns run-level result."""
    import torch as _torch
    _torch.manual_seed(seed)

    rng = np.random.default_rng(seed)
    gnn_model_names = ["spire", "gcn", "gat"]
    baseline_per_region: dict = {}
    baseline_predictions_per_region: dict = {}  # Comment (4)-D
    gnn_per_region: dict = {name: {} for name in gnn_model_names}

    for i, region in enumerate(regions):
        print(f"{'='*60}")
        print(f"  Test region: {region}  (seed={seed})")

        split = _make_sample_split(processed_df, region, n_sample, rng, val_seed=seed + i)
        n = split["n"]
        print(f"  Sampled {n} test nodes (train size={len(split['train_idx'])})")

        print("  [Baseline]")
        bl_results, bl_predictions = _run_baseline_for_region(
            processed_df, split, skip_kriging=skip_kriging, seed=seed
        )
        baseline_per_region[region] = bl_results
        baseline_predictions_per_region[region] = bl_predictions

        print("  [GNN]")
        gnn_metrics_dict, gnn_predictions_dict = _run_gnn_for_region(processed_df, split, seed=seed)
        for name in gnn_model_names:
            gnn_per_region[name][region] = {
                "n_test_sample": n,
                "sample_idx":    split["test_idx"].tolist(),
                "test_metrics":  gnn_metrics_dict[name],
                "predictions":   gnn_predictions_dict[name],  # Comment (4)-D
            }

    all_bl_models = (
        [m for m in baseline_per_region[regions[0]].keys() if not (skip_kriging and m == "kriging")]
        if baseline_per_region else []
    )
    baseline_avg = {
        m: _avg_metrics({r: baseline_per_region[r].get(m) for r in regions})
        for m in all_bl_models
    }

    gnn_sections = {
        name: {
            "avg_metrics": _avg_metrics({r: gnn_per_region[name][r]["test_metrics"] for r in regions}),
            "per_region": gnn_per_region[name],
        }
        for name in gnn_model_names
    }

    return {
        **gnn_sections,
        "baselines": {
            "avg_metrics": baseline_avg,
            "per_region": baseline_per_region,
            "predictions_per_region": baseline_predictions_per_region,  # Comment (4)-D
        },
    }


def run_inductive_sample_sweep(
    processed_df: pd.DataFrame,
    n_sample: int = N_SAMPLE,
    metrics_path: str = SAMPLE_INDUCTIVE_METRICS_PATH,
    skip_kriging: bool = False,
    seeds: list[int] | None = None,
) -> dict:
    """
    For each region: sample n_sample nodes → test; all others → train+val.
    Runs N_RUNS times with different random seeds.
    """
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    print(f"Regions ({len(regions)}): {regions}")

    if seeds is None:
        seeds = [42] + random.sample([s for s in range(61) if s != 42], N_RUNS - 1)
    print(f"Seeds for this run: {seeds}")

    runs: list = []
    for run_id, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"  RUN {run_id}  seed={seed}  n_sample={n_sample}")
        print(f"{'#'*60}")
        run_result = _run_one_seed(processed_df, regions, n_sample, seed, skip_kriging)
        runs.append({"run_id": run_id, "seed": seed, **run_result})

    # Merge with any existing results file so a partial re-run (e.g. a
    # single seed) doesn't discard other seeds already saved there.
    if Path(metrics_path).exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing_runs = json.load(f).get("runs", [])
        by_seed = {r["seed"]: r for r in existing_runs}
        for r in runs:
            by_seed[r["seed"]] = r  # new runs overwrite same-seed old ones
        runs = [by_seed[s] for s in sorted(by_seed.keys())]
        for i, r in enumerate(runs):
            r["run_id"] = i
        print(f"Merged with existing results → {len(runs)} seed(s) total: {sorted(by_seed.keys())}")

    payload = {
        "experiment": "inductive_sample",
        "description": (
            f"{n_sample} nodes sampled from each region as test; "
            "all remaining nodes (including rest of same region) are in train+val. "
            "GCN and GAT use the same hyperparameters tuned for SPIRE."
        ),
        "n_sample": n_sample,
        "regions":  regions,
        "runs":     runs,
    }

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {metrics_path}")
    return payload


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sample",     type=int,  default=N_SAMPLE)
    parser.add_argument("--skip-kriging", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                         help="Explicit seed list (default: 42 + 4 random)")
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    run_inductive_sample_sweep(
        processed_df,
        n_sample=args.n_sample,
        skip_kriging=args.skip_kriging,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()