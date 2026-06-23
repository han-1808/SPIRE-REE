"""
experiment/run_baseline_per_region.py

Per-region baseline evaluation — mirrors run_gnn_per_region.py exactly.

For each of the 10 regions:
  - Use ONLY nodes from that region.
  - Hold out the same 5 test nodes (same seed=42) as run_gnn_per_region.py.
  - Train XGBoost, LightGBM, CatBoost, RandomForest, SVM, MLP, Autoencoder, Spatial kNN, Kriging on the rest.
  - Features: full tabular X from prepare_xgboost_inputs (fitted on train only).
  - Evaluate on the 5 held-out test nodes.

Results saved to result/baseline_per_region.json.
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
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from baseline import SimpleAutoencoder
from config import OUTPUT_PATH
from feature_engineering import (
    build_baseline_features,
    prepare_xgboost_inputs,
    train_xgboost_and_select_features,
)

_N_TEST = 5
_RESULTS_PATH = "result/baseline_per_region.json"
N_RUNS = 5


def _split_region_indices(
    region_local_idx: np.ndarray,
    n_test: int = _N_TEST,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(region_local_idx))

    test_local = perm[:n_test]
    rest = perm[n_test:]

    n_val = max(1, int(len(rest) * val_ratio))
    val_local = rest[:n_val]
    train_local = rest[n_val:]

    return train_local, val_local, test_local


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    has_both = len(np.unique(y_true)) > 1
    single_score = 1.0 if np.all(y_pred == y_true) else 0.0
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)) if has_both else single_score,
        "pr_auc":    float(average_precision_score(y_true, y_prob)) if has_both else single_score,
    }


def _make_models(seed: int = 42) -> dict:
    return {
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            random_state=seed, verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbose=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            loss_function="Logloss", random_seed=seed, verbose=False,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1,
        ),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, kernel="rbf", random_state=seed)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)),
        ]),
    }


def _run_region(processed_df: pd.DataFrame, region: str, random_seed: int = 42) -> dict:
    region_global_idx = np.where(processed_df["Region"] == region)[0]
    n_nodes = len(region_global_idx)
    print(f"  Region '{region}': {n_nodes} nodes total")

    if n_nodes < _N_TEST + 5:
        print(f"  Skipping — too few nodes ({n_nodes})")
        return {}

    train_local, val_local, test_local = _split_region_indices(region_global_idx, random_seed=random_seed)
    train_global = region_global_idx[train_local]

    # Feature engineering fitted on train only (11 features: 10 SHAP + xgb_score)
    _, X_full, y_full = prepare_xgboost_inputs(processed_df, fit_idx=train_global)
    X_region = X_full.iloc[region_global_idx].reset_index(drop=True)
    y_region = y_full.iloc[region_global_idx].reset_index(drop=True)

    xgb_model, _, X_selected = train_xgboost_and_select_features(
        X_region, y_region, fit_idx=train_local
    )
    X_bl = build_baseline_features(X_selected, X_region, xgb_model)

    X_train = X_bl.iloc[train_local].astype(float)
    y_train = y_region.iloc[train_local]
    X_test  = X_bl.iloc[test_local].astype(float)
    y_test  = y_region.iloc[test_local].to_numpy()

    coords_region = processed_df[["Latitude", "Longitude"]].values[region_global_idx]
    coords_train  = coords_region[train_local]
    coords_test   = coords_region[test_local]

    per_model: dict = {}
    for name, model in _make_models(seed=random_seed).items():
        try:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = _compute_metrics(y_test, y_prob)
        except Exception as exc:
            metrics = {"error": str(exc)}
        print(f"    [{name}] {metrics}")
        per_model[name] = metrics

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
        y_prob   = pred.predict_proba(te_lat)[:, 1]
        metrics  = _compute_metrics(y_test, y_prob)
    except Exception as exc:
        print(f"    [WARN] autoencoder_predictor: {exc}")
        metrics = {"error": str(exc)}
    print(f"    [autoencoder_predictor] {metrics}")
    per_model["autoencoder_predictor"] = metrics

    # ── Spatial KNN ───────────────────────────────────────────────────────────
    try:
        knn = KNeighborsRegressor(n_neighbors=min(15, len(train_local)), weights="distance", metric="haversine")
        knn.fit(np.radians(coords_train), y_train.to_numpy())
        y_prob  = np.clip(knn.predict(np.radians(coords_test)), 0.0, 1.0)
        metrics = _compute_metrics(y_test, y_prob)
    except Exception as exc:
        print(f"    [WARN] spatial_knn_regression: {exc}")
        metrics = {"error": str(exc)}
    print(f"    [spatial_knn_regression] {metrics}")
    per_model["spatial_knn_regression"] = metrics

    # ── Kriging ───────────────────────────────────────────────────────────────
    try:
        from pykrige.ok import OrdinaryKriging
        kriging = OrdinaryKriging(
            coords_train[:, 1], coords_train[:, 0],
            y_train.to_numpy(dtype=float),
            variogram_model="linear", coordinates_type="geographic",
            verbose=False, enable_plotting=False,
        )
        raw, _ = kriging.execute("points", coords_test[:, 1], coords_test[:, 0])
        y_prob  = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
        metrics = _compute_metrics(y_test, y_prob)
    except Exception as exc:
        print(f"    [WARN] kriging: {exc}")
        metrics = {"error": str(exc)}
    print(f"    [kriging] {metrics}")
    per_model["kriging"] = metrics

    return {
        "n_total": int(n_nodes),
        "n_train": int(len(train_local)),
        "n_val":   int(len(val_local)),
        "n_test":  int(len(test_local)),
        "per_model": per_model,
    }


def _avg_metrics(per_region: dict) -> dict:
    model_names = ["xgboost", "lightgbm", "catboost", "random_forest", "svm", "mlp",
                   "autoencoder_predictor", "spatial_knn_regression", "kriging"]
    metric_keys = ["accuracy", "f1", "roc_auc", "pr_auc"]
    result = {}
    for m in model_names:
        vals = {k: [] for k in metric_keys}
        for region_data in per_region.values():
            if not region_data or "per_model" not in region_data:
                continue
            mm = region_data["per_model"].get(m, {})
            for k in metric_keys:
                v = mm.get(k)
                if v is not None:
                    vals[k].append(v)
        result[m] = {k: float(np.mean(v)) if v else None for k, v in vals.items()}
    return result


def run_baseline_per_region(
    processed_df: pd.DataFrame,
    results_path: str = _RESULTS_PATH,
) -> dict:
    regions = sorted(processed_df["Region"].dropna().unique().tolist())
    print(f"Regions ({len(regions)}): {regions}")

    seeds = [42] + random.sample([s for s in range(61) if s != 42], N_RUNS - 1)
    print(f"Seeds for this run: {seeds}")

    runs: list = []
    for run_id, seed in enumerate(seeds):
        print(f"\n{'#'*60}")
        print(f"  RUN {run_id}  seed={seed}")
        print(f"{'#'*60}")

        per_region: dict = {}
        for region in regions:
            print(f"\n{'='*60}")
            print(f"  Region: {region}  (seed={seed})")
            print(f"{'='*60}")
            per_region[region] = _run_region(processed_df, region, random_seed=seed)

        avg = _avg_metrics(per_region)
        print(f"\n  Avg metrics (seed={seed}):")
        for m, v in avg.items():
            print(f"    {m:20s} Acc={v.get('accuracy', 0) or 0:.3f}  F1={v.get('f1', 0) or 0:.3f}")
        runs.append({"run_id": run_id, "seed": seed, "avg_metrics": avg, "per_region": per_region})

    payload = {
        "experiment": "baseline_per_region",
        "description": (
            f"Tabular baselines trained per-region. "
            f"{_N_TEST} nodes randomly held out as test (same split as gnn_per_region). "
            f"Features: 10 SHAP-selected + xgb_score (11 total, same as GNN)."
        ),
        "n_test_per_region": _N_TEST,
        "models": ["xgboost", "lightgbm", "catboost", "random_forest", "svm", "mlp",
                   "autoencoder_predictor", "spatial_knn_regression", "kriging"],
        "regions": regions,
        "runs": runs,
    }

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {results_path}")
    return payload


def main():
    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)
    run_baseline_per_region(processed_df)


if __name__ == "__main__":
    main()
