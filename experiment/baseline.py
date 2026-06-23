import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import json

import numpy as np
import torch
import torch.nn as nn
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

from config import (
    BASELINE_METRICS_PATH,
)
from feature_engineering import create_spatial_split_indices


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }



def _load_json_dict(metrics_path):
    path = Path(metrics_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload if isinstance(payload, dict) else {}


def _save_results(results, metrics_path, model_key):
    if metrics_path is not None:
        payload = _load_json_dict(metrics_path)
        payload.setdefault("models", {})
        payload["models"][model_key] = results

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"{results['model_name']} metrics saved to: {metrics_path}")


def _split_tabular_data(graph_df, X, y, test_region, val_ratio):
    split = create_spatial_split_indices(graph_df, test_region=test_region, val_ratio=val_ratio)
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "X_train": X.iloc[train_idx],
        "X_val": X.iloc[val_idx],
        "X_test": X.iloc[test_idx],
        "y_train": y.iloc[train_idx],
        "y_val": y.iloc[val_idx],
        "y_test": y.iloc[test_idx],
    }


def train_tabular_baseline(
    model,
    model_key,
    model_name,
    graph_df,
    X,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=None,
):
    split = _split_tabular_data(graph_df, X, y, test_region, val_ratio)

    model.fit(split["X_train"], split["y_train"])

    test_prob = model.predict_proba(split["X_test"])[:, 1]
    test_metrics = compute_metrics(split["y_test"].to_numpy(), test_prob)

    results = {
        "model_name": model_name,
        "test_region": test_region,
        "val_ratio": val_ratio,
        "train_size": int(len(split["train_idx"])),
        "val_size": int(len(split["val_idx"])),
        "test_size": int(len(split["test_idx"])),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    _save_results(results, metrics_path, model_key)

    print(f"{model_name} test metrics:", results["test_metrics"])
    return {
        "model": model,
        "train_idx": split["train_idx"],
        "val_idx": split["val_idx"],
        "test_idx": split["test_idx"],
        "results": results,
    }


def _get_coords(processed_df, indices):
    coords = processed_df.iloc[indices][["Latitude", "Longitude"]].to_numpy(dtype=float)
    return coords


def train_spatial_regression_baseline(
    model,
    model_key,
    model_name,
    processed_df,
    graph_df,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=None,
):
    split = create_spatial_split_indices(graph_df, test_region=test_region, val_ratio=val_ratio)
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]

    coords_train = _get_coords(processed_df, train_idx)
    coords_test = _get_coords(processed_df, test_idx)

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model.fit(coords_train, y_train)

    test_prob = np.clip(model.predict(coords_test), 0.0, 1.0)
    test_metrics = compute_metrics(y_test.to_numpy(), test_prob)

    results = {
        "model_name": model_name,
        "test_region": test_region,
        "val_ratio": val_ratio,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    _save_results(results, metrics_path, model_key)

    print(f"{model_name} test metrics:", results["test_metrics"])
    return {
        "model": model,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "results": results,
    }


def train_xgboost_baseline(
    graph_df,
    X,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=BASELINE_METRICS_PATH,
    seed=42,
):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
    )
    return train_tabular_baseline(
        model=model,
        model_key="xgboost",
        model_name="XGBoost baseline",
        graph_df=graph_df,
        X=X,
        y=y,
        test_region=test_region,
        val_ratio=val_ratio,
        metrics_path=metrics_path,
    )


def train_lightgbm_baseline(
    graph_df,
    X,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=BASELINE_METRICS_PATH,
    seed=42,
):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    return train_tabular_baseline(
        model=model,
        model_key="lightgbm",
        model_name="LightGBM baseline",
        graph_df=graph_df,
        X=X,
        y=y,
        test_region=test_region,
        val_ratio=val_ratio,
        metrics_path=metrics_path,
    )


def train_catboost_baseline(
    graph_df,
    X,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=BASELINE_METRICS_PATH,
    seed=42,
):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=seed,
        verbose=False,
    )
    return train_tabular_baseline(
        model=model,
        model_key="catboost",
        model_name="CatBoost baseline",
        graph_df=graph_df,
        X=X,
        y=y,
        test_region=test_region,
        val_ratio=val_ratio,
        metrics_path=metrics_path,
    )


def train_random_forest_baseline(
    graph_df,
    X,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=BASELINE_METRICS_PATH,
    seed=42,
):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    return train_tabular_baseline(
        model=model,
        model_key="random_forest",
        model_name="RandomForest baseline",
        graph_df=graph_df,
        X=X,
        y=y,
        test_region=test_region,
        val_ratio=val_ratio,
        metrics_path=metrics_path,
    )


def train_svm_baseline(
    graph_df,
    X,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=BASELINE_METRICS_PATH,
    seed=42,
):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    return train_tabular_baseline(
        model=model,
        model_key="svm",
        model_name="SVM baseline",
        graph_df=graph_df,
        X=X,
        y=y,
        test_region=test_region,
        val_ratio=val_ratio,
        metrics_path=metrics_path,
    )


def train_mlp_baseline(
    graph_df,
    X,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=BASELINE_METRICS_PATH,
    seed=42,
):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=128,
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )
    return train_tabular_baseline(
        model=model,
        model_key="mlp",
        model_name="MLP baseline",
        graph_df=graph_df,
        X=X,
        y=y,
        test_region=test_region,
        val_ratio=val_ratio,
        metrics_path=metrics_path,
    )


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder_predictor_baseline(
    graph_df,
    X,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=BASELINE_METRICS_PATH,
    latent_dim=32,
    epochs=50,
    seed=42,
):
    split = _split_tabular_data(graph_df, X, y, test_region, val_ratio)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(split["X_train"])
    X_val = scaler.transform(split["X_val"])
    X_test = scaler.transform(split["X_test"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = SimpleAutoencoder(X_train.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    batch_size = 128

    autoencoder.train()
    for _ in range(epochs):
        perm = torch.randperm(X_train_tensor.size(0), device=device)
        for i in range(0, X_train_tensor.size(0), batch_size):
            idx = perm[i : i + batch_size]
            batch = X_train_tensor[idx]
            optimizer.zero_grad()
            recon, _ = autoencoder(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

    autoencoder.eval()
    with torch.no_grad():
        train_latent = autoencoder.encoder(
            torch.tensor(X_train, dtype=torch.float32, device=device)
        ).cpu().numpy()
        val_latent = autoencoder.encoder(
            torch.tensor(X_val, dtype=torch.float32, device=device)
        ).cpu().numpy()
        test_latent = autoencoder.encoder(
            torch.tensor(X_test, dtype=torch.float32, device=device)
        ).cpu().numpy()

    predictor = LogisticRegression(max_iter=1000, random_state=seed)
    predictor.fit(train_latent, split["y_train"])

    test_prob = predictor.predict_proba(test_latent)[:, 1]
    test_metrics = compute_metrics(split["y_test"].to_numpy(), test_prob)

    results = {
        "model_name": "Autoencoder+Predictor baseline",
        "test_region": test_region,
        "val_ratio": val_ratio,
        "train_size": int(len(split["train_idx"])),
        "val_size": int(len(split["val_idx"])),
        "test_size": int(len(split["test_idx"])),
        "latent_dim": int(latent_dim),
        "epochs": int(epochs),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    _save_results(results, metrics_path, "autoencoder_predictor")

    print("Autoencoder+Predictor baseline test metrics:", results["test_metrics"])
    return {
        "autoencoder": autoencoder,
        "predictor": predictor,
        "train_idx": split["train_idx"],
        "val_idx": split["val_idx"],
        "test_idx": split["test_idx"],
        "results": results,
    }


def train_spatial_knn_regression_baseline(
    processed_df,
    graph_df,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=BASELINE_METRICS_PATH,
):
    class HaversineKNNRegressor:
        def __init__(self, n_neighbors=15, weights="distance"):
            self.model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                metric="haversine",
            )

        def fit(self, coords, target):
            self.model.fit(np.radians(coords), target)

        def predict(self, coords):
            return self.model.predict(np.radians(coords))

    model = HaversineKNNRegressor(n_neighbors=15, weights="distance")
    return train_spatial_regression_baseline(
        model=model,
        model_key="spatial_knn_regression",
        model_name="Spatial kNN regression baseline",
        processed_df=processed_df,
        graph_df=graph_df,
        y=y,
        test_region=test_region,
        val_ratio=val_ratio,
        metrics_path=metrics_path,
    )


def train_kriging_baseline(
    processed_df,
    graph_df,
    y,
    test_region="South and Central Asia",
    val_ratio=0.2,
    metrics_path=BASELINE_METRICS_PATH,
):
    from pykrige.ok import OrdinaryKriging

    split = create_spatial_split_indices(graph_df, test_region=test_region, val_ratio=val_ratio)
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]

    train_coords = _get_coords(processed_df, train_idx)
    test_coords = _get_coords(processed_df, test_idx)

    y_train = y.iloc[train_idx].to_numpy(dtype=float)
    y_test = y.iloc[test_idx].to_numpy(dtype=float)

    kriging = OrdinaryKriging(
        train_coords[:, 1],
        train_coords[:, 0],
        y_train,
        variogram_model="linear",
        coordinates_type="geographic",
        verbose=False,
        enable_plotting=False,
    )

    test_prob, _ = kriging.execute("points", test_coords[:, 1], test_coords[:, 0])
    test_prob = np.clip(np.asarray(test_prob, dtype=float), 0.0, 1.0)
    test_metrics = compute_metrics(y_test, test_prob)

    results = {
        "model_name": "Kriging baseline",
        "test_region": test_region,
        "val_ratio": val_ratio,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    _save_results(results, metrics_path, "kriging")

    print("Kriging baseline test metrics:", results["test_metrics"])
    return {
        "model": kriging,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "results": results,
    }


# ── Proportion-prediction helpers (return raw y_prob arrays) ──────────────────
# These accept explicit numpy index arrays so proportion experiments can reuse
# the same train/test split without a second internal split.

def predict_proportion_autoencoder(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    latent_dim: int = 32,
    epochs: int = 50,
) -> np.ndarray:
    """Autoencoder + LogisticRegression. Returns y_prob for test nodes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    autoencoder = SimpleAutoencoder(X_tr.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)

    autoencoder.train()
    for _ in range(epochs):
        perm = torch.randperm(X_tr_t.size(0), device=device)
        for i in range(0, X_tr_t.size(0), 128):
            batch = X_tr_t[perm[i : i + 128]]
            optimizer.zero_grad()
            recon, _ = autoencoder(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

    autoencoder.eval()
    with torch.no_grad():
        train_latent = autoencoder.encoder(X_tr_t).cpu().numpy()
        test_latent = autoencoder.encoder(
            torch.tensor(X_te, dtype=torch.float32, device=device)
        ).cpu().numpy()

    predictor = LogisticRegression(max_iter=1000, random_state=42)
    predictor.fit(train_latent, y_train)
    return predictor.predict_proba(test_latent)[:, 1]


def predict_proportion_spatial_knn(
    coords_train: np.ndarray,
    coords_test: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 15,
) -> np.ndarray:
    """Haversine KNN regressor. Returns predictions clipped to [0, 1]."""
    n_neighbors = min(n_neighbors, len(coords_train))
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors, weights="distance", metric="haversine",
    )
    model.fit(np.radians(coords_train), y_train.astype(float))
    return np.clip(model.predict(np.radians(coords_test)), 0.0, 1.0)


def predict_proportion_kriging(
    coords_train: np.ndarray,
    coords_test: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """Ordinary Kriging. Returns predictions clipped to [0, 1]."""
    from pykrige.ok import OrdinaryKriging

    kriging = OrdinaryKriging(
        coords_train[:, 1],
        coords_train[:, 0],
        y_train.astype(float),
        variogram_model="linear",
        coordinates_type="geographic",
        verbose=False,
        enable_plotting=False,
    )
    y_pred, _ = kriging.execute("points", coords_test[:, 1], coords_test[:, 0])
    return np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
