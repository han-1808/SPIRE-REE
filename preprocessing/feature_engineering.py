"""
preprocessing/feature_engineering.py

Builds the GNN-ready inputs (X, y, edge_index) from the processed REE
deposit DataFrame.

Call order in pipeline:
  pre_drop_feats = extract_pre_drop_features(raw_df)   # before drop_unused_columns
  ... (normal pipeline steps) ...
  df = attach_pre_drop_features(processed_df, pre_drop_feats)
  df = engineer_spatial_features(df)
  df = engineer_commodity_features(df)
  df = engineer_system_features(df)
  df = engineer_region_features(df)
  df = engineer_rec_type_features(df)
  X, y = build_gnn_inputs(df)
  edge_index = build_knn_graph(df[['Latitude', 'Longitude']].values)
  save_gnn_artifacts(X, y, edge_index)
"""

import re
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


from config import (
    COMMODITY_ELEMENTS,
    GNN_EDGE_INDEX_PATH,
    GNN_EDGE_WEIGHT_PATH,
    GNN_FEATURES_PATH,
    GNN_TARGET_PATH,
    KNN_K,
    REGION_CATEGORIES,
    REC_TYPE_CATEGORIES,
    SYSTEM_CLASS_REE_PATTERN,
    SYSTEM_CLASS_RULES,
    TOP_K_SHAP_FEATURES,
)

NOTEBOOK_DROP_COLUMNS = [
    "Host_Lith",
    "Dep_Type",
    "State_Prov",
    "Status",
    "Commods",
    "REE_Mins",
]


def create_spatial_split_indices(
    df: pd.DataFrame,
    test_region: str = "South and Central Asia",
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    regions = df["Region"].unique()
    if test_region not in regions:
        raise ValueError(f"{test_region} not in regions")

    test_idx_bool = df["Region"] == test_region
    train_idx_bool = ~test_idx_bool

    candidate_idx = np.where(train_idx_bool)[0]
    test_idx = np.where(test_idx_bool)[0]

    val_size = int(len(candidate_idx) * val_ratio)

    np.random.seed(random_seed)
    perm = np.random.permutation(candidate_idx)

    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def create_deposit_type_split_indices(
    df: pd.DataFrame,
    held_out_group: str,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Leave-one-deposit-type-out split (Comment (1)-C / LODTO), analogous to
    `create_spatial_split_indices` but held out by one of the 9 `is_*`
    deposit-type group columns instead of `Region`.

    The 9 groups are NOT mutually exclusive (a record can carry more than
    one `is_*` flag -- 300 records carry 2, 55 carry 3, ... out of 3112).
    So test_idx = every record with `held_out_group == 1`, regardless of
    its other group memberships, and train pool = every record with
    `held_out_group == 0` (a record belonging to the held-out group AND
    another group is still excluded from training entirely, since it does
    carry the held-out type). This means test sets across different calls
    (one per held-out group) can overlap -- expected given the underlying
    labels are multi-label, not a bug.
    """
    if held_out_group not in df.columns:
        raise ValueError(f"{held_out_group} not in df.columns")

    test_idx_bool = df[held_out_group] == 1
    if test_idx_bool.sum() == 0:
        raise ValueError(f"{held_out_group} has no positive records to hold out")
    train_idx_bool = ~test_idx_bool

    candidate_idx = np.where(train_idx_bool)[0]
    test_idx = np.where(test_idx_bool)[0]

    val_size = int(len(candidate_idx) * val_ratio)

    np.random.seed(random_seed)
    perm = np.random.permutation(candidate_idx)

    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


_EARTH_RADIUS_KM = 6371.0


def compute_mean_knn_distance_km(coords: np.ndarray, k: int = KNN_K) -> float:
    """
    Comment (4)-B: mean haversine distance (km) from every point to its k
    nearest neighbours, averaged over all points -- used as the default
    spatial-buffer radius for `apply_spatial_buffer` (no fixed radius is
    specified anywhere else in the pipeline, so this mirrors the graph's
    own k rather than picking an arbitrary number).
    """
    n_neighbors = min(k + 1, len(coords))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="haversine")
    nbrs.fit(np.radians(coords))
    dist_rad, _ = nbrs.kneighbors(np.radians(coords))
    dist_km = dist_rad[:, 1:] * _EARTH_RADIUS_KM  # drop self (column 0)
    return float(dist_km.mean())


def create_buffered_region_split_indices(
    region_global_idx: np.ndarray,
    coords_region: np.ndarray,
    test_ratio: float = 0.15,
    val_ratio: float = 0.2,
    buffer_km: float | None = None,
    k_for_buffer: int = KNN_K,
    random_seed: int = 42,
) -> dict:
    """
    Comment (4)-A/B: per-region split with a percentage-based test holdout
    (was a fixed n=5 in run_gnn_per_region.py / run_baseline_per_region.py)
    and spatial block / buffered CV (train/val candidates within `buffer_km`
    of any test node are excluded from the split entirely, not just
    re-labelled). Default `buffer_km` is the region's own mean k-NN distance
    (`compute_mean_knn_distance_km`), so no region uses an arbitrary fixed
    radius.

    Shared by run_gnn_per_region.py and run_baseline_per_region.py (which
    must produce the *same* split for a given seed -- the baseline script's
    own docstring says so -- so this is one function, not two independently
    maintained copies that could silently drift apart).

    If buffering would leave too few nodes for a usable train/val split
    (can happen in small/dense regions), falls back to no buffering for
    this region rather than crashing or training on 1-2 nodes -- reported
    in the returned `split_info["buffer_fallback"]`.

    Returns {"train_local", "val_local", "test_local", "split_info"} --
    indices all in local (region) space; `coords_region` must be in the
    same order as `region_global_idx` (i.e. `coords_region[i]` is the
    coordinate of `region_global_idx[i]`).
    """
    n = len(region_global_idx)
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(n)

    n_test = max(1, int(round(n * test_ratio)))
    test_local = perm[:n_test]
    rest = perm[n_test:]

    if buffer_km is None:
        buffer_km = compute_mean_knn_distance_km(coords_region, k=min(k_for_buffer, n - 1))

    keep_mask = apply_spatial_buffer(coords_region[rest], coords_region[test_local], buffer_km)
    rest_kept = rest[keep_mask]
    n_excluded = int(len(rest) - len(rest_kept))

    buffer_fallback = False
    min_needed = max(4, n_test)  # need at least a few train + a few val nodes
    if len(rest_kept) < min_needed:
        buffer_fallback = True
        rest_kept = rest

    n_val = max(1, int(len(rest_kept) * val_ratio))
    val_local = rest_kept[:n_val]
    train_local = rest_kept[n_val:]

    split_info = {
        "test_ratio": test_ratio,
        "n_test": int(n_test),
        "buffer_km": float(buffer_km),
        "n_excluded_by_buffer": 0 if buffer_fallback else n_excluded,
        "buffer_fallback": buffer_fallback,
    }
    return {
        "train_local": train_local,
        "val_local": val_local,
        "test_local": test_local,
        "split_info": split_info,
    }


def create_buffered_sample_split_indices(
    processed_df: pd.DataFrame,
    region: str,
    rng: np.random.Generator,
    test_ratio: float = 0.15,
    val_ratio: float = 0.2,
    buffer_km: float | None = None,
    k_for_buffer: int = KNN_K,
    val_seed: int = 42,
) -> dict:
    """
    Comment (4)-A/B: global-inductive-sample split (run_inductive_sample.py /
    run_exp6_proportion_inductive_sample.py's `_make_sample_split`) -- test
    = `test_ratio` of one region's own nodes (was a fixed n_sample=5),
    train+val = every OTHER node in the ENTIRE dataset (all regions, unlike
    `create_buffered_region_split_indices`'s per-region-only train pool).
    Train/val candidates within `buffer_km` of a test node are excluded
    (Comment 4-B) -- since test nodes all come from one region but
    candidates span the whole dataset, only nodes from that region (or a
    geographically adjacent one) can realistically fall within a
    few-hundred-km buffer; distant regions' candidates pass trivially.

    `rng` is a shared `np.random.Generator` the caller advances across
    regions within one seed (matching the existing call pattern in
    run_inductive_sample.py/run_exp6, which builds one `rng` per seed and
    calls this once per region in a loop) -- not re-seeded here.

    Returns {"train_idx", "val_idx", "test_idx", "n", "split_info"}.
    """
    region_idx = np.where(processed_df["Region"].values == region)[0]
    n = len(region_idx)
    n_test = max(1, min(n, int(round(n * test_ratio))))
    test_idx = np.sort(rng.choice(region_idx, size=n_test, replace=False))

    all_idx = np.arange(len(processed_df))
    tv_idx = np.setdiff1d(all_idx, test_idx)

    coords_all = processed_df[["Latitude", "Longitude"]].to_numpy()
    if buffer_km is None:
        buffer_km = compute_mean_knn_distance_km(
            coords_all[region_idx], k=min(k_for_buffer, n - 1)
        )

    keep_mask = apply_spatial_buffer(coords_all[tv_idx], coords_all[test_idx], buffer_km)
    tv_kept = tv_idx[keep_mask]
    n_excluded = int(len(tv_idx) - len(tv_kept))

    buffer_fallback = False
    min_needed = max(4, n_test)
    if len(tv_kept) < min_needed:
        buffer_fallback = True
        tv_kept = tv_idx

    local_rng = np.random.default_rng(val_seed)
    tv_shuffled = tv_kept.copy()
    local_rng.shuffle(tv_shuffled)
    n_val = int(len(tv_shuffled) * val_ratio)
    val_idx = np.sort(tv_shuffled[:n_val])
    train_idx = np.sort(tv_shuffled[n_val:])

    split_info = {
        "test_ratio": test_ratio,
        "n_test": int(len(test_idx)),
        "buffer_km": float(buffer_km),
        "n_excluded_by_buffer": 0 if buffer_fallback else n_excluded,
        "buffer_fallback": buffer_fallback,
    }
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "n": int(len(test_idx)),
        "split_info": split_info,
    }


def apply_spatial_buffer(
    coords_candidates: np.ndarray,
    coords_test: np.ndarray,
    buffer_km: float,
) -> np.ndarray:
    """
    Comment (4)-B: spatial block / buffered CV. Returns a boolean mask
    (len(coords_candidates),) -- True for candidates whose nearest test-node
    distance exceeds `buffer_km` (safe to keep in train/val), False for
    candidates within the buffer of any test node (excluded entirely, not
    just re-labelled -- the point of buffered CV is that these nodes are too
    spatially close to a test node to give an honest measure of extrapolation).
    """
    nbrs = NearestNeighbors(n_neighbors=1, metric="haversine")
    nbrs.fit(np.radians(coords_test))
    dist_rad, _ = nbrs.kneighbors(np.radians(coords_candidates))
    dist_km = dist_rad[:, 0] * _EARTH_RADIUS_KM
    return dist_km > buffer_km


# ── Private helpers ──────────────────────────────────────────────────────────

def _classify_system_row(row: pd.Series) -> str:
    """REE-sanitised deposit system classification for a single row."""
    text = " ".join([
        str(row.get("Dep_Type", "") or ""),
        str(row.get("Dep_Note", "") or ""),
        str(row.get("Rec_Note", "") or ""),
    ])
    text_clean = re.sub(SYSTEM_CLASS_REE_PATTERN, "", text, flags=re.IGNORECASE).lower()

    for label, keywords in SYSTEM_CLASS_RULES:
        if any(kw in text_clean for kw in keywords):
            return label

    dep_type = str(row.get("Dep_Type", "") or "").strip().lower()
    if dep_type not in ("", "nan", "unknown"):
        return "other"
    return "unknown"


def _flag_commodity(series: pd.Series, element_pattern: str) -> pd.Series:
    """1 where element matches AND the entry does NOT also mention REE."""
    has_elem     = series.str.contains(element_pattern, case=False, na=False, regex=True)
    mentions_ree = series.str.contains(r"\bREE\b|rare earth", case=False, na=False, regex=True)
    return (has_elem & ~mentions_ree).astype(int)


# ── Pre-drop extraction ──────────────────────────────────────────────────────

def extract_pre_drop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from columns removed by drop_unused_columns().

    Must be called on the raw DataFrame BEFORE drop_unused_columns().
    Returns a narrow 4-column DataFrame positionally aligned with df.

    Captures:
      - is_composite_system : 1 if Components is not NaN
      - is_part_of_complex  : 1 if Part_of is not NaN
      - system_class        : REE-sanitised deposit classification string
      - rec_type_raw        : raw Rec_Type value (for one-hot encoding later)
    """
    return pd.DataFrame({
        "is_composite_system": df["Components"].notna().astype(int).values,
        "is_part_of_complex":  df["Part_of"].notna().astype(int).values,
        "system_class":        df.apply(_classify_system_row, axis=1).values,
        "rec_type_raw":        df["Rec_Type"].values,
    })


def attach_pre_drop_features(df: pd.DataFrame, pre_drop: pd.DataFrame) -> pd.DataFrame:
    """
    Re-attach the pre-drop features to the processed DataFrame.

    Uses positional alignment (reset_index) since no rows are added or
    removed between extract_pre_drop_features and this call.
    """
    original_index = df.index
    result = pd.concat(
        [df.reset_index(drop=True), pre_drop.reset_index(drop=True)],
        axis=1,
    )
    result.index = original_index
    return result


# ── Feature engineering steps ────────────────────────────────────────────────

def engineer_spatial_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """
    Standardise Latitude and Longitude into z-scored spatial features.

    Mirrors REE_detector.ipynb Step 5.1.
    Accepts an optional pre-fitted scaler for reproducibility in tests.
    """
    if df[["Latitude", "Longitude"]].isna().any().any():
        raise ValueError("Latitude or Longitude contains NaN — cannot build spatial features.")
    if scaler is None:
        scaler = StandardScaler()
    df[["lat_z", "lon_z"]] = scaler.fit_transform(df[["Latitude", "Longitude"]])
    return df


def engineer_commodity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary commodity-presence flags and a count feature.

    Flags are REE-safe: an element is only flagged 1 when the Commods entry
    contains it AND does NOT also mention REE/rare earth.
    Mirrors REE_detector.ipynb Step 5.7 (has_element_safe pattern).
    """
    for col_name, pattern in COMMODITY_ELEMENTS.items():
        df[col_name] = _flag_commodity(df["Commods"], pattern)
    df["commodity_count"] = df[list(COMMODITY_ELEMENTS.keys())].sum(axis=1)
    return df


def engineer_system_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode system_class into system_* indicator columns.

    Uses pd.Categorical with fixed categories so the column set is
    deterministic regardless of which classes appear in the data.
    system_class column is retained for inspection.
    """
    all_labels = [label for label, _ in SYSTEM_CLASS_RULES] + ["other", "unknown"]
    cat = pd.Categorical(df["system_class"], categories=all_labels)
    dummies = pd.get_dummies(cat, prefix="system")
    dummies.index = df.index
    return pd.concat([df, dummies], axis=1)


def engineer_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode Region into region_* indicator columns (11 columns).

    Uses pd.Categorical with REGION_CATEGORIES to guarantee a fixed
    column set across runs.
    """
    cat = pd.Categorical(df["Region"], categories=REGION_CATEGORIES)
    dummies = pd.get_dummies(cat, prefix="region")
    dummies.index = df.index
    return pd.concat([df, dummies], axis=1)


def engineer_rec_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode rec_type_raw into rec_type_* indicator columns (6 columns).

    Uses pd.Categorical with REC_TYPE_CATEGORIES to guarantee a fixed
    column set across runs.
    """
    cat = pd.Categorical(df["rec_type_raw"], categories=REC_TYPE_CATEGORIES)
    dummies = pd.get_dummies(cat, prefix="rec_type")
    dummies.index = df.index
    return pd.concat([df, dummies], axis=1)


# ── GNN input assembly ───────────────────────────────────────────────────────

def build_gnn_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Assemble the final feature matrix X and target vector y.

    Feature columns (~227 total):
      - lat_z, lon_z                              (2)
      - is_composite_system, is_part_of_complex   (2)
      - has_Nb/Ta/Th/U/P/F/Zr, commodity_count   (8)
      - region_* one-hot                          (11)
      - rec_type_* one-hot                        (6)
      - system_* one-hot                          (9, excluding system_unknown)
      - has_* mineral features (non-REE)          (180)
      - is_* lithology group features             (9)

    num_groups is excluded (linear combination of is_* columns).
    REE_Mins must NOT be present in df (leakage guard).

    Returns
    -------
    X : pd.DataFrame  shape (N, F) float32
    y : np.ndarray    shape (N,)   int8
    """
    df.drop(["REE_Mins"], axis=1, inplace=True, errors="ignore")
    if "REE_Mins" in df.columns:
        raise ValueError(
            "REE_Mins is still present in df — remove it before building X to prevent leakage."
        )

    spatial_cols      = ["lat_z", "lon_z"]
    system_hier_cols  = ["is_composite_system", "is_part_of_complex"]
    commodity_cols    = list(COMMODITY_ELEMENTS.keys()) + ["commodity_count"]
    region_cols       = [f"region_{c}" for c in REGION_CATEGORIES]
    rec_type_cols     = [f"rec_type_{c}" for c in REC_TYPE_CATEGORIES]
    system_class_cols = [f"system_{label}" for label, _ in SYSTEM_CLASS_RULES] + ["system_other"]
    has_mineral_cols  = sorted(c for c in df.columns if c.startswith("has_") and c != "has_ree")
    is_lith_cols      = sorted(c for c in df.columns if c.startswith("is_") and c not in system_hier_cols)

    feature_cols = (
        spatial_cols
        + system_hier_cols
        + commodity_cols
        + region_cols
        + rec_type_cols
        + system_class_cols
        + has_mineral_cols
        + is_lith_cols
    )

    # Warn on any potential leakage column
    ree_cols = [c for c in feature_cols if "ree" in c.lower()]
    if ree_cols:
        warnings.warn(f"Potential leakage columns in feature set: {ree_cols}", stacklevel=2)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Expected feature columns missing from df: {missing}")

    X = df[feature_cols].fillna(0).astype("float32")
    y = df["has_ree"].to_numpy(dtype=np.int8)
    return X, y


# ── Graph construction ───────────────────────────────────────────────────────

def build_knn_graph(coords: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """
    Build a symmetric k-nearest-neighbour spatial graph using haversine distance.

    Mirrors REE_detector.ipynb Step 7 exactly.

    Parameters
    ----------
    coords : np.ndarray  shape (N, 2)  [Latitude, Longitude] in decimal degrees
    k      : int         number of neighbours per node (default KNN_K)

    Returns
    -------
    edge_index : np.ndarray  shape (2, E)  int32, bidirectional
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="haversine")
    nbrs.fit(np.radians(coords))
    _, indices = nbrs.kneighbors(np.radians(coords))

    edge_list = [
        (i, int(j))
        for i, neighbours in enumerate(indices)
        for j in neighbours[1:]  # skip self-loop
    ]

    edge_arr = np.array(list(set(edge_list)), dtype=np.int32).T  # (2, E_uniq)
    edge_index = np.concatenate([edge_arr, edge_arr[::-1]], axis=1)  # bidirectional
    print(f"Graph built: {coords.shape[0]} nodes, {edge_index.shape[1]} edges (k={k})")
    return edge_index


def prepare_xgboost_inputs(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    missing_threshold: float = 0.5,
    drop_coordinate_features: bool = False,
    fit_idx: np.ndarray | list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Reproduce the XGBoost input-preparation flow from XGBoost+GNN.ipynb.

    Steps mirrored from the notebook:
      1. Drop notebook-specific leakage/context columns
      2. Standardise Latitude/Longitude in place
      3. Remove columns with >= 50% missing values
      4. Split target y and features X
      5. Impute missing values and one-hot encode categorical columns
    """
    prepared_df = df.drop(columns=NOTEBOOK_DROP_COLUMNS, errors="ignore").copy()
    if fit_idx is None:
        fit_idx = np.arange(len(prepared_df))
    fit_idx = np.asarray(fit_idx, dtype=int)
    fit_df = prepared_df.iloc[fit_idx]

    if prepared_df[["Latitude", "Longitude"]].isna().any().any():
        raise ValueError("Latitude or Longitude contains NaN — cannot build spatial features.")

    if scaler is None:
        scaler = StandardScaler()
    scaler.fit(fit_df[["Latitude", "Longitude"]])
    prepared_df[["Latitude", "Longitude"]] = scaler.transform(prepared_df[["Latitude", "Longitude"]])

    missing_ratio = fit_df.isnull().mean()
    features_to_keep = missing_ratio[missing_ratio < missing_threshold].index
    if "has_ree" not in features_to_keep:
        features_to_keep = features_to_keep.insert(len(features_to_keep), "has_ree")
    if "Region" not in features_to_keep and "Region" in prepared_df.columns:
        features_to_keep = features_to_keep.insert(len(features_to_keep), "Region")
    prepared_df = prepared_df[features_to_keep].copy()

    if "has_ree" not in prepared_df.columns:
        raise KeyError("Expected target column 'has_ree' is missing from the prepared dataframe.")

    y = prepared_df["has_ree"].copy()
    X = prepared_df.drop(columns=["has_ree", "ID_No"], errors="ignore").copy()

    if drop_coordinate_features:
        X = X.drop(columns=["Latitude", "Longitude"], errors="ignore")

    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    if len(cat_cols) > 0:
        X.loc[:, cat_cols] = X[cat_cols].fillna("Unknown")
    if len(num_cols) > 0:
        train_medians = X.iloc[fit_idx][num_cols].median()
        X.loc[:, num_cols] = X[num_cols].fillna(train_medians)

    encoded_parts = [X[num_cols].copy()] if len(num_cols) > 0 else []
    for col in cat_cols:
        train_values = X.iloc[fit_idx][col].fillna("Unknown")
        categories = sorted(set(train_values.tolist()) | {"Unknown"})
        full_values = X[col].fillna("Unknown").where(X[col].isin(categories), "Unknown")
        cat = pd.Categorical(full_values, categories=categories)
        dummies = pd.get_dummies(cat, prefix=col)
        dummies.index = X.index
        encoded_parts.append(dummies)

    if encoded_parts:
        X = pd.concat(encoded_parts, axis=1)
    else:
        X = pd.DataFrame(index=X.index)
    return prepared_df, X, y


def train_xgboost_and_select_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = TOP_K_SHAP_FEATURES,
    fit_idx: np.ndarray | list[int] | None = None,
):
    """
    Train the notebook's XGBoost model, compute SHAP values, and keep top-k features.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Missing dependency 'shap'. Install it to run the XGBoost feature selector.") from exc

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'xgboost'. Install it to run the XGBoost feature selector."
        ) from exc

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    if fit_idx is None:
        fit_idx = np.arange(len(X))
    fit_idx = np.asarray(fit_idx, dtype=int)

    X_train = X.iloc[fit_idx]
    y_train = y.iloc[fit_idx]

    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[-1]

    important_features = np.abs(shap_array).mean(axis=0).argsort()[-top_k:]
    X_selected = X.iloc[:, important_features].copy()
    return model, shap_values, X_selected


def _make_ensemble_models(random_state: int = 42):
    """XGBoost + CatBoost + RandomForest, same hyperparameters used throughout."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Missing dependency 'xgboost'.") from exc
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Missing dependency 'catboost'.") from exc
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise ImportError("Missing dependency 'scikit-learn'.") from exc

    return [
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        ),
        CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            random_seed=random_state,
            verbose=False,
        ),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
    ]


def _make_single_xgb_model(random_state: int = 42):
    """
    Single XGBoost model, same hyperparameters as the XGBoost member of
    `_make_ensemble_models`. Used as a `models_factory` for
    `compute_ensemble_scores_no_leakage` when only one model's score is
    wanted (e.g. the tabular baselines' `xgb_score` feature — see
    `build_baseline_features_no_leakage`).
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Missing dependency 'xgboost'.") from exc

    return [
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        ),
    ]


def compute_ensemble_scores_no_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    other_idx: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    train_scoring: str = "oof",
    models_factory=_make_ensemble_models,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Model probability score with no train-node leakage (Comment 12 fix).
    Defaults to the 3-model (XGBoost+CatBoost+RandomForest) ensemble; pass
    `models_factory=_make_single_xgb_model` for a leakage-free single-model
    score instead (Comment 17: the tabular baselines' `xgb_score`).

    Fitting the ensemble on `train_idx` and then calling `predict_proba` on
    those same rows gives training nodes an in-sample, overfit score — this
    is what `build_node_features` does when called directly on `train_idx`.
    Here, training-node scores instead come from K-fold out-of-fold
    prediction: each fold's nodes are scored by models fit only on the
    other folds. `other_idx` (val/test) scores come from the 3 models
    refit on the FULL `train_idx` — that path was already leakage-free
    (validation/test nodes never appear in any training fold), so it is
    kept unchanged and the refit models are returned for reuse at
    downstream held-out inference time.

    Parameters
    ----------
    X, y       : full feature matrix / target (same ones passed to
                 train_xgboost_and_select_features).
    train_idx  : indices used for fitting.
    other_idx  : indices to score with the full-train-refit models
                 (typically val_idx, or val_idx + test_idx).
    n_folds    : number of OOF folds for the training-node scores.
    train_scoring : "oof" (default, leakage-free) or "in_sample" — reproduces
                 the pre-fix behaviour (train-node scores from the same
                 models fit on train_idx) so the two can be compared
                 side by side (Comment 12-C: OOF vs in-sample vs no-ensemble).
                 Only affects `train_scores`; `other_scores` is identical
                 either way (always the full-train refit).
    models_factory : callable(random_state) -> list[model]. Defaults to the
                 3-model (XGBoost+CatBoost+RandomForest) ensemble; pass
                 `_make_single_xgb_model` to reuse this same OOF machinery
                 for a single-model score — e.g. the tabular baselines'
                 `xgb_score` feature (Comment 17 fairness finding: baselines
                 had the same in-sample leak Comment 12 fixed for SPIRE,
                 just never patched).

    Returns
    -------
    train_scores  : np.ndarray, shape (len(train_idx),) — aligned to train_idx order.
    other_scores  : np.ndarray, shape (len(other_idx),) — aligned to other_idx order.
    fitted_models : [xgb_model, cat_model, rf_model] refit on the full train_idx.
    """
    if train_scoring not in ("oof", "in_sample"):
        raise ValueError(f"train_scoring must be 'oof' or 'in_sample', got {train_scoring!r}")

    from sklearn.model_selection import StratifiedKFold

    train_idx = np.asarray(train_idx, dtype=int)
    other_idx = np.asarray(other_idx, dtype=int)
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    oof_scores = None
    if train_scoring == "oof":
        # Small/imbalanced regions (more likely now that Comment (4)-B's
        # spatial buffer can shrink an already-small region's train pool
        # further) can leave a minority class with very few examples in
        # `train_idx`. `n_folds=5` StratifiedKFold on e.g. 1 minority
        # example puts it in exactly one fold's holdout, leaving that
        # fold's *training* set with zero examples of that class --
        # XGBoost's sklearn API raises `ValueError: Invalid classes
        # inferred...` when fit on a y containing only one class label
        # that isn't 0 (a real crash hit in practice, not hypothetical).
        # Guard by capping n_folds to the minority class count, and -- if
        # there are fewer than 2 minority examples, genuine OOF is
        # impossible (can't hold out a fold without emptying that class
        # from training) -- fall back to in-sample scoring for train
        # nodes only in that specific region/split, clearly logged.
        class_counts = y_train.value_counts()
        min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0

        if min_class_count < 2:
            print(f"    [WARN] compute_ensemble_scores_no_leakage: minority class has "
                  f"only {min_class_count} example(s) among {len(train_idx)} train nodes -- "
                  f"{n_folds}-fold OOF is not possible here, falling back to in-sample "
                  f"scoring for train nodes only (other_idx stays leakage-free as always).")
            oof_scores = None  # filled by the in-sample fallback below
            train_scoring = "in_sample"
        else:
            effective_n_folds = min(n_folds, min_class_count)
            if effective_n_folds < n_folds:
                print(f"    [WARN] compute_ensemble_scores_no_leakage: reducing OOF folds "
                      f"from {n_folds} to {effective_n_folds} (minority class has only "
                      f"{min_class_count} examples among {len(train_idx)} train nodes).")
            oof_scores = np.zeros(len(train_idx), dtype=float)
            kf = StratifiedKFold(n_splits=effective_n_folds, shuffle=True, random_state=random_state)
            for fold_train_pos, fold_holdout_pos in kf.split(X_train, y_train):
                fold_models = models_factory(random_state=random_state)
                X_fold_train = X_train.iloc[fold_train_pos]
                y_fold_train = y_train.iloc[fold_train_pos]
                X_fold_holdout = X_train.iloc[fold_holdout_pos]
                fold_probs = [
                    m.fit(X_fold_train, y_fold_train).predict_proba(X_fold_holdout)[:, 1]
                    for m in fold_models
                ]
                oof_scores[fold_holdout_pos] = np.mean(fold_probs, axis=0)

    fitted_models = models_factory(random_state=random_state)
    for m in fitted_models:
        m.fit(X_train, y_train)

    if train_scoring == "in_sample":
        # Pre-fix behaviour: score train_idx with the same models fit on
        # train_idx — reproduced here only for the Comment 12-C comparison.
        oof_scores = np.mean(
            [m.predict_proba(X_train)[:, 1] for m in fitted_models], axis=0
        )

    other_scores = np.mean(
        [m.predict_proba(X.iloc[other_idx])[:, 1] for m in fitted_models], axis=0
    )

    return oof_scores, other_scores, fitted_models


def build_node_features(X_selected: pd.DataFrame, X_full: pd.DataFrame, models):
    """
    Build GNN node features: SHAP-selected features + ensemble probability score.
    Returns a torch.Tensor of shape (N, top_k+1).

    `models` may be a single sklearn model or a list of models; when a list is
    given the score is the mean probability across all models (ensemble).
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Missing dependency 'torch'. Install it to create GNN node features.") from exc

    if not isinstance(models, (list, tuple)):
        models = [models]
    scores = np.mean([m.predict_proba(X_full)[:, 1] for m in models], axis=0)
    X_num = torch.tensor(X_selected.astype(float).values, dtype=torch.float)
    score_t = torch.tensor(scores, dtype=torch.float).unsqueeze(1)
    return torch.cat([X_num, score_t], dim=1)


def build_node_features_from_scores(X_selected: pd.DataFrame, scores: np.ndarray):
    """
    Like `build_node_features`, but takes an already-computed score array
    instead of models to call `predict_proba` on — used with
    `compute_ensemble_scores_no_leakage` (Comment 12), where training-node
    and val/test-node scores come from two different (both leakage-free)
    procedures and are assembled by the caller before reaching here.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Missing dependency 'torch'. Install it to create GNN node features.") from exc

    X_num = torch.tensor(X_selected.astype(float).values, dtype=torch.float)
    score_t = torch.tensor(np.asarray(scores, dtype=float), dtype=torch.float).unsqueeze(1)
    return torch.cat([X_num, score_t], dim=1)


def build_node_features_no_leakage(
    X_selected_tv: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    new_train: np.ndarray,
    new_val: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    train_scoring: str = "oof",
):
    """
    Drop-in, leakage-free replacement for the common call-site pattern
    `build_node_features(X_selected.iloc[tv_pos], X.iloc[tv_pos], ensemble_models)`
    where `ensemble_models` was fit on `train_idx` alone (Comment 12).

    Parameters
    ----------
    X_selected_tv : SHAP-selected feature columns, already sliced to `tv_pos`
                     order (i.e. `X_selected.iloc[tv_pos]`).
    X, y          : full feature matrix / target (original index space).
    train_idx, val_idx : original indices (not tv-local positions).
    new_train, new_val  : positions of train_idx / val_idx within tv_pos
                     (the `old_to_new` remapping every caller already builds).
    train_scoring : "oof" (default) or "in_sample" — see
                     `compute_ensemble_scores_no_leakage` (Comment 12-C).

    Returns
    -------
    X_node        : torch.Tensor, shape (len(tv_pos), X_selected_tv.shape[1] + 1).
    fitted_models  : [xgb_model, cat_model, rf_model] refit on the full
                     train_idx — reuse this (not a fresh fit) for any
                     downstream held-out (test) node inference.
    """
    train_scores, val_scores, fitted_models = compute_ensemble_scores_no_leakage(
        X, y, train_idx, val_idx, n_folds=n_folds, random_state=random_state,
        train_scoring=train_scoring,
    )
    scores_tv = np.empty(len(X_selected_tv), dtype=float)
    scores_tv[np.asarray(new_train, dtype=int)] = train_scores
    scores_tv[np.asarray(new_val, dtype=int)] = val_scores
    X_node = build_node_features_from_scores(X_selected_tv, scores_tv)
    return X_node, fitted_models


_GRAPH_SIM_EXCLUDE_COLS = {"Latitude", "Longitude"}


def select_geo_similarity_columns(X_selected: pd.DataFrame) -> pd.DataFrame:
    """
    Columns of a SHAP-selected node feature table to use for the
    cosine-similarity ("geological similarity") term in `build_weighted_graph`
    (Comment 13 fix).

    Drops Latitude/Longitude — SHAP top-k selection sometimes keeps them, but
    they're already used directly as the spatial edge-weight term (haversine
    distance on `coords`), so reusing them here would double-count spatial
    information into the "geological" edge weight. The ensemble_score column
    is never part of `X_selected` in the first place (it's appended
    separately by `build_node_features*`), so no extra handling is needed
    for it here — callers must build `X_geo` from `X_selected` (pre-score),
    never from the already-scored `X_node`.
    """
    geo_cols = [c for c in X_selected.columns if c not in _GRAPH_SIM_EXCLUDE_COLS]
    return X_selected[geo_cols]


def build_baseline_features(X_selected: pd.DataFrame, X_full: pd.DataFrame, model) -> pd.DataFrame:
    """
    Build baseline model features: SHAP-selected features + XGBoost probability score.
    Returns a pd.DataFrame of shape (N, top_k+1).
    """
    xgb_score = pd.Series(
        model.predict_proba(X_full)[:, 1],
        index=X_selected.index,
        name="xgb_score",
    )
    return pd.concat([X_selected, xgb_score], axis=1)


def build_weighted_graph(
    coords: np.ndarray,
    X_node,
    alpha: float = 0.6,
    k: int = KNN_K,
    weighting_strategy: str = "mixed",
    sigma: float | None = None,
    return_sigma: bool = False,
    max_distance_km: float | None = None,
    fit_idx: np.ndarray | None = None,
    X_geo=None,
    unify_edge_rule: bool = False,
):
    """
    Weighted k-NN spatial graph with optional adaptive-k filtering.

    Parameters
    ----------
    coords : np.ndarray  (N, 2)  Real [Latitude, Longitude] decimal degrees.
    X_node : torch.Tensor  (N, F)  node feature matrix (the actual GNN input;
             not used for cosine similarity when `X_geo` is given).
    alpha  : float  weight for spatial component in "mixed" strategy.
    k      : int    maximum number of spatial neighbours per node.
    weighting_strategy : "distance" | "feature_similarity" | "mixed"
    sigma  : float | None  bandwidth for Gaussian spatial weight; computed from
             mean k-NN distance if None.
    return_sigma : bool  if True, return (edge_index, edge_weight, sigma).
    max_distance_km : float | None  adaptive-k threshold — edges to neighbours
             farther than this are dropped.  The single closest neighbour is
             always kept as a fallback so no node becomes isolated.
    X_geo  : torch.Tensor | None  (N, F_geo)  node vector used ONLY for the
             cosine-similarity ("geological similarity") term (Comment 13
             fix). Coordinates are already used directly for the spatial
             term (`coords`), and `X_node` typically also carries the
             ensemble_score and (when SHAP selects them) Latitude/Longitude
             — reusing `X_node` for cosine similarity would double-count
             that information into the edge weight. Falls back to `X_node`
             (reproducing the pre-fix double-counting) when omitted, for
             backward compatibility with callers not yet updated.
    unify_edge_rule : bool  (Comment 14) when `fit_idx` is given, the default
             (False) applies the full feature_similarity/mixed formula only
             to edges where BOTH endpoints are in `fit_idx` (train-train);
             edges touching a non-fit (val) node fall back to spatial-only —
             the asymmetric "current" rule Comment 14 flags, kept as the
             default only for backward compatibility with callers not yet
             updated. `unify_edge_rule=True` applies the same formula to
             every edge regardless of `fit_idx` (Comment 14 Solution A,
             recommended) — safe because `w_geo` depends only on observable
             node features (x), never the target (y), so there is no
             leakage risk in using it on non-fit (val) edges too.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Missing dependency 'torch'. Install it to build the weighted graph.") from exc

    X_sim_source = X_geo if X_geo is not None else X_node

    start_time = perf_counter()
    n_nodes = len(coords)
    print(f"  [graph] N={n_nodes}, k={k}, strategy={weighting_strategy}")

    coords_rad = np.radians(coords)
    lat = coords_rad[:, 0:1]          # (N, 1)
    lon = coords_rad[:, 1:2]          # (N, 1)
    dlat = lat - lat.T                # (N, N)
    dlon = lon - lon.T                # (N, N)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    )
    dist_matrix = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # (N, N), in radians

    np.fill_diagonal(dist_matrix, np.inf)
    part = np.argpartition(dist_matrix, k, axis=1)[:, :k]   # (N, k) — unordered
    rows = np.arange(n_nodes)[:, None]
    order = np.argsort(dist_matrix[rows, part], axis=1)
    knn_idx = part[rows, order]                              # (N, k) sorted
    knn_dist = dist_matrix[rows, knn_idx]                   # (N, k) sorted

    if sigma is None:
        if fit_idx is not None:
            sigma = float(knn_dist[np.asarray(fit_idx, dtype=int)].mean())
        else:
            sigma = float(knn_dist.mean())
    src = np.repeat(np.arange(n_nodes, dtype=np.int64), k)
    dst = knn_idx.reshape(-1).astype(np.int64)
    distance_flat = knn_dist.reshape(-1)

    # ── Adaptive k: drop edges beyond max_distance_km ─────────────────────────
    if max_distance_km is not None:
        max_dist_rad = max_distance_km / 6371.0
        valid    = distance_flat <= max_dist_rad
        is_first = (np.arange(len(src)) % k) == 0  # closest neighbour per node
        keep     = valid | is_first                  # fallback: always keep closest
        dropped  = int((~keep).sum())
        src           = src[keep]
        dst           = dst[keep]
        distance_flat = distance_flat[keep]
        print(f"  [graph]   adaptive-k: dropped {dropped} far edges "
              f"(threshold={max_distance_km:.0f} km)")

    w_spatial = np.exp(-distance_flat / sigma)

    # Cosine similarity in pure NumPy — avoids PyTorch/OpenBLAS thread deadlock
    # that occurs when torch ops (F.normalize, torch.tensor) follow heavy numpy ops.
    X_node_np = X_sim_source.detach().to(dtype=torch.float32, device="cpu").numpy()
    norms = np.linalg.norm(X_node_np, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)   # in-place, avoids division by zero
    X_node_norm = X_node_np / norms

    n_edges = len(src)
    w_geo = (X_node_norm[src] * X_node_norm[dst]).sum(axis=1).clip(min=0.0)

    # Restrict feature-similarity to edges where both endpoints are fit
    # (train) nodes; edges touching a non-fit (val) node fall back to
    # spatial-only weighting — the asymmetric "current" rule (Comment 14),
    # applied only when `unify_edge_rule=False` (the backward-compatible
    # default). `unify_edge_rule=True` (Comment 14 Solution A) skips this
    # gating entirely and applies the same formula to every edge.
    apply_asymmetric_gate = fit_idx is not None and not unify_edge_rule
    if apply_asymmetric_gate:
        fit_set = np.zeros(n_nodes, dtype=bool)
        fit_set[np.asarray(fit_idx, dtype=int)] = True
        both_fit = fit_set[src] & fit_set[dst]

    if weighting_strategy == "distance":
        edge_weight = w_spatial
    elif weighting_strategy == "feature_similarity":
        if apply_asymmetric_gate:
            edge_weight = np.where(both_fit, w_geo, w_spatial)
        else:
            edge_weight = w_geo
    elif weighting_strategy == "mixed":
        if apply_asymmetric_gate:
            edge_weight = np.where(both_fit, alpha * w_spatial + (1 - alpha) * w_geo, w_spatial)
        else:
            edge_weight = alpha * w_spatial + (1 - alpha) * w_geo
    else:
        raise ValueError(
            "weighting_strategy must be one of: "
            "'distance', 'feature_similarity', 'mixed'"
        )
    print("  [graph]   5a done")

    print("  [graph]   5b: building bidirectional arrays...")
    edge_array = np.column_stack([src, dst])
    edge_array_bidirectional = np.concatenate([edge_array, edge_array[:, ::-1]], axis=0)
    edge_weight_bidirectional = np.concatenate([edge_weight, edge_weight], axis=0)
    print("  [graph]   5b done")

    # Use from_numpy (zero-copy, no thread spawn) instead of torch.tensor()
    # to avoid PyTorch/OpenBLAS thread deadlock after heavy numpy ops.
    print("  [graph]   5c: converting to torch tensors (from_numpy)...")
    edge_index = torch.from_numpy(
        np.ascontiguousarray(edge_array_bidirectional.T, dtype=np.int64)
    )
    edge_weight = torch.from_numpy(
        np.ascontiguousarray(edge_weight_bidirectional, dtype=np.float32)
    )
    print(f"Created weighted k-NN graph (k={k}, strategy={weighting_strategy})")
    print(f"  - Total edges: {edge_index.shape[1]}")
    print(f"  - Average degree: {edge_index.shape[1] / n_nodes:.1f}")
    print(f"  - Graph build time: {perf_counter() - start_time:.2f}s")
    if return_sigma:
        return edge_index, edge_weight, sigma
    return edge_index, edge_weight


# ── Artifact persistence ─────────────────────────────────────────────────────

def save_gnn_artifacts(
    X: pd.DataFrame,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None = None,
    features_path: str   = GNN_FEATURES_PATH,
    target_path: str     = GNN_TARGET_PATH,
    edge_index_path: str = GNN_EDGE_INDEX_PATH,
    edge_weight_path: str = GNN_EDGE_WEIGHT_PATH,
) -> None:
    """
    Persist GNN inputs to disk.

      X          → Parquet  (preserves column names and float32 dtype)
      y          → .npy
      edge_index → .npy
    """
    X.to_parquet(features_path)
    print(f"X saved: {X.shape} → {features_path}")

    np.save(target_path, y)
    print(f"y saved: {y.shape} → {target_path}")

    np.save(edge_index_path, edge_index)
    print(f"edge_index saved: {edge_index.shape} → {edge_index_path}")

    if edge_weight is not None:
        np.save(edge_weight_path, edge_weight)
        print(f"edge_weight saved: {edge_weight.shape} → {edge_weight_path}")
