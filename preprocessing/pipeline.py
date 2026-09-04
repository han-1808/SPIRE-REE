import pandas as pd
import torch
from time import perf_counter

from cleaning import drop_unused_columns
from config import (
    DATA_PATH,
    HOST_LITH_STATS_PATH,
    KNN_K,
    OUTPUT_PATH,
    TOP_K_SHAP_FEATURES,
)
from feature_engineering import (
    build_node_features,
    build_weighted_graph,
    create_spatial_split_indices,
    prepare_xgboost_inputs,
    save_gnn_artifacts,
    select_geo_similarity_columns,
    train_xgboost_and_select_features,
)
from host_lith import compute_host_lith_stats, create_lith_group_features, fill_host_lith
from mineral_processing import process_minerals


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_excel(path)


def main():
    pipeline_start = perf_counter()

    print("Loading data...")
    df = load_data()

    # Step 1: drop Antarctica records, then drop unused columns
    before = len(df)
    df = df[df["Region"] != "Antarctica"].reset_index(drop=True)
    print(f"Dropped {before - len(df)} Antarctica records ({len(df)} remaining)")
    print("Dropping unused columns...")
    df = drop_unused_columns(df)

    # Step 2: fill missing host lithology values
    print("Filling missing Host_Lith values...")
    df = fill_host_lith(df)

    # Step 3: process mineral text fields and mineral indicators
    print("Processing minerals...")
    df = process_minerals(df)

    # Step 4: compute host lithology statistics
    print("Computing Host_Lith statistics...")
    host_lith_stats = compute_host_lith_stats(df)
    host_lith_stats.to_excel(HOST_LITH_STATS_PATH, index=False)
    print(f"Host lithology stats saved to: {HOST_LITH_STATS_PATH}")

    # Step 5: create lithology group features and save the processed table
    print("Creating lithology group features...")
    df = create_lith_group_features(df)

    df = df.drop(["Name"], axis=1, errors="ignore")

    print(f"Saving processed output to: {OUTPUT_PATH}")
    df.to_excel(OUTPUT_PATH, index=False)

    default_test_region = "South and Central Asia"
    split_indices = create_spatial_split_indices(
        df,
        test_region=default_test_region,
        val_ratio=0.2,
    )

    # Step 6: prepare XGBoost-ready tabular inputs
    print("Preparing XGBoost inputs from the processed dataframe...")
    graph_df, X, y = prepare_xgboost_inputs(
        df,
        fit_idx=split_indices["train_idx"],
    )
    print(f"Prepared feature matrix: X={X.shape}, y={y.shape}")

    # Step 7: train XGBoost and keep the top SHAP-ranked features
    print(f"Training XGBoost and selecting top {TOP_K_SHAP_FEATURES} SHAP features...")
    model, _shap_values, X_selected = train_xgboost_and_select_features(
        X,
        y,
        top_k=TOP_K_SHAP_FEATURES,
        fit_idx=split_indices["train_idx"],
    )
    print(f"Selected features: {list(X_selected.columns)}")

    # Step 8: build node features from SHAP-selected columns plus xgb_score
    print("Building node features from selected columns + XGBoost score...")
    X_node = build_node_features(X_selected, X, model)
    print(f"Node feature shape: {tuple(X_node.shape)}")
    X_node_df = pd.DataFrame(
        X_node.cpu().numpy(),
        columns=[*X_selected.columns.tolist(), "xgb_score"],
        index=X_selected.index,
    )

    # Step 9: build the weighted spatial graph
    print(f"Building weighted spatial graph (k={KNN_K}, haversine + cosine similarity)...")
    graph_start = perf_counter()
    # Geo-only vector for cosine similarity (Comment 13): excludes
    # Latitude/Longitude and ensemble_score to avoid double-counting them
    # into the "geological similarity" edge weight.
    X_geo = torch.tensor(
        select_geo_similarity_columns(X_selected).astype(float).values, dtype=torch.float32,
    )
    edge_index, edge_weight = build_weighted_graph(
        df[["Latitude", "Longitude"]].values, X_node, k=KNN_K, X_geo=X_geo,
    )
    print(f"Weighted graph step finished in {perf_counter() - graph_start:.2f}s")

    # Step 10: save node, target, and graph artifacts for downstream GNN runs
    print("Saving GNN artifacts...")
    save_start = perf_counter()
    save_gnn_artifacts(
        X_node_df,
        y.to_numpy(),
        edge_index.cpu().numpy(),
        edge_weight=edge_weight.cpu().numpy(),
    )
    print(f"Artifact save step finished in {perf_counter() - save_start:.2f}s")

    print(f"Done preprocessing data. Total time: {perf_counter() - pipeline_start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
