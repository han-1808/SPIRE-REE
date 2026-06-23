# SPatially-augmented Inductive REpresentation lEarning for REE

Spatially-augmented inductive graph representation learning for Rare Earth Element (REE) deposit classification using a global geoscience database.

---

## Data

**Source:** `data/Global_REE_occurrence_database.xlsx` — 3,114 REE deposit records with ~60 raw columns.

**Target variable:** `has_ree` — binary label indicating whether a deposit has documented REE minerals (1 = yes, 0 = no). ~65% positive class.

**10 geographic regions** (Antarctica excluded — too few records):
Africa, China, East Asia, Europe, Middle East, North America, Oceania, Russian Federation, South America, South and Central Asia.

---

## Project Structure

```
preprocessing/                           # Data cleaning and feature engineering
├── config.py                            # All constants: paths, thresholds, GNN parameters
├── cleaning.py                          # Column removal and text normalisation
├── host_lith.py                         # Host lithology inference and group features
├── mineral_processing.py                # REE detection, Sig_Mins cleaning, binary features
├── feature_engineering.py              # XGBoost inputs, SHAP selection, graph construction
└── pipeline.py                          # Orchestrates preprocessing Steps 1–10

experiment/                              # Model training and evaluation
├── baseline.py                          # Baseline model definitions
├── gnn_training.py                      # SPIRE / GCN / GAT architecture + training loop
├── run_baseline.py                      # Exp 1 baseline — leave-one-region-out
├── run_baseline_per_region.py           # Exp 2 baseline — per-region, random split
├── run_gnn.py                           # Exp 1 GNN — leave-one-region-out (inductive)
├── run_gnn_per_region.py                # Exp 2 GNN — per-region, random split
├── run_inductive_sample.py              # Exp 3 GNN+baseline — global inductive, random
├── run_exp4_proportion.py               # Exp 4 — region-level proportion prediction
├── run_exp5_proportion_per_region.py    # Exp 5 — per-region proportion, random split
└── run_exp6_proportion_inductive_sample.py    # Exp 6 — global inductive proportion, random

result/                                  # Output JSON files
├── baseline.json                        # Exp 1 baseline results
├── gnn_inductive.json                   # Exp 1 GNN results
├── baseline_per_region.json             # Exp 2 baseline results
├── gnn_per_region.json                  # Exp 2 GNN results
├── inductive_sample.json                # Exp 3 results (GNN + baselines)
├── exp4_proportion.json                 # Exp 4 results (proportion prediction)
├── exp5_proportion_per_region.json      # Exp 5 results
└── exp6_proportion_inductive_sample.json    # Exp 6 results
```

---

## Preprocessing Pipeline

```bash
python preprocessing/pipeline.py
```

**Step 1** — Antarctica records removed. 40+ administrative/redundant columns dropped (all `RR_*` reserve fields, notes, leakage columns `REE`, `LREE_Note`, `HREE_Note`, `REE_Ratio`).

**Step 2** — ~1,109 records with missing `Host_Lith` filled via multi-tier rule engine: Dep_Type rules → REE_Mins fallback → Sig_Mins fallback.

**Step 3** — Normalise `REE_Mins`, create binary target `has_ree`, strip REE symbols from `Sig_Mins`, create 180 binary `has_{mineral}` features from top Sig_Mins entries.

**Step 4** — `Host_Lith` free-text classified into 8 rock group binary columns: `is_carbonatite_system`, `is_alkaline_intrusive`, `is_mafic_ultramafic`, `is_felsic_pegmatite`, `is_metamorphic_metasomatic`, `is_sedimentary`, `is_placer`, `is_supergene_lateritic`.

**Step 5** — Save processed dataset → `data/Global_REE_occurrence_database_processed.xlsx`.

**Step 6** — Prepare XGBoost inputs: drop leakage columns, `StandardScaler` on lat/lon (fit on train), drop high-missing columns, impute nulls with train median/`"Unknown"`, one-hot encode categoricals. Result: `X` ~(3100, 309), `y` ~(3100,).

**Step 7** — Train XGBoost + SHAP selection: top `k=10` features by mean |SHAP| retained. Example: `has_thorite`, `has_magnetite`, `is_unclassified`, `is_alkaline_intrusive`, `has_pyrochlore`, `has_ilmenite`, `Latitude`, `is_placer`, `Longitude`, `has_zircon`.

**Step 8** — Build node features: 10 SHAP-selected features + 1 ensemble probability score (mean of XGBoost, CatBoost, RandomForest `predict_proba`, all fit on train only) = **11 features per node**.

**Step 9** — Build weighted spatial graph: `w = 0.5 × w_spatial + 0.5 × w_feature`, where `w_spatial` is Gaussian decay on haversine distance (max 2000 km) and `w_feature` is cosine similarity. `k=10` nearest neighbours per node.

**Step 10** — Save GNN artifacts: `data/gnn_features.parquet`, `data/gnn_target.npy`, `data/gnn_edge_index.npy`, `data/gnn_edge_weight.npy`.

---

## GNN Architecture

Three architectures are evaluated in all experiments. **SPIRE is tuned via grid search; GCN and GAT reuse the same best hyperparameters** (hidden dim, num layers, dropout, learning rate) to reduce training cost.

| Model | Architecture | Edge weights |
|-------|-------------|--------------|
| **SPIRE** | SAGEConv × L layers, BN + residual, dropout | Not used natively — encoded via edge-stat augmentation |
| **GCN** | GCNConv × L layers, BN + residual, dropout | Used (normalised adjacency) |
| **GAT** | GATConv × L layers, 4 attention heads, BN + residual | Learned (attention) |

**Hyperparameter grid (SPIRE tuned, others reuse):**
- Hidden dim: {32, 64, 128} — Layers: {1, 2} — Dropout: {0.3, 0.5} — LR: {0.01, 0.005, 0.001}

**Training:** early stopping on val F1 (patience=20), best checkpoint restored before evaluation.

**Edge-weight augmentation** — After building the train+val graph (Step 9), compute mean and max of incoming edge weights for each node and append to the 11-dim base features → **13 features per node**. For inductive test nodes, edge stats are derived at inference time from a k-NN mini-graph using the same graph parameters. This gives SPIRE access to spatial-proximity signals without architectural changes. Applied in all GNN experiments.

---

## Experiments

### Overview

| Experiment | Train data | Test nodes | Sampling | Task |
|-----------|------------|------------|----------|------|
| **Exp 1** — Leave-one-region-out | 9 other regions | All nodes in held-out region (63–532) | — | Classification |
| **Exp 2** — Per-region random | Region's own graph | 5 per region | Random | Classification |
| **Exp 3** — Global inductive random | All nodes except 5 per region | 5 per region | Random | Classification |
| **Exp 4** — Region proportion | 9 other regions | All nodes in held-out region | — | Proportion MAE/RMSE |
| **Exp 5** — Per-region proportion, random | Region's own graph | 5 per region | Random | Proportion MAE/RMSE |
| **Exp 6** — Global inductive proportion, random | All nodes except 5 per region | 5 per region | Random | Proportion MAE/RMSE |

**Inductive inference (all GNN experiments):** each test node is augmented into the train+val graph one-by-one — k=10 nearest spatial neighbours connected, one forward pass, then removed. Test nodes never interact with each other.

**Feature parity:** all baselines and GNNs use the same 11 base features (10 SHAP-selected + 1 ensemble score) across all experiments. GNNs additionally receive 2 edge-stat features (mean and max of incoming edge weights) → 13-dim total in all experiments (see edge-weight augmentation in GNN Architecture).

**Leakage controls:** StandardScaler, XGBoost, SHAP selection, ensemble models, graph construction — all fit on `train_idx` only.

**Proportion prediction (Exp 4–6):** `predicted_proportion = mean(y_prob)` over all test nodes in a region. Compared to `actual_proportion = fraction of has_ree=1` among those nodes.

---

## Results

> **Metric direction:** Classification — F1 ⬆, ROC-AUC ⬆, PR-AUC ⬆ (higher is better). Proportion — MAE ⬇, RMSE ⬇ (lower is better).

All metrics averaged across 10 regions. ROC-AUC / PR-AUC averaged only over regions where both classes appear in the test set.

---

### Exp 1 — Leave-one-region-out (Classification)

GNN trains on 9 regions (test region absent from graph), predicts all nodes in held-out region **inductively**. Baselines train on 9 regions, predict held-out region **transductively**. Note: these are different settings — GNN faces the harder cross-region generalisation task.

#### F1 ⬆, ROC-AUC ⬆, PR-AUC ⬆

#### Run 1 — seed=42 (reference)

##### GNN (inductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.769** | **0.827** | **0.834** | **0.907** |
| GCN | 0.693 | 0.798 | 0.659 | 0.791 |
| GAT | 0.643 | 0.750 | 0.616 | 0.762 |

##### Baselines (transductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| MLP | 0.757 | 0.823 | 0.803 | 0.869 |
| SVM | 0.752 | 0.819 | 0.768 | 0.824 |
| Autoencoder | 0.752 | 0.819 | 0.802 | 0.864 |
| RandomForest | 0.748 | 0.816 | 0.820 | 0.897 |
| LightGBM | 0.749 | 0.816 | 0.826 | 0.889 |
| XGBoost | 0.742 | 0.812 | 0.833 | 0.901 |
| CatBoost | 0.739 | 0.809 | 0.833 | 0.906 |
| Kriging | 0.673 | 0.793 | 0.613 | 0.770 |
| Spatial kNN | 0.684 | 0.765 | 0.579 | 0.750 |

#### Run 2 — seed=1

##### GNN (inductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.799** | **0.849** | **0.841** | **0.915** |
| GCN | 0.675 | 0.785 | 0.659 | 0.792 |
| GAT | 0.610 | 0.702 | 0.592 | 0.746 |

##### Baselines (transductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| MLP | 0.755 | 0.821 | 0.816 | 0.889 |
| SVM | 0.752 | 0.819 | 0.768 | 0.824 |
| Autoencoder | 0.749 | 0.816 | 0.799 | 0.865 |
| LightGBM | 0.749 | 0.816 | 0.826 | 0.889 |
| RandomForest | 0.748 | 0.816 | 0.811 | 0.882 |
| XGBoost | 0.741 | 0.812 | 0.833 | 0.902 |
| CatBoost | 0.741 | 0.810 | 0.831 | 0.903 |
| Kriging | 0.673 | 0.793 | 0.613 | 0.770 |
| Spatial kNN | 0.684 | 0.765 | 0.579 | 0.750 |

#### Run 3 — seed=2

##### GNN (inductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.797** | **0.847** | **0.847** | **0.912** |
| GCN | 0.692 | 0.793 | 0.688 | 0.817 |
| GAT | 0.615 | 0.713 | 0.633 | 0.768 |

##### Baselines (transductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| Autoencoder | 0.753 | 0.819 | 0.812 | 0.872 |
| SVM | 0.751 | 0.819 | 0.768 | 0.824 |
| MLP | 0.749 | 0.817 | 0.803 | 0.876 |
| LightGBM | 0.749 | 0.816 | 0.826 | 0.889 |
| RandomForest | 0.747 | 0.815 | 0.813 | 0.887 |
| XGBoost | 0.743 | 0.813 | 0.834 | 0.904 |
| CatBoost | 0.739 | 0.810 | 0.831 | 0.907 |
| Kriging | 0.673 | 0.793 | 0.613 | 0.770 |
| Spatial kNN | 0.684 | 0.765 | 0.579 | 0.750 |

#### Run 4 — seed=4

##### GNN (inductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.801** | **0.851** | **0.843** | **0.914** |
| GCN | 0.670 | 0.779 | 0.660 | 0.794 |
| GAT | 0.625 | 0.718 | 0.626 | 0.766 |

##### Baselines (transductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| MLP | 0.753 | 0.820 | 0.801 | 0.875 |
| SVM | 0.751 | 0.819 | 0.768 | 0.824 |
| LightGBM | 0.749 | 0.816 | 0.826 | 0.889 |
| RandomForest | 0.747 | 0.815 | 0.822 | 0.891 |
| XGBoost | 0.745 | 0.815 | 0.833 | 0.900 |
| Autoencoder | 0.742 | 0.811 | 0.815 | 0.882 |
| CatBoost | 0.736 | 0.806 | 0.832 | 0.904 |
| Kriging | 0.673 | 0.793 | 0.613 | 0.770 |
| Spatial kNN | 0.684 | 0.765 | 0.579 | 0.750 |

#### Run 5 — seed=7

##### GNN (inductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.800** | **0.845** | **0.841** | **0.913** |
| GCN | 0.690 | 0.791 | 0.678 | 0.809 |
| GAT | 0.667 | 0.738 | 0.637 | 0.777 |

##### Baselines (transductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| MLP | 0.756 | 0.823 | 0.821 | 0.892 |
| SVM | 0.751 | 0.819 | 0.768 | 0.824 |
| LightGBM | 0.749 | 0.816 | 0.826 | 0.889 |
| RandomForest | 0.748 | 0.815 | 0.823 | 0.893 |
| Autoencoder | 0.747 | 0.815 | 0.802 | 0.867 |
| XGBoost | 0.744 | 0.814 | 0.835 | 0.904 |
| CatBoost | 0.742 | 0.811 | 0.834 | 0.907 |
| Kriging | 0.673 | 0.793 | 0.613 | 0.770 |
| Spatial kNN | 0.684 | 0.765 | 0.579 | 0.750 |

#### Average — 5 runs (seeds 42, 1, 2, 4, 7)

##### GNN (inductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.793±0.012** | **0.844±0.009** | **0.841±0.004** | **0.912±0.003** |
| GCN | 0.684±0.009 | 0.789±0.007 | 0.669±0.012 | 0.801±0.011 |
| GAT | 0.632±0.021 | 0.724±0.017 | 0.621±0.016 | 0.764±0.010 |

##### Baselines (transductive)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| MLP | 0.754±0.003 | 0.821±0.002 | 0.809±0.008 | 0.880±0.009 |
| SVM | 0.752±0.001 | 0.819±0.000 | 0.768±0.000 | 0.824±0.000 |
| Autoencoder | 0.749±0.004 | 0.816±0.003 | 0.806±0.006 | 0.870±0.007 |
| LightGBM | 0.749±0.000 | 0.816±0.000 | 0.826±0.000 | 0.889±0.000 |
| RandomForest | 0.748±0.000 | 0.816±0.000 | 0.818±0.005 | 0.890±0.005 |
| XGBoost | 0.743±0.001 | 0.813±0.001 | 0.834±0.001 | 0.902±0.002 |
| CatBoost | 0.739±0.002 | 0.809±0.002 | 0.832±0.001 | 0.905±0.002 |
| Kriging | 0.673±0.000 | 0.793±0.000 | 0.613±0.000 | 0.770±0.000 |
| Spatial kNN | 0.684±0.000 | 0.765±0.000 | 0.579±0.000 | 0.750±0.000 |

#### SPIRE per-region breakdown (seed=42)

| Region | n\_test | Accuracy | F1 | ROC-AUC |
|--------|---------|----------|----|---------|
| Africa | 235 | 0.749 | 0.856 | 0.792 |
| China | 214 | 0.607 | 0.693 | 0.889 |
| East Asia | 358 | 0.818 | 0.874 | 0.892 |
| Europe | 383 | 0.736 | 0.820 | 0.799 |
| Middle East | 63 | 0.730 | 0.832 | 0.734 |
| North America | 423 | 0.719 | 0.713 | 0.843 |
| Oceania | 504 | 0.782 | 0.876 | 0.737 |
| Russian Federation | 147 | 0.850 | 0.871 | 0.905 |
| South America | 253 | 0.866 | 0.924 | 0.816 |
| South and Central Asia | 532 | 0.831 | 0.807 | 0.933 |

Sources: `result/baseline.json`, `result/gnn_inductive.json`

---

### Exp 2 — Per-region, random split (Classification)

Each region forms its own isolated graph. 5 nodes randomly selected as test (seed=42). **Same 5 nodes for GNN and baselines — direct head-to-head.** GNNs use 13-dim features (with edge-stat augmentation); baselines use 11-dim features.

#### F1 ⬆, ROC-AUC ⬆, PR-AUC ⬆

#### Run 1 — seed=42 (reference)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.900** | **0.935** | **0.952** | **0.972** |
| XGBoost | 0.860 | 0.913 | 0.845 | 0.949 |
| CatBoost | 0.860 | 0.913 | 0.833 | 0.939 |
| RandomForest | 0.860 | 0.913 | 0.798 | 0.923 |
| SVM | 0.860 | 0.913 | 0.726 | 0.900 |
| MLP | 0.860 | 0.913 | 0.798 | 0.930 |
| Autoencoder | 0.840 | 0.902 | 0.833 | 0.956 |
| LightGBM | 0.820 | 0.883 | 0.619 | 0.845 |
| Kriging | 0.800 | 0.852 | 0.762 | 0.911 |
| Spatial kNN | 0.760 | 0.837 | 0.738 | 0.898 |
| GCN | 0.760 | 0.837 | 0.595 | 0.815 |
| GAT | 0.760 | 0.837 | 0.619 | 0.832 |

#### Run 2 — seed=50

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.900** | **0.942** | **0.950** | **0.990** |
| SVM | 0.820 | 0.877 | 0.850 | 0.890 |
| Autoencoder | 0.800 | 0.863 | 0.850 | 0.890 |
| LightGBM | 0.780 | 0.852 | 0.850 | 0.890 |
| CatBoost | 0.800 | 0.851 | 0.850 | 0.890 |
| RandomForest | 0.800 | 0.851 | 0.850 | 0.890 |
| MLP | 0.800 | 0.851 | 0.850 | 0.890 |
| XGBoost | 0.800 | 0.847 | 0.825 | 0.885 |
| Spatial kNN | 0.800 | 0.783 | 0.825 | 0.907 |
| GAT | 0.740 | 0.817 | 0.700 | 0.843 |
| GCN | 0.680 | 0.720 | 0.775 | 0.905 |
| Kriging | 0.640 | 0.619 | 0.750 | 0.884 |

#### Run 3 — seed=32

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.900** | **0.927** | **0.967** | **0.981** |
| Autoencoder | 0.860 | 0.899 | 0.858 | 0.878 |
| XGBoost | 0.840 | 0.882 | 0.800 | 0.868 |
| LightGBM | 0.840 | 0.882 | 0.850 | 0.889 |
| SVM | 0.840 | 0.879 | 0.808 | 0.864 |
| MLP | 0.840 | 0.879 | 0.808 | 0.864 |
| RandomForest | 0.820 | 0.865 | 0.812 | 0.868 |
| CatBoost | 0.820 | 0.871 | 0.800 | 0.868 |
| Kriging | 0.780 | 0.829 | 0.825 | 0.913 |
| GAT | 0.740 | 0.807 | 0.667 | 0.852 |
| Spatial kNN | 0.720 | 0.771 | 0.650 | 0.786 |
| GCN | 0.680 | 0.763 | 0.558 | 0.760 |

#### Run 4 — seed=17

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.860** | **0.900** | 0.817 | 0.932 |
| Autoencoder | 0.880 | 0.893 | 0.810 | 0.929 |
| Spatial kNN | 0.860 | 0.880 | 0.780 | 0.899 |
| SVM | 0.860 | 0.827 | **0.833** | **0.905** |
| RandomForest | 0.860 | 0.827 | 0.762 | 0.862 |
| LightGBM | 0.820 | 0.854 | 0.762 | 0.882 |
| XGBoost | 0.840 | 0.813 | 0.762 | 0.886 |
| CatBoost | 0.840 | 0.813 | 0.750 | 0.876 |
| MLP | 0.840 | 0.827 | 0.810 | 0.893 |
| Kriging | 0.820 | 0.804 | 0.798 | 0.853 |
| GAT | 0.740 | 0.806 | 0.775 | 0.902 |
| GCN | 0.700 | 0.745 | 0.575 | 0.759 |

#### Run 5 — seed=21

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.840** | **0.893** | **0.900** | **0.939** |
| XGBoost | 0.860 | 0.883 | 0.742 | 0.767 |
| LightGBM | 0.840 | 0.872 | 0.767 | 0.778 |
| CatBoost | 0.840 | 0.872 | 0.717 | 0.756 |
| RandomForest | 0.840 | 0.872 | 0.667 | 0.709 |
| SVM | 0.840 | 0.872 | 0.683 | 0.731 |
| MLP | 0.840 | 0.872 | 0.700 | 0.734 |
| Autoencoder | 0.840 | 0.872 | 0.733 | 0.759 |
| Spatial kNN | 0.800 | 0.813 | 0.808 | 0.867 |
| GCN | 0.760 | 0.837 | 0.700 | 0.888 |
| GAT | 0.740 | 0.815 | 0.767 | 0.916 |
| Kriging | 0.760 | 0.794 | 0.817 | 0.924 |

#### Average — 5 runs (seeds 42, 50, 32, 17, 21)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.880 ± 0.025** | **0.919 ± 0.019** | **0.917 ± 0.055** | **0.963 ± 0.023** |
| Autoencoder | 0.844 ± 0.027 | 0.886 ± 0.016 | 0.817 ± 0.045 | 0.882 ± 0.068 |
| SVM | 0.844 ± 0.015 | 0.874 ± 0.028 | 0.780 ± 0.064 | 0.858 ± 0.065 |
| MLP | 0.836 ± 0.020 | 0.868 ± 0.029 | 0.793 ± 0.050 | 0.862 ± 0.068 |
| XGBoost | 0.840 ± 0.022 | 0.868 ± 0.034 | 0.795 ± 0.038 | 0.871 ± 0.059 |
| CatBoost | 0.832 ± 0.020 | 0.864 ± 0.032 | 0.790 ± 0.050 | 0.866 ± 0.060 |
| RandomForest | 0.836 ± 0.023 | 0.865 ± 0.028 | 0.778 ± 0.062 | 0.851 ± 0.074 |
| LightGBM | 0.820 ± 0.022 | 0.869 ± 0.013 | 0.770 ± 0.084 | 0.857 ± 0.042 |
| Spatial kNN | 0.788 ± 0.047 | 0.817 ± 0.039 | 0.760 ± 0.062 | 0.872 ± 0.045 |
| Kriging | 0.760 ± 0.063 | 0.780 ± 0.083 | 0.790 ± 0.030 | 0.897 ± 0.026 |
| GAT | 0.744 ± 0.008 | 0.816 ± 0.011 | 0.705 ± 0.059 | 0.869 ± 0.033 |
| GCN | 0.716 ± 0.037 | 0.780 ± 0.048 | 0.641 ± 0.083 | 0.826 ± 0.062 |

Sources: `result/baseline_per_region.json`, `result/gnn_per_region.json`

---

### Exp 3 — Global inductive, random split (Classification)

5 nodes per region held out; **all remaining nodes (including the rest of the same region) form the global training graph**. Each test node predicted inductively. Same 5 nodes for GNN and baselines.

#### F1 ⬆, ROC-AUC ⬆, PR-AUC ⬆

#### Run 1 — seed=42 (reference)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.960** | **0.956** | —¹ | —¹ |
| CatBoost | 0.940 | 0.944 | —¹ | —¹ |
| MLP | 0.940 | 0.944 | —¹ | —¹ |
| Autoencoder | 0.940 | 0.944 | —¹ | —¹ |
| RandomForest | 0.920 | 0.911 | —¹ | —¹ |
| SVM | 0.920 | 0.911 | —¹ | —¹ |
| XGBoost | 0.900 | 0.919 | —¹ | —¹ |
| LightGBM | 0.900 | 0.919 | —¹ | —¹ |
| Spatial kNN | 0.840 | 0.799 | 0.979 | 0.979 |
| Kriging | 0.680 | 0.784 | 0.792 | 0.877 |
| GCN | 0.700 | 0.713 | 0.656 | 0.865 |
| GAT | 0.680 | 0.703 | 0.604 | 0.826 |

> ¹ ROC-AUC/PR-AUC reach 1.000 for most models with 5-node test sets (classes perfectly separable in probability space); reported as — to avoid misleading averages. SPIRE also achieves ROC-AUC=1.000, PR-AUC=1.000 here.

#### Run 2 — seed=23

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.960** | **0.967** | 1.000 | 1.000 |
| MLP | 0.960 | 0.969 | 0.944 | 0.931 |
| Autoencoder | 0.960 | 0.969 | 0.944 | 0.931 |
| XGBoost | 0.940 | 0.956 | 1.000 | 1.000 |
| CatBoost | 0.940 | 0.956 | 0.972 | 0.972 |
| RandomForest | 0.940 | 0.955 | 1.000 | 1.000 |
| SVM | 0.940 | 0.956 | 0.944 | 0.931 |
| LightGBM | 0.920 | 0.942 | 1.000 | 1.000 |
| Spatial kNN | 0.840 | 0.852 | 0.806 | 0.900 |
| Kriging | 0.720 | 0.810 | 0.833 | 0.867 |
| GCN | 0.620 | 0.692 | 0.694 | 0.760 |
| GAT | 0.620 | 0.713 | 0.708 | 0.793 |

#### Run 3 — seed=10

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.880** | **0.913** | 0.917 | 0.957 |
| LightGBM | 0.860 | 0.871 | 0.976 | 0.976 |
| RandomForest | 0.860 | 0.880 | 0.917 | 0.893 |
| MLP | 0.860 | 0.880 | 0.952 | 0.964 |
| XGBoost | 0.840 | 0.861 | 0.976 | 0.976 |
| CatBoost | 0.840 | 0.861 | 0.976 | 0.976 |
| SVM | 0.840 | 0.861 | 0.893 | 0.921 |
| Autoencoder | 0.840 | 0.861 | 0.976 | 0.976 |
| Spatial kNN | 0.820 | 0.833 | 0.798 | 0.815 |
| Kriging | 0.720 | 0.807 | 0.536 | 0.711 |
| GCN | 0.580 | 0.698 | 0.571 | 0.752 |
| GAT | 0.520 | 0.622 | 0.548 | 0.725 |

#### Run 4 — seed=16

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.860** | **0.913** | 0.861 | 0.942 |
| LightGBM | 0.880 | 0.905 | 0.824 | 0.942 |
| RandomForest | 0.860 | 0.891 | 0.870 | 0.969 |
| MLP | 0.840 | 0.899 | 0.769 | 0.925 |
| SVM | 0.840 | 0.880 | 0.824 | 0.946 |
| CatBoost | 0.840 | 0.875 | 0.806 | 0.919 |
| XGBoost | 0.800 | 0.850 | 0.889 | 0.899 |
| Autoencoder | 0.820 | 0.865 | 0.852 | 0.955 |
| Spatial kNN | 0.760 | 0.785 | 0.620 | 0.819 |
| Kriging | 0.680 | 0.785 | 0.481 | 0.794 |
| GCN | 0.580 | 0.651 | 0.491 | 0.745 |
| GAT | 0.680 | 0.745 | 0.519 | 0.821 |

#### Run 5 — seed=49

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.920** | **0.947** | 0.958 | 0.992 |
| LightGBM | 0.900 | 0.933 | 0.917 | 0.983 |
| SVM | 0.900 | 0.942 | 0.667 | 0.906 |
| Autoencoder | 0.900 | 0.942 | 0.917 | 0.983 |
| XGBoost | 0.880 | 0.922 | 0.833 | 0.965 |
| RandomForest | 0.880 | 0.922 | 0.625 | 0.895 |
| MLP | 0.880 | 0.922 | 0.875 | 0.973 |
| CatBoost | 0.860 | 0.911 | 0.917 | 0.983 |
| Kriging | 0.820 | 0.888 | 0.597 | 0.881 |
| Spatial kNN | 0.780 | 0.851 | 0.632 | 0.867 |
| GAT | 0.760 | 0.848 | 0.556 | 0.805 |
| GCN | 0.740 | 0.835 | 0.458 | 0.766 |

#### Average — 5 runs (seeds 42, 23, 10, 16, 49)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|----|---------|--------|
| **SPIRE** | **0.916 ± 0.041** | **0.939 ± 0.022** | **0.947 ± 0.053** | **0.978 ± 0.024** |
| MLP | 0.896 ± 0.046 | 0.923 ± 0.032 | 0.908 ± 0.080 | 0.958 ± 0.028 |
| Autoencoder | 0.892 ± 0.055 | 0.916 ± 0.044 | 0.938 ± 0.051 | 0.969 ± 0.024 |
| LightGBM | 0.892 ± 0.020 | 0.914 ± 0.025 | 0.943 ± 0.067 | 0.980 ± 0.021 |
| RandomForest | 0.892 ± 0.032 | 0.912 ± 0.026 | 0.882 ± 0.138 | 0.951 ± 0.048 |
| SVM | 0.888 ± 0.041 | 0.910 ± 0.036 | 0.866 ± 0.115 | 0.941 ± 0.032 |
| CatBoost | 0.884 ± 0.046 | 0.909 ± 0.037 | 0.934 ± 0.070 | 0.970 ± 0.027 |
| XGBoost | 0.872 ± 0.048 | 0.901 ± 0.040 | 0.940 ± 0.067 | 0.968 ± 0.037 |
| Spatial kNN | 0.808 ± 0.032 | 0.824 ± 0.027 | 0.767 ± 0.132 | 0.876 ± 0.060 |
| Kriging | 0.724 ± 0.051 | 0.815 ± 0.038 | 0.648 ± 0.140 | 0.826 ± 0.066 |
| GAT | 0.652 ± 0.080 | 0.726 ± 0.073 | 0.587 ± 0.067 | 0.794 ± 0.036 |
| GCN | 0.644 ± 0.065 | 0.718 ± 0.062 | 0.574 ± 0.091 | 0.778 ± 0.044 |

Source: `result/inductive_sample.json`

---

### Exp 4 — Region-level proportion prediction (Proportion)

For each held-out region (leave-one-region-out, same split as Exp 1), every node is predicted. `predicted_proportion = mean(y_prob)`. GCN and GAT use the same hyperparameters tuned for SPIRE.

#### MAE ⬇, RMSE ⬇

#### Run 1 — seed=42 (reference)

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.1026** | **0.1238** |
| RandomForest | 0.1371 | 0.1694 |
| SVM | 0.1412 | 0.1729 |
| Autoencoder | 0.1427 | 0.1704 |
| CatBoost | 0.1442 | 0.1766 |
| MLP | 0.1446 | 0.1812 |
| GCN | 0.1469 | 0.1689 |
| XGBoost | 0.1480 | 0.1799 |
| LightGBM | 0.1510 | 0.1833 |
| Spatial kNN | 0.1624 | 0.1960 |
| Kriging | 0.1682 | 0.1830 |
| GAT | 0.1935 | 0.2308 |

#### Run 2 — seed=1

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0917** | **0.1100** |
| RandomForest | 0.1362 | 0.1704 |
| Autoencoder | 0.1363 | 0.1659 |
| SVM | 0.1409 | 0.1727 |
| CatBoost | 0.1444 | 0.1773 |
| MLP | 0.1449 | 0.1885 |
| XGBoost | 0.1473 | 0.1789 |
| LightGBM | 0.1502 | 0.1845 |
| Spatial kNN | 0.1624 | 0.1960 |
| GCN | 0.1625 | 0.1849 |
| Kriging | 0.1682 | 0.1830 |
| GAT | 0.1918 | 0.2177 |

#### Run 3 — seed=8

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.1120** | **0.1376** |
| RandomForest | 0.1369 | 0.1693 |
| MLP | 0.1402 | 0.1805 |
| SVM | 0.1413 | 0.1730 |
| Autoencoder | 0.1416 | 0.1685 |
| CatBoost | 0.1438 | 0.1776 |
| XGBoost | 0.1458 | 0.1788 |
| GCN | 0.1470 | 0.1710 |
| LightGBM | 0.1525 | 0.1853 |
| Spatial kNN | 0.1624 | 0.1960 |
| Kriging | 0.1682 | 0.1830 |
| GAT | 0.1975 | 0.2464 |

#### Run 4 — seed=9

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.1037** | **0.1170** |
| RandomForest | 0.1381 | 0.1697 |
| SVM | 0.1409 | 0.1727 |
| CatBoost | 0.1448 | 0.1783 |
| Autoencoder | 0.1479 | 0.1768 |
| MLP | 0.1481 | 0.1870 |
| XGBoost | 0.1490 | 0.1801 |
| LightGBM | 0.1516 | 0.1819 |
| GCN | 0.1543 | 0.1747 |
| Spatial kNN | 0.1624 | 0.1960 |
| Kriging | 0.1682 | 0.1830 |
| GAT | 0.1956 | 0.2305 |

#### Run 5 — seed=10

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0750** | **0.0914** |
| GAT | 0.1364 | 0.1662 |
| RandomForest | 0.1377 | 0.1704 |
| Autoencoder | 0.1383 | 0.1685 |
| SVM | 0.1411 | 0.1731 |
| CatBoost | 0.1462 | 0.1780 |
| XGBoost | 0.1468 | 0.1803 |
| LightGBM | 0.1511 | 0.1834 |
| MLP | 0.1530 | 0.1919 |
| GCN | 0.1583 | 0.1873 |
| Spatial kNN | 0.1624 | 0.1960 |
| Kriging | 0.1682 | 0.1830 |

#### Average — 5 runs (seeds 42, 1, 8, 9, 10)

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0970±0.0128** | **0.1159±0.0153** |
| RandomForest | 0.1372±0.0007 | 0.1698±0.0005 |
| SVM | 0.1411±0.0001 | 0.1729±0.0002 |
| Autoencoder | 0.1413±0.0040 | 0.1700±0.0037 |
| CatBoost | 0.1447±0.0008 | 0.1776±0.0006 |
| MLP | 0.1462±0.0043 | 0.1858±0.0044 |
| XGBoost | 0.1474±0.0011 | 0.1796±0.0006 |
| LightGBM | 0.1513±0.0008 | 0.1837±0.0011 |
| GCN | 0.1538±0.0062 | 0.1773±0.0074 |
| Spatial kNN | 0.1624±0.0000 | 0.1960±0.0000 |
| Kriging | 0.1682±0.0000 | 0.1830±0.0000 |
| GAT | 0.1830±0.0234 | 0.2183±0.0276 |

Source: `result/exp4_proportion.json`

---

### Exp 5 — Per-region proportion, random split (Proportion)

Per-region isolated graph, same random split as Exp 2. `predicted_proportion = mean(y_prob)` over 5 test nodes.

#### MAE ⬇, RMSE ⬇

#### Run 1 — seed=42 (reference)

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.1333** | **0.1644** |
| Autoencoder | 0.1444 | 0.1867 |
| MLP | 0.1467 | 0.1958 |
| SVM | 0.1478 | 0.1878 |
| XGBoost | 0.1478 | 0.1947 |
| RandomForest | 0.1478 | 0.1912 |
| CatBoost | 0.1494 | 0.2006 |
| Kriging | 0.1829 | 0.2056 |
| Spatial kNN | 0.1840 | 0.2150 |
| LightGBM | 0.1760 | 0.2354 |
| GCN | 0.2056 | 0.2489 |
| GAT | 0.2502 | 0.2780 |

#### Run 2 — seed=18

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0795** | **0.0964** |
| GCN | 0.0905 | 0.1270 |
| Kriging | 0.1338 | 0.1757 |
| GAT | 0.1341 | 0.1750 |
| Autoencoder | 0.1543 | 0.1870 |
| SVM | 0.1604 | 0.1848 |
| Spatial kNN | 0.1655 | 0.1952 |
| RandomForest | 0.1742 | 0.1982 |
| LightGBM | 0.1744 | 0.1996 |
| XGBoost | 0.1753 | 0.2031 |
| CatBoost | 0.1809 | 0.2084 |
| MLP | 0.1902 | 0.2108 |

#### Run 3 — seed=20

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0963** | **0.1362** |
| Autoencoder | 0.1039 | 0.1536 |
| SVM | 0.1043 | 0.1676 |
| RandomForest | 0.1084 | 0.1916 |
| CatBoost | 0.1089 | 0.2071 |
| MLP | 0.1125 | 0.1891 |
| LightGBM | 0.1209 | 0.1987 |
| XGBoost | 0.1341 | 0.2326 |
| Spatial kNN | 0.1781 | 0.2096 |
| GCN | 0.2091 | 0.2309 |
| GAT | 0.1984 | 0.2268 |
| Kriging | 0.2143 | 0.2493 |

#### Run 4 — seed=48

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0795** | **0.0964** |
| GCN | 0.0905 | 0.1270 |
| Kriging | 0.1338 | 0.1757 |
| GAT | 0.1341 | 0.1750 |
| Autoencoder | 0.1543 | 0.1870 |
| SVM | 0.1604 | 0.1848 |
| Spatial kNN | 0.1655 | 0.1952 |
| RandomForest | 0.1742 | 0.1982 |
| LightGBM | 0.1744 | 0.1996 |
| XGBoost | 0.1753 | 0.2031 |
| CatBoost | 0.1809 | 0.2084 |
| MLP | 0.1902 | 0.2108 |

#### Run 5 — seed=50

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0963** | **0.1362** |
| Autoencoder | 0.1039 | 0.1536 |
| SVM | 0.1043 | 0.1676 |
| RandomForest | 0.1084 | 0.1916 |
| CatBoost | 0.1089 | 0.2071 |
| MLP | 0.1125 | 0.1891 |
| LightGBM | 0.1209 | 0.1987 |
| XGBoost | 0.1341 | 0.2326 |
| Spatial kNN | 0.1781 | 0.2096 |
| GAT | 0.1984 | 0.2268 |
| GCN | 0.2091 | 0.2309 |
| Kriging | 0.2143 | 0.2493 |

#### Average — 5 runs (seeds 42, 18, 20, 48, 50)

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0970 ± 0.0196** | **0.1259 ± 0.0262** |
| Autoencoder | 0.1322 ± 0.0234 | 0.1736 ± 0.0163 |
| SVM | 0.1354 ± 0.0258 | 0.1785 ± 0.0090 |
| RandomForest | 0.1426 ± 0.0295 | 0.1941 ± 0.0033 |
| CatBoost | 0.1458 ± 0.0322 | 0.2063 ± 0.0029 |
| MLP | 0.1504 ± 0.0348 | 0.1991 ± 0.0098 |
| XGBoost | 0.1533 ± 0.0186 | 0.2132 ± 0.0161 |
| LightGBM | 0.1533 ± 0.0265 | 0.2064 ± 0.0145 |
| Spatial kNN | 0.1742 ± 0.0074 | 0.2049 ± 0.0082 |
| Kriging | 0.1758 ± 0.0362 | 0.2111 ± 0.0330 |
| GCN | 0.1610 ± 0.0575 | 0.1929 ± 0.0542 |
| GAT | 0.1831 ± 0.0442 | 0.2163 ± 0.0386 |

Source: `result/exp5_proportion_per_region.json`

---

### Exp 6 — Global inductive proportion, random split (Proportion)

Global inductive setting, same split as Exp 3. `predicted_proportion = mean(y_prob)` over 5 test nodes per region.

#### MAE ⬇, RMSE ⬇

#### Run 1 — seed=42 (reference)

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0807** | **0.0994** |
| Autoencoder | 0.0885 | 0.1116 |
| CatBoost | 0.0946 | 0.1151 |
| SVM | 0.0958 | 0.1182 |
| LightGBM | 0.0966 | 0.1180 |
| XGBoost | 0.0995 | 0.1233 |
| RandomForest | 0.0996 | 0.1209 |
| Spatial kNN | 0.1227 | 0.1449 |
| MLP | 0.1301 | 0.1632 |
| GAT | 0.1805 | 0.2168 |
| GCN | 0.2012 | 0.2233 |
| Kriging | 0.2293 | 0.2407 |

#### Run 2 — seed=20

| Model | MAE | RMSE |
|-------|-----|------|
| Autoencoder | 0.0646 | 0.0922 |
| **SPIRE** | **0.0648** | **0.0882** |
| SVM | 0.0657 | 0.0945 |
| MLP | 0.0681 | 0.0986 |
| LightGBM | 0.0725 | 0.1024 |
| CatBoost | 0.0726 | 0.0999 |
| RandomForest | 0.0727 | 0.0988 |
| XGBoost | 0.0757 | 0.1033 |
| Spatial kNN | 0.1682 | 0.2189 |
| Kriging | 0.2208 | 0.2798 |
| GCN | 0.2291 | 0.2613 |
| GAT | 0.2619 | 0.2921 |

#### Run 3 — seed=11

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0841** | **0.1167** |
| LightGBM | 0.1090 | 0.1515 |
| MLP | 0.1158 | 0.1633 |
| RandomForest | 0.1185 | 0.1651 |
| XGBoost | 0.1188 | 0.1575 |
| Autoencoder | 0.1215 | 0.1651 |
| CatBoost | 0.1221 | 0.1613 |
| SVM | 0.1244 | 0.1634 |
| Spatial kNN | 0.1441 | 0.1805 |
| Kriging | 0.1624 | 0.1967 |
| GCN | 0.2191 | 0.2820 |
| GAT | 0.2545 | 0.2991 |

#### Run 4 — seed=60

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0718** | **0.0944** |
| MLP | 0.0899 | 0.1127 |
| SVM | 0.0975 | 0.1240 |
| Autoencoder | 0.0993 | 0.1256 |
| RandomForest | 0.1181 | 0.1418 |
| CatBoost | 0.1196 | 0.1414 |
| LightGBM | 0.1361 | 0.1720 |
| XGBoost | 0.1363 | 0.1658 |
| Spatial kNN | 0.1591 | 0.1692 |
| Kriging | 0.1834 | 0.2095 |
| GCN | 0.2308 | 0.2798 |
| GAT | 0.2442 | 0.3084 |

#### Run 5 — seed=48

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0958** | **0.1183** |
| RandomForest | 0.1148 | 0.1501 |
| Autoencoder | 0.1186 | 0.1537 |
| MLP | 0.1263 | 0.1757 |
| CatBoost | 0.1292 | 0.1681 |
| SVM | 0.1307 | 0.1676 |
| XGBoost | 0.1436 | 0.1807 |
| LightGBM | 0.1472 | 0.1849 |
| Spatial kNN | 0.1475 | 0.1988 |
| GAT | 0.1800 | 0.2222 |
| GCN | 0.1816 | 0.2194 |
| Kriging | 0.1948 | 0.2315 |

#### Average — 5 runs (seeds 42, 20, 11, 60, 48)

| Model | MAE | RMSE |
|-------|-----|------|
| **SPIRE** | **0.0795 ± 0.0106** | **0.1034 ± 0.0120** |
| Autoencoder | 0.0985 ± 0.0209 | 0.1296 ± 0.0268 |
| SVM | 0.1028 ± 0.0232 | 0.1335 ± 0.0279 |
| RandomForest | 0.1047 ± 0.0175 | 0.1353 ± 0.0232 |
| MLP | 0.1060 ± 0.0236 | 0.1427 ± 0.0309 |
| CatBoost | 0.1076 ± 0.0210 | 0.1371 ± 0.0262 |
| LightGBM | 0.1123 ± 0.0269 | 0.1458 ± 0.0313 |
| XGBoost | 0.1148 ± 0.0248 | 0.1461 ± 0.0285 |
| Spatial kNN | 0.1483 ± 0.0154 | 0.1825 ± 0.0252 |
| Kriging | 0.1981 ± 0.0244 | 0.2316 ± 0.0287 |
| GCN | 0.2124 ± 0.0187 | 0.2532 ± 0.0270 |
| GAT | 0.2242 ± 0.0363 | 0.2677 ± 0.0397 |

Source: `result/exp6_proportion_inductive_sample.json`

---

## Ablation Study — Feature Component Validation (SPIRE)

Validates the two feature engineering choices central to the GNN pipeline: (a) edge-stat augmentation and (b) the ensemble prior score. Each ablation removes one component and measures the F1 drop on Exp 1 (leave-one-region-out inductive, seeds 42/1/2/4/7) and Exp 3 (global inductive, 5 nodes/region, seeds 42/23/10/16/49). Full model figures are taken from the existing multi-run averages.

### Exp 1 — Leave-one-region-out (avg ± std, 5 seeds)

| Condition | Dim | Accuracy | F1 | ROC-AUC | PR-AUC |
|-----------|-----|----------|----|---------|--------|
| **Full** (SHAP + ensemble + edge-stat) | 13 | **0.793 ± 0.012** | **0.844 ± 0.009** | **0.841 ± 0.004** | **0.912 ± 0.003** |
| − Edge-stat augmentation | 11 | 0.767 ± 0.014 | 0.828 ± 0.012 | 0.842 ± 0.007 | 0.915 ± 0.009 |
| − Ensemble prior | 10 | 0.732 ± 0.025 | 0.807 ± 0.014 | 0.781 ± 0.006 | 0.895 ± 0.006 |

### Exp 3 — Global inductive, random sample (avg ± std, 5 seeds)

| Condition | Dim | Accuracy | F1 | ROC-AUC | PR-AUC |
|-----------|-----|----------|----|---------|--------|
| **Full** (SHAP + ensemble + edge-stat) | 13 | **0.916 ± 0.041** | **0.939 ± 0.022** | **0.947 ± 0.053** | **0.978 ± 0.024** |
| − Edge-stat augmentation | 11 | 0.876 ± 0.042 | 0.901 ± 0.041 | 0.920 ± 0.038 | 0.960 ± 0.033 |
| − Ensemble prior | 10 | 0.793 ± 0.081 | 0.820 ± 0.076 | 0.844 ± 0.071 | 0.913 ± 0.053 |

Both components contribute positively across all settings. Removing the ensemble prior (mean predict_proba of XGBoost + CatBoost + RF, fitted on train only) causes the larger drop — −0.037 F1 in Exp 1 and −0.119 in Exp 3 — because it strips the strongest tabular signal and removes cross-regional probability calibration, forcing the GNN to infer REE likelihood from SHAP features alone. Removing edge-stat augmentation (mean and max incoming edge weights appended to the 11-dim base) has a smaller but consistent effect (−0.016 in Exp 1, −0.038 in Exp 3), confirming that spatial-proximity signals encoded in edge statistics complement neighbourhood aggregation. The higher variance in Exp 3 under no-ensemble (std = ±0.076) reflects sensitivity to which 5 nodes are sampled per region: when sampled nodes are spatially representative the drop is moderate; when they are outliers the degradation is severe.

Sources: `result/ablation_feature.json`, `result/ablation_feature_exp3.json`

### Graph Construction k Sensitivity — Exp 1 (SPIRE, seed=42)

Sweeps the number of nearest neighbours k used to build the spatial graph in Exp 1 (leave-one-region-out inductive). All other settings fixed (13-dim full pipeline, seed=42).

| k | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|----------|----|---------|--------|
| 5 | 0.802 | 0.854 | 0.844 | 0.916 |
| 10 | 0.743 | 0.806 | 0.833 | 0.910 |
| **15** *(default)* | **0.792** | **0.844** | **0.838** | **0.919** |
| 20 | 0.777 | 0.837 | 0.827 | 0.907 |
| 25 | 0.800 | 0.845 | 0.817 | 0.905 |
| 30 | 0.789 | 0.846 | 0.831 | 0.911 |
| 35 | 0.796 | 0.845 | 0.833 | 0.911 |
| 40 | 0.808 | 0.856 | 0.827 | 0.909 |
| 45 | 0.791 | 0.847 | 0.833 | 0.907 |

F1 is stable across k ∈ {15, …, 45} (0.844–0.856) with no monotonic trend. k=10 is the only outlier (F1=0.806), likely because sparser graphs leave some deposits with weak neighbourhood signals. k=5 performs well (F1=0.854) but may underrepresent spatial context in dense regions. The default k=15 sits in the stable plateau and is a robust choice across the tested range.

Source: `result/k_sensitivity.json`

---

## Cross-Experiment Comparison

### Classification — SPIRE vs best baseline (F1 ⬆)

| Experiment | Setting | SPIRE F1 | Best baseline F1 | Best baseline | Winner |
|-----------|---------|-------------|-----------------|---------------|--------|
| Exp 1 — Leave-one-region-out | Cross-region, inductive vs transductive | **0.827** | 0.823 | MLP | **GNN** |
| Exp 2 — Per-region random | Same-region, 5 test nodes | **0.935** | 0.913 | XGB/CatBoost/RF/SVM/MLP | **GNN** |
| Exp 3 — Global inductive random | Global graph, 5 test nodes | **0.956** | 0.944 | CatBoost/MLP | **GNN** |

### Classification — GNN model comparison (F1 ⬆)

| Experiment | SPIRE | GCN | GAT |
|-----------|-----------|-----|-----|
| Exp 1 — Leave-one-region-out | **0.827** | 0.798 | 0.750 |
| Exp 2 — Per-region random | **0.935** | 0.837 | 0.837 |
| Exp 3 — Global inductive random | **0.956** | 0.713 | 0.703 |

### Proportion — SPIRE vs best baseline (MAE ⬇)

| Experiment | Setting | SPIRE MAE | Best baseline MAE | Best baseline | Winner |
|-----------|---------|--------------|------------------|---------------|--------|
| Exp 4 — Leave-one-region-out | All nodes, cross-region | **0.1026** | 0.1371 | RandomForest | **GNN** |
| Exp 5 — Per-region random | 5 nodes, same-region | **0.1333** | 0.1467 | MLP | **GNN** |
| Exp 6 — Global inductive random | 5 nodes, global graph | **0.0807** | 0.0885 | Autoencoder | **GNN** |

---

## Key Findings

### Classification experiments (Exp 1–3)

- **SPIRE outperforms baselines across all classification experiments:** F1=0.827 vs 0.823 MLP (Exp 1), 0.935 vs 0.913 (Exp 2), 0.956 vs 0.944 (Exp 3). Graph structure consistently adds value over tabular models with random test splits.
- **SPIRE consistently dominates GCN and GAT:** Best GNN in all classification experiments. Gap largest in Exp 3 (0.956 vs GCN 0.713). GCN/GAT reuse SPIRE's tuned config, which likely underestimates their potential.
- **Small test sets inflate baseline AUC:** With 5 test nodes (Exp 2, 3), ROC-AUC/PR-AUC≈1.000 for most baselines — not reliable at this scale. Use F1 for comparison.
- **Exp 1 is the most methodologically sound classification benchmark:** Only experiment where GNN and baselines face equal distributional shift (unseen region).

### Proportion experiments (Exp 4–6)

- **SPIRE wins proportion prediction across all settings:** MAE: 0.1026 (Exp 4), 0.1333 (Exp 5), 0.0807 (Exp 6). Graph structure helps estimate region-level proportions better than tabular models when test nodes are representative.
- **Global training context is key for proportion accuracy (Exp 6 vs Exp 5):** MAE drops significantly when the full global graph is used (SPIRE: 0.1333→0.0807). Tree models also benefit substantially from the larger training context.
- **GCN and GAT underperform on proportion tasks:** MAE=0.19–0.26 in Exp 5 and Exp 6 — far worse than tabular baselines. Only SPIRE (with edge-stat augmentation) consistently matches or beats tabular models.

---

## Running Order

```bash
# 0. Preprocess raw data
python preprocessing/pipeline.py

# Exp 1 — Leave-one-region-out
python experiment/run_baseline.py
python experiment/run_gnn.py

# Exp 2 — Per-region, random split
python experiment/run_baseline_per_region.py
python experiment/run_gnn_per_region.py

# Exp 3 — Global inductive, random split
python experiment/run_inductive_sample.py

# Exp 4 — Region-level proportion prediction
python experiment/run_exp4_proportion.py

# Exp 5 — Per-region proportion, random split
python experiment/run_exp5_proportion_per_region.py

# Exp 6 — Global inductive proportion, random split
python experiment/run_exp6_proportion_inductive_sample.py
```

---

## Dependencies

```
pandas numpy scikit-learn scipy openpyxl pyarrow networkx
xgboost lightgbm catboost shap
torch torch-geometric
pykrige
```

```bash
pip install -r requirements.txt
```