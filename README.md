# SPIRE-REE

Code, processed data and result files for the manuscript

> **Integrative Tabular and Inductive Graph Representation Learning for Cross-Region Rare Earth Occurrence Classification**

SPIRE (SPatially-augmented Inductive REpresentation lEarning) combines an out-of-fold tabular ensemble prior with an inductive GraphSAGE model. It classifies whether a USGS occurrence record documents at least one REE mineral and estimates the positive fraction among held-out records.

This README is also the reproducibility note for the manuscript. It covers the experiment protocol, seed sets, the use of the validation set, the treatment of single-class test folds, and where each table and figure comes from.

**Repository version cited in the manuscript:** release tag `v2.1-revision3`. The results and figures in that release are the ones reported in the revised manuscript.

---

## Contents

- [Installation](#installation)
- [Data](#data)
- [Experiments](#experiments)
- [Seeds](#seeds)
- [Training, validation and test partitions](#training-validation-and-test-partitions)
- [Metrics and single-class test folds](#metrics-and-single-class-test-folds)
- [Where each manuscript result comes from](#where-each-manuscript-result-comes-from)
- [Headline results](#headline-results)
- [Reproducing the results](#reproducing-the-results)
- [Repository layout](#repository-layout)
- [Known limitations of the record](#known-limitations-of-the-record)

---

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/han-1808/SPIRE-REE.git
cd SPIRE-REE
git checkout v2.1-revision3
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data

- **Source:** USGS global database of REE deposits and occurrences (Orris et al., 2018), stored as `data/Global_REE_occurrence_database.xlsx`.
- **Records:** after removing Antarctic records, 3,112 records remain.
- **Label (`has_ree`):** 1 if the record documents at least one REE mineral (2,025 records, 65.1%); 0 if no REE mineral is documented (1,087 records). Class 0 means *no REE mineral documented*. It does not establish that REEs are absent.
- **Regions:** the ten groups of the source `Region` field, used without redefining their boundaries: Africa, China, East Asia, Europe, Middle East, North America, Oceania, Russian Federation, South America, South and Central Asia. One China record near the Mongolian border is coded East Asia, and three Labrador (Canada) records are coded Europe in the source field. These assignments are retained.
- **Processed outputs** (written by `preprocessing/pipeline.py`):
  - `data/Global_REE_occurrence_database_processed.xlsx`
  - `data/gnn_features.parquet`, `data/gnn_target.npy`, `data/gnn_edge_index.npy`, `data/gnn_edge_weight.npy`

---

## Experiments

Six main experiments cross two tasks (classification and proportion estimation) with three spatial settings.

| Exp | Task | Setting | Held-out set | Graph |
|---|---|---|---|---|
| 1 | Classification | Leave-one-region-out | All records of one region | Global, nine training regions |
| 2 | Classification | Per-region spatial blocks | About 15% of each region, as whole spatial blocks | One graph per region |
| 3 | Classification | Global spatial blocks | About 15% of each region, as whole spatial blocks | Global |
| 4 | Proportion | Leave-one-region-out | All records of one region | Global, nine training regions |
| 5 | Proportion | Per-region spatial blocks | As Exp 2 | One graph per region |
| 6 | Proportion | Global spatial blocks | As Exp 3 | Global |

**Spatial blocks (Exp 2, 3, 5, 6).** Within each region, k-means on unit-sphere coordinates forms spatial blocks. Whole blocks are held out until about 15% of the region is selected. Training and validation candidates within the region's mean k-nearest-neighbour distance of any held-out record are then excluded (the buffer). The realised held-out fraction is 15.0–23.8% (Exp 2 and 5) and 14.3–26.4% (Exp 3 and 6). Exp 1 and 4 hold out complete regions and use no buffer.

**Proportion (Exp 4–6).** The predicted fraction is the mean positive-class probability over the held-out records. It is compared with the observed fraction of positive labels.

**Models.**

- *Graph models:* SPIRE (GraphSAGE, unweighted mean aggregation) and its GCN and GAT variants. All three use the same 13-dimensional node vector: 10 SHAP-selected attributes, the ensemble prior (mean of XGBoost, CatBoost and Random Forest probabilities), and the mean and maximum incoming edge weights.
- *Tabular baselines:* XGBoost, LightGBM, CatBoost, Random Forest, RBF-kernel SVM, MLP, and an autoencoder followed by logistic regression. Their input is the 10 SHAP-selected attributes plus an out-of-fold XGBoost score.
- *Position-only baselines:* ordinary kriging and distance-weighted spatial k-NN regression. These use coordinates only.

**Neighbourhood sizes.** Training graph k and attachment size k_pred are 15/15 in Exp 1 and 4, 10/10 in Exp 2 and 5, and 15/10 in Exp 3 and 6.

---

## Seeds

Each main experiment uses five runs. Seeds vary the training/validation partition, the out-of-fold prior and the initialisation. In the block experiments they also vary the held-out blocks. These are the seed lists stored in the result files and reported in the protocol-configuration table of the manuscript.

| Run | Seeds | Result file |
|---|---|---|
| Exp 1 graph models | 42, 1, 2, 4, 7 | `result/gnn_inductive.json` |
| Exp 1 tabular baselines | 42, 1, 2, 4, 7 | `result/baseline.json` |
| Exp 2 graph models and baselines | 42, 1, 2, 4, 7 | `result/gnn_per_region.json`, `result/baseline_per_region.json` |
| Exp 3 | 42, 47, 33, 25, 31 | `result/inductive_sample.json` |
| Exp 4 | 42, 47, 14, 30, 35 | `result/exp4_proportion.json` |
| Exp 5 | 42, 59, 52, 46, 9 | `result/exp5_proportion_per_region.json` |
| Exp 6 | 42, 46, 11, 4, 34 | `result/exp6_proportion_inductive_sample.json` |
| Feature ablation, Exp 1 | 42, 1, 2, 4, 7 (initialisation only) | `result/ablation_feature.json` |
| Feature ablation, Exp 3 | 42, 23, 10, 16, 49 (initialisation only) | `result/ablation_feature_exp3.json` |
| Same-input MLP control | 42, 1, 2, 4, 7 (initialisation only) | `result/ablation_no_message_passing.json` |
| Ensemble-only control | 42, 1, 2, 4, 7 | `result/ablation_ensemble_only.json` |
| Rewiring control | 42, 1, 2, 4, 7 | `result/ablation_random_rewire.json` |
| Leave-one-deposit-type-out | 42, 1, 2 | `result/lodto.json` |
| k sweep, hyperparameter sweep, feature-selection record | 42 | `result/k_sensitivity.json`, `result/hparam_sensitivity.json`, `result/feature_selection_per_fold.json` |

Controlled ablations fix the split and the prior at seed 42 and vary only initialisation (and, for rewiring, the topology). Their Exp 1 reference (F1 = 0.837) therefore differs from the main Exp 1 result (0.829).

Exp 2 graph models and baselines use the same held-out records by construction. The record identifiers of every split are stored in `result/split_manifest_per_region.json`, and both scripts check their splits against it.

---

## Training, validation and test partitions

Within one outer fold and seed (Algorithm 1 in the manuscript):

1. The held-out records T (a region or spatial blocks) are separated first. Then 20% of the remaining records (after the buffer, in block experiments) are drawn at random as the validation set V. The rest form the training set Tr.
2. Encoders, coordinate scaling and SHAP feature selection are fitted on Tr. Feature selection is done once on Tr and is not repeated inside the inner folds.
3. Inner five-fold cross-validation on Tr gives the out-of-fold ensemble prior for training records. The ensemble is then refitted on all of Tr to give the prior for V and T.
4. The k-NN graph is built on Tr ∪ V. The edge-weight scale σ is computed from training nodes only.
5. SPIRE is trained with the loss computed on Tr labels only.
6. Each record in T is attached individually to its k_pred nearest nodes of the Tr ∪ V graph, predicted and removed before the next record. Held-out records never interact.

**What the validation set is used for.**

- **Validation labels** are used only for model selection:
  - choosing among the 36 SPIRE configurations by validation F1;
  - early stopping and checkpoint selection;
  - choosing the neighbourhood size k (Exp 1 k sweep, by validation F1).
- **Validation labels are not used** for the training loss, feature selection, ensemble fitting, the edge-weight scale or the decision threshold (fixed at 0.5). The selected model is not refitted on Tr ∪ V.
- **Validation features** are part of the graph. Validation nodes take part in message passing during training, and they can be neighbours of held-out records at inference.
- **GCN and GAT** reuse the configuration selected for SPIRE.
- **The same-input MLP control** uses the same 36-configuration search on the same validation set.
- **Tabular baselines** use fixed configurations, are trained on Tr only, and do not use V.

Held-out labels are never used for fitting or selection.

---

## Metrics and single-class test folds

- **Classification metrics:** F1, ROC-AUC and PR-AUC.
- **Proportion metrics:** MAE and RMSE.
- **Spread:** ± is the population standard deviation (`ddof=0`) over the five runs.
- **Averaging:** leave-one-region-out and per-region results are averaged over regions within a run, then over runs.
- **Pooled ROC-AUC** concatenates all held-out predictions. It is a different quantity from the mean of per-region ROC-AUC values.

**Single-class test folds.** ROC-AUC and PR-AUC are undefined when a test fold contains only one class. Such folds store `null` for both metrics and are left out of the regional mean. F1, accuracy, precision and recall are still computed for these folds. The rule is the same for every model.

| Experiment | Single-class folds (seed, region, n) | Where recorded |
|---|---|---|
| Exp 2 | (2, Middle East, 14), all positive | `per_region["Middle East"]` of the seed-2 run in `result/gnn_per_region.json` and `result/baseline_per_region.json`; field `_note_single_class_folds` |
| Exp 3 | (42, South America, 54), all positive | seed-42 run of `result/inductive_sample.json` (`roc_auc` and `pr_auc` are `null`) |

Exp 1 and 4 hold out whole regions, which always contain both classes. Exp 5 and 6 report pooled ROC-AUC only. Pooled ROC-AUC and the region-cluster bootstrap intervals (all experiments) use the stored predictions directly and are not affected by this rule.

---

## Where each manuscript result comes from

Tables and figures are listed by content because their numbers depend on the journal layout.

**Main experiments**

| Manuscript item | Result file(s) | Produced by |
|---|---|---|
| Protocol configurations (seeds, k, k_pred) | see [Seeds](#seeds) | — |
| Per-region leave-one-region-out F1 | `result/gnn_inductive.json`, `result/baseline.json` | `experiment/run_gnn.py`, `experiment/run_baseline.py` |
| Classification bar chart and radar plot (Exp 1–3) | `result/gnn_inductive.json`, `result/baseline.json`, `result/gnn_per_region.json`, `result/baseline_per_region.json`, `result/inductive_sample.json` | `figures/draw_combined.py` → `fig_X_panel_A.png`, `fig_Y_panel_B.png` |
| Per-region F1 of SPIRE, GCN and GAT | same files | `figures/draw_combined.py` → `fig_Z_panel_C.png` |
| Proportion figures (Exp 4–6) | `result/exp4_proportion.json`, `result/exp5_proportion_per_region.json`, `result/exp6_proportion_inductive_sample.json` | `figures/draw_combined_prop.py` → `fig_A/B/C_panel_*.png` |
| Paired comparisons with the strongest tabular baseline | `result/exp1_paired_tests.json`, `result/paired_significance_test.json` | `experiment/run_exp1_paired_tests.py`, `experiment/run_paired_significance_test.py` |
| Pooled ROC-AUC with region-cluster bootstrap intervals | `result/pooled_auc_ci.json` | `experiment/run_pooled_auc_ci.py` |
| Paired region-cluster bootstrap differences | `result/cluster_bootstrap_ci.json` | `experiment/run_cluster_bootstrap_ci.py` |
| Brier scores and reliability curves | `result/calibration_analysis.json` | `experiment/run_calibration_analysis.py` → `fig_calibration.png` |

**Controls and ablations**

| Manuscript item | Result file(s) | Produced by |
|---|---|---|
| Same-input MLP, ensemble-only and rewiring controls | `result/ablation_no_message_passing.json`, `result/ablation_ensemble_only.json`, `result/ablation_random_rewire.json` | `experiment/run_ablation_no_message_passing.py`, `experiment/run_ablation_ensemble_only.py`, `experiment/run_ablation_random_rewire.py` (+ `merge_rewire_shards.py`) |
| Two-factor feature ablation and ablation figure | `result/ablation_feature.json`, `result/ablation_feature_exp3.json` | `experiment/run_ablation.py`, `experiment/run_ablation_exp3.py`, `figures/draw_ablation.py` |
| k sensitivity | `result/k_sensitivity.json` | `experiment/run_k_sensitivity.py`, `figures/draw_k_sensitivity.py` |
| Hyperparameter heat map | `result/hparam_sensitivity.json` | `experiment/run_hparam_sensitivity.py`, `figures/draw_hparam_heatmap.py` |
| Leave-one-deposit-type-out | `result/lodto.json` | `experiment/run_lodto.py` |
| Pooled versus group-specific ensemble | `result/pooled_vs_stratified.json` | `experiment/run_pooled_vs_stratified.py` |

**Diagnostics**

| Manuscript item | Result file(s) | Produced by |
|---|---|---|
| Node representation and selection frequencies | `result/feature_selection_per_fold.json` | `experiment/run_feature_selection_stability.py` |
| Correlates of misclassification | `result/region_error_decomposition.json` | `experiment/run_region_error_decomposition.py` |
| Completeness-stratified ROC-AUC | `result/completeness_stratified.json`, `result/documentation_completeness*.json` | `experiment/run_completeness_stratified.py`, `experiment/run_documentation_completeness.py` |
| Edge-weight distributions | `result/graph_diagnostics.json`, `result/graph_diagnostics_edge_weights.png` | `experiment/run_graph_diagnostics.py` |
| Deposit-type SHAP and geological rationale | `result/deposit_type_*.json`, `result/geology_rationale.json` | `experiment/run_deposit_type_analysis.py`, `experiment/run_geology_rationale.py` |
| Test-to-training distances after buffering | `result/test_train_distances.json` | `experiment/run_test_train_distances.py` |
| World overview map | `data/` | `visualize_case_study.py` → `fig_world_overview_exp1.png` |

**Split information**

- **Train/validation/test sizes and buffer details** for each block fold are stored under `per_region[<region>]` of each run: `n_train`, `n_val`, `n_test` and `split_info` (`buffer_km`, `n_excluded_by_buffer`, `n_test_frac`, `buffer_fallback`).
- **Exp 3 and 6 post-buffer counts** are in `result/buffer_counts_exp3_exp6.json`.
- **Held-out class counts** for any fold can be recovered from the stored `y_true` in `predictions`.
- **Record identifiers:**
  - Exp 2 splits: `result/split_manifest_per_region.json`.
  - Exp 3 held-out records: `sample_idx` in `result/inductive_sample.json`.
  - Exp 5 splits are reconstructed from the recorded seeds; `experiment/run_check_exp5_splits.py` checks the reconstruction against the saved test labels and sizes.

---

## Headline results

These numbers are from the manuscript's paired comparison with the strongest evaluated tabular baseline. Exp 1 pairs regions; Exp 2–6 pair seeds on matched splits.

| Exp | Metric | SPIRE | Best tabular baseline | p (Wilcoxon) |
|---|---|---|---|---|
| 1 | F1 ↑ | 0.829 | 0.830 (autoencoder) | 1.000 |
| 2 | F1 ↑ | 0.817 | 0.851 (SVM) | 0.438 |
| 3 | F1 ↑ | 0.834 | 0.842 (autoencoder) | 0.625 |
| 4 | MAE ↓ | 0.110 | 0.113 (Random Forest) | 1.000 |
| 5 | MAE ↓ | 0.140 | 0.100 (SVM) | 0.125 |
| 6 | MAE ↓ | 0.126 | 0.076 (XGBoost) | 0.062 |

With five pairs, the smallest possible exact two-sided Wilcoxon p-value is 0.0625. Most tabular baselines have higher pooled ROC-AUC in the spatial-block settings. The same-input MLP, ensemble-only and rewiring controls show no consistent additional gain from message passing.

---

## Reproducing the results

`run_all.sh` runs the pipeline in dependency order, one log per script under `logs/`:

```bash
./run_all.sh            # everything
./run_all.sh core       # preprocessing and the six main experiments
./run_all.sh analysis   # controls, ablations and response analyses
```

Individual steps:

```bash
python preprocessing/pipeline.py                          # processed data and graph artefacts

python experiment/run_baseline.py                         # Exp 1 baselines
python experiment/run_gnn.py --seeds 1 2 4 7              # Exp 1 graph models (seed 42 is the reference run)
python experiment/run_baseline_per_region.py              # Exp 2 baselines
python experiment/run_gnn_per_region.py                   # Exp 2 graph models
python experiment/run_inductive_sample.py                 # Exp 3
python experiment/run_exp4_proportion.py                  # Exp 4
python experiment/run_exp5_proportion_per_region.py       # Exp 5
python experiment/run_exp6_proportion_inductive_sample.py # Exp 6

python experiment/recompute_single_class_auc.py           # Exp 2 regional means from the stored predictions (single-class rule)

python figures/draw_combined.py                           # classification figures
python figures/draw_combined_prop.py                      # proportion figures
python figures/flatten_png.py                             # RGBA -> RGB for pdfTeX
```

`JOBS` and `THREADS` in `run_all.sh` control parallelism.

---

## Repository layout

```
preprocessing/      cleaning, host-lithology inference, features, graph construction, split manifest
experiment/         training code (gnn_training.py, baseline.py) and one run_*.py per experiment or analysis
figures/            plotting scripts and the figures used in the manuscript
result/             JSON outputs of every experiment and analysis
data/               raw USGS workbook and processed artefacts
run_all.sh          full pipeline
```

---

## Known limitations of the record

- Feature selection is fitted once on each outer training set. The ensemble prior is therefore out of fold with respect to ensemble fitting, but not with respect to the preceding supervised feature-selection step.
- Model-selection effort differs between models. SPIRE and the same-input MLP each receive a 36-configuration search; GCN and GAT inherit SPIRE's configuration; tabular baselines use fixed settings.
- The Middle East (n = 63, about 87% positive) gives small and near-single-class test folds in the block experiments.
- Exp 5 splits are reconstructed from seeds rather than stored as record identifiers. The check matches labels and sizes but does not by itself establish record identity.

---

## Citation and availability

The raw data are publicly available from the USGS (Orris et al., 2018). Processed data, code and result files are in this repository. An archived DOI will be issued on acceptance. The files are also available from the corresponding authors on request.
