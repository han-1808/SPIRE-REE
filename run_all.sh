#!/usr/bin/env bash
# Re-runs the revision pipeline on the corrected code, one log per script under
# logs/.  Run from the repository root:
#
#     ./run_all.sh            # everything, in dependency order
#     ./run_all.sh core       # stage 1-2 only (the six main experiments)
#     ./run_all.sh analysis   # stage 3-4 only (reviewer-response analyses)
#
# Scripts inside a stage write disjoint result/*.json files, so a stage runs
# JOBS at a time; stages themselves are strictly ordered because stage 4 reads
# what stage 2 and 3 produce.  A failure is logged and the run continues; the
# summary at the end lists what failed.
#
# JOBS x THREADS should stay at or under the core count.  Each job's libraries
# (OpenMP/MKL/XGBoost/CatBoost) otherwise each try to grab every core, and the
# resulting oversubscription runs slower than plain sequential execution.

set -u
cd "$(dirname "$0")"
PY="$PWD/.venv/bin/python"
JOBS="${JOBS:-3}"
THREADS="${THREADS:-5}"

# Only the joblib worker count is pinned.  The OpenMP variables are left alone
# on purpose: sklearn/joblib limit nested BLAS threads themselves via
# threadpoolctl, and forcing OMP_NUM_THREADS on top of that produced 30-100
# threads per process and a load average of 60+ on 20 cores.  Bounding the
# outer worker count is enough, and results are unaffected (RandomForest
# parallelises over trees, so a given random_state gives the same forest).
export SKLEARN_N_JOBS="$THREADS"
# OpenMP threads sleep instead of spinning while waiting.  Two OpenMP runtimes
# share these processes (libgomp via scikit-learn/XGBoost/CatBoost, plus the one
# torch ships), and under oversubscription their spin-waiting pools deadlocked
# against each other -- a run would stop dead at the first torch call after a
# heavy sklearn section and never resume.  See preprocessing/torch_threads.py.
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
export TORCH_NUM_THREADS=1

mkdir -p logs result

# Exp 1's seeds, matching the five runs stored in the previous results file.
# Seed 42 is run separately (it is the validation run), so the sweep here picks
# up the remaining four and run_gnn.py merges them into the same file.
GNN_SEEDS="${GNN_SEEDS:-1 2 4 7}"

STAGE1=(
  "preprocessing/pipeline.py"
)
STAGE2=(
  "experiment/run_baseline.py"
  "experiment/run_gnn.py --seeds $GNN_SEEDS"
  "experiment/run_baseline_per_region.py"
  "experiment/run_gnn_per_region.py"
  "experiment/run_inductive_sample.py"
  "experiment/run_exp4_proportion.py"
  "experiment/run_exp5_proportion_per_region.py"
  "experiment/run_exp6_proportion_inductive_sample.py"
)
# Depend only on the data, not on stage 2's result files.
STAGE3=(
  "experiment/run_graph_diagnostics.py"
  "experiment/run_class_counts_per_fold.py"
  "experiment/run_documentation_completeness.py"
  "experiment/run_deposit_type_analysis.py"
  "experiment/run_lodto.py"
  "experiment/run_max_dist_sensitivity.py"
  "experiment/run_lithology_ablation.py"
  "experiment/run_split_strategy_sensitivity.py"
  "experiment/run_ablation.py"
  "experiment/run_ablation_exp3.py"
  "experiment/run_ablation_coordinates.py"
  "experiment/run_ablation_minerals.py"
  "experiment/run_ablation_edge_rule.py"
  "experiment/run_ablation_ensemble_scoring.py"
  "experiment/run_ablation_ensemble_only.py"
  "experiment/run_ablation_no_message_passing.py"
  "experiment/run_ablation_graph_weighting.py"
  "experiment/run_ablation_random_rewire.py"
  "experiment/run_k_sensitivity.py"
)
# Read result files produced by stage 2 (and, for the error decomposition,
# stage 3's documentation_completeness_scores.json).
STAGE4=(
  "experiment/run_pooled_auc_ci.py"
  "experiment/run_paired_significance_test.py"
  "experiment/run_calibration_analysis.py"
  "experiment/run_region_error_decomposition.py"
  "experiment/run_hparam_sensitivity.py"
)

FIGURES=(
  "experiment/run_ablation.py"
  "experiment/run_ablation_exp3.py"
  "experiment/run_k_sensitivity.py"
  "experiment/run_hparam_sensitivity.py"
)

run_one() {
  # $1 = "script.py [args...]"
  local spec="$1"
  local script="${spec%% *}"
  local name log start rc
  name="$(basename "${script%.py}")"
  log="logs/${name}.log"
  if [ ! -f "$script" ]; then
    printf 'SKIP  %-52s (missing)\n' "$script"
    return 0
  fi
  if [ -n "${RESUME:-}" ] && [ -f "logs/.done/$name" ]; then
    printf 'SKIP  %-52s (already done; unset RESUME to force)\n' "$name"
    return 0
  fi
  start=$SECONDS
  # shellcheck disable=SC2086
  if "$PY" -u $spec >"$log" 2>&1; then
    rc=0; mkdir -p logs/.done && touch "logs/.done/$name"
    printf 'OK    %-52s %6ds  -> %s\n' "$name" "$((SECONDS-start))" "$log"
  else
    rc=1; printf 'FAIL  %-52s %6ds  -> %s\n' "$name" "$((SECONDS-start))" "$log"
    tail -n 12 "$log" | sed 's/^/      | /'
    echo "$name" >> logs/.failed
  fi
  return $rc
}
export -f run_one
export PY

run_stage() {
  local label="$1"; shift
  [ "$#" -eq 0 ] && return 0
  printf '\n########## %s  (%d scripts, %d at a time, %s threads each)  %s\n' \
    "$label" "$#" "$JOBS" "$THREADS" "$(date +%H:%M:%S)"
  printf '%s\n' "$@" | xargs -d '\n' -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
}

: > logs/.failed
mkdir -p logs/.done
case "${1:-all}" in
  core)     run_stage "STAGE 1 preprocessing" "${STAGE1[@]}"
            run_stage "STAGE 2 main experiments" "${STAGE2[@]}" ;;
  figures)  run_stage "FIGURE INPUTS (priority)" "${FIGURES[@]}" ;;
  analysis) run_stage "STAGE 3 independent analyses" "${STAGE3[@]}"
            run_stage "STAGE 4 aggregations" "${STAGE4[@]}" ;;
  all)      run_stage "STAGE 1 preprocessing" "${STAGE1[@]}"
            run_stage "STAGE 2 main experiments" "${STAGE2[@]}"
            run_stage "STAGE 3 independent analyses" "${STAGE3[@]}"
            run_stage "STAGE 4 aggregations" "${STAGE4[@]}" ;;
  *) echo "usage: $0 [all|core|analysis|figures]" >&2; exit 2 ;;
esac

printf '\n===== SUMMARY  %s =====\n' "$(date +%H:%M:%S)"
if [ -s logs/.failed ]; then
  echo "failed:"; sed 's/^/  /' logs/.failed
else
  echo "all scripts completed"
fi
