"""
report_old_vs_new.py

Builds the "old versus new pipeline" comparison for the six main experiments
directly from the stored result/*.json files, so every number in the response
letter can be traced to a file on disk rather than transcribed by hand.

    python report_old_vs_new.py --old /path/to/previous/result

`--old` points at a directory of result files from the pipeline being compared
against (the pre-revision run).  Without it, only the current results are
printed.  Experiments 1-3 are classification and report F1 / ROC-AUC / PR-AUC;
experiments 4-6 are region-level proportion prediction and report MAE / RMSE,
where lower is better.

Each result file stores its runs in a slightly different shape (Exp 1 keeps a
precomputed cross-seed `avg_metrics`, Exp 2/3/5/6 keep a list of per-seed runs,
Exp 4 keeps a dict keyed by seed), so one reader per layout is defined below
and everything downstream works on a uniform {model: {metric: mean}} mapping.
"""

import argparse
import json
import statistics
from pathlib import Path

# Baseline model keys are spelled slightly differently across files
# (autoencoder vs autoencoder_predictor, spatial_knn vs spatial_knn_regression).
_GNN_MODELS = {"spire", "gcn", "gat"}
_PRETTY = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
    "random_forest": "Random Forest", "svm": "SVM", "mlp": "MLP",
    "autoencoder": "Autoencoder", "autoencoder_predictor": "Autoencoder",
    "spatial_knn": "Spatial k-NN", "spatial_knn_regression": "Spatial k-NN",
    "kriging": "Kriging", "spire": "SPIRE", "gcn": "GCN", "gat": "GAT",
}


def _mean_over(dicts: list[dict]) -> dict:
    """Average a list of {model: {metric: value}} into one such mapping."""
    out: dict = {}
    models = {m for d in dicts for m in d}
    for model in models:
        per_metric: dict = {}
        for d in dicts:
            for metric, value in (d.get(model) or {}).items():
                if isinstance(value, (int, float)):
                    per_metric.setdefault(metric, []).append(float(value))
        if per_metric:
            out[model] = {k: statistics.fmean(v) for k, v in per_metric.items()}
    return out


def _load(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_exp1(result_dir: Path) -> dict | None:
    """
    Exp 1: GNNs in gnn_inductive.json, baselines in baseline.json.

    Both are recomputed from the per-seed `runs` rather than read from each
    file's top-level `avg_metrics`. That field is not trustworthy in files
    written before this revision: when extra seeds were merged into an existing
    result file, the top-level average was not always recomputed, so
    gnn_inductive.json reports 0.8266 (seed 42 alone) where the mean across its
    five seeds is 0.8437 -- the latter being the number published in the paper
    (0.844 +/- 0.009, Table `tab:per_region_f1`). Averaging the per-seed values
    reproduces every published per-region figure exactly.
    """
    gnn, base = _load(result_dir / "gnn_inductive.json"), _load(result_dir / "baseline.json")
    if gnn is None and base is None:
        return None

    def from_runs(doc):
        runs = doc.get("runs") if doc else None
        if not runs:
            return {}
        runs = list(runs.values()) if isinstance(runs, dict) else runs
        acc: dict = {}
        for r in runs:
            for model, metrics in (r.get("avg_metrics") or {}).items():
                for k, v in (metrics or {}).items():
                    if isinstance(v, (int, float)):
                        acc.setdefault(model, {}).setdefault(k, []).append(float(v))
        return {m: {k: statistics.fmean(v) for k, v in d.items()} for m, d in acc.items()}

    merged: dict = {}
    merged.update(from_runs(gnn) or (gnn or {}).get("avg_metrics", {}))
    merged.update(from_runs(base) or (base or {}).get("avg_metrics", {}))
    return merged or None


def read_exp2(result_dir: Path) -> dict | None:
    """Exp 2: per-seed run lists, GNNs and baselines in separate files."""
    gnn, base = _load(result_dir / "gnn_per_region.json"), _load(result_dir / "baseline_per_region.json")
    if gnn is None and base is None:
        return None
    parts = []
    for doc in (gnn, base):
        if not doc:
            continue
        runs = doc.get("runs", [])
        runs = list(runs.values()) if isinstance(runs, dict) else runs
        parts.append(_mean_over([r.get("avg_metrics", {}) for r in runs]))
    return _mean_over(parts) or None


def read_exp3(result_dir: Path) -> dict | None:
    """Exp 3: each run holds spire/gcn/gat plus a nested `baselines` block."""
    doc = _load(result_dir / "inductive_sample.json")
    if doc is None:
        return None
    runs = doc.get("runs", [])
    runs = list(runs.values()) if isinstance(runs, dict) else runs
    per_run = []
    for run in runs:
        flat: dict = {}
        for model in _GNN_MODELS:
            if model in run:
                flat[model] = run[model].get("avg_metrics", {})
        flat.update((run.get("baselines") or {}).get("avg_metrics", {}) or {})
        per_run.append(flat)
    return _mean_over(per_run) or None


def read_proportion(result_dir: Path, filename: str) -> dict | None:
    """Exp 4/5/6: `aggregated_metrics` per seed, or precomputed at top level."""
    doc = _load(result_dir / filename)
    if doc is None:
        return None
    runs = doc.get("runs")
    if runs:
        runs = list(runs.values()) if isinstance(runs, dict) else runs
        agg = [r.get("aggregated_metrics", {}) for r in runs if r.get("aggregated_metrics")]
        if agg:
            return _mean_over(agg)
    return doc.get("aggregated_metrics") or None


_EXPERIMENTS = [
    ("1", "Leave-one-region-out (classification)", "f1", True, read_exp1),
    ("2", "Per-region split (classification)", "f1", True, read_exp2),
    ("3", "Global inductive split (classification)", "f1", True, read_exp3),
    ("4", "Leave-one-region-out (proportion)", "mae", False,
     lambda d: read_proportion(d, "exp4_proportion.json")),
    ("5", "Per-region split (proportion)", "mae", False,
     lambda d: read_proportion(d, "exp5_proportion_per_region.json")),
    ("6", "Global inductive split (proportion)", "mae", False,
     lambda d: read_proportion(d, "exp6_proportion_inductive_sample.json")),
]


def _best_baseline(metrics: dict, key: str, higher_is_better: bool):
    """Strongest non-GNN model on `key`, returned as (pretty name, value)."""
    cands = [
        (m, v[key]) for m, v in metrics.items()
        if m not in _GNN_MODELS and isinstance(v.get(key), (int, float))
    ]
    if not cands:
        return None, None
    name, value = (max if higher_is_better else min)(cands, key=lambda t: t[1])
    return _PRETTY.get(name, name), value


def _fmt(value):
    return "  --  " if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", default="result", help="current result directory")
    parser.add_argument("--old", default=None, help="previous result directory to compare against")
    args = parser.parse_args()

    new_dir = Path(args.new)
    old_dir = Path(args.old) if args.old else None

    print()
    print("SPIRE vs strongest tabular baseline, old vs new pipeline")
    print("Exp 1-3: F1 (higher is better).  Exp 4-6: MAE (lower is better).")
    print("=" * 100)
    header = f"{'Exp':<4}{'Metric':<7}"
    if old_dir:
        header += f"{'old SPIRE':>11}{'old best baseline':>26}"
    header += f"{'new SPIRE':>11}{'new best baseline':>26}{'delta':>9}"
    print(header)
    print("-" * 100)

    rows = []
    for exp, label, key, higher, reader in _EXPERIMENTS:
        new = reader(new_dir) or {}
        old = (reader(old_dir) or {}) if old_dir else {}

        new_spire = (new.get("spire") or {}).get(key)
        old_spire = (old.get("spire") or {}).get(key)
        new_bname, new_bval = _best_baseline(new, key, higher)
        old_bname, old_bval = _best_baseline(old, key, higher)

        delta = None
        if new_spire is not None and old_spire is not None:
            delta = new_spire - old_spire

        line = f"{exp:<4}{key.upper():<7}"
        if old_dir:
            line += f"{_fmt(old_spire):>11}"
            line += f"{(_fmt(old_bval) + ' (' + (old_bname or '-') + ')'):>26}"
        line += f"{_fmt(new_spire):>11}"
        line += f"{(_fmt(new_bval) + ' (' + (new_bname or '-') + ')'):>26}"
        line += f"{('  --  ' if delta is None else f'{delta:+.3f}'):>9}"
        print(line)
        rows.append({
            "experiment": exp, "description": label, "metric": key,
            "higher_is_better": higher,
            "old_spire": old_spire, "old_best_baseline": old_bname, "old_best_baseline_value": old_bval,
            "new_spire": new_spire, "new_best_baseline": new_bname, "new_best_baseline_value": new_bval,
            "delta_spire": delta,
        })

    print("=" * 100)

    # Full SPIRE / GCN / GAT detail for the classification experiments.
    print("\nSPIRE detail, classification experiments (new pipeline)")
    print("-" * 100)
    print(f"{'Exp':<4}{'Model':<8}{'F1':>9}{'ROC-AUC':>10}{'PR-AUC':>10}")
    for exp, _, key, _, reader in _EXPERIMENTS[:3]:
        new = reader(new_dir) or {}
        for model in ("spire", "gcn", "gat"):
            m = new.get(model)
            if not m:
                continue
            print(f"{exp:<4}{_PRETTY[model]:<8}{_fmt(m.get('f1')):>9}"
                  f"{_fmt(m.get('roc_auc')):>10}{_fmt(m.get('pr_auc')):>10}")

    out = new_dir / "old_vs_new_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"old_dir": str(old_dir) if old_dir else None,
                   "new_dir": str(new_dir), "rows": rows}, f, indent=2)
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
