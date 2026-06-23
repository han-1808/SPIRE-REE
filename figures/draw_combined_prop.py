"""
Combined figures for Exp 4, 5a, 6a (proportion regression).
Produces:
  figures/fig_A_panel_A.png  — Panel A (MAE+RMSE bar)              × 3 experiments
  figures/fig_B_panel_B.png  — Panel B (actual vs predicted scatter) × 3 experiments
  figures/fig_C_panel_C.png  — Panel C (GNN abs_error per-region)   × 3 experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Constants ─────────────────────────────────────────────────────────────────

COLORS_GNN = {
    "spire": "#7B2D8B",
    "gcn":       "#1A6FA8",
    "gat":       "#2A9D5C",
}
COLORS_BASE = {
    "mlp":           "#E76F51",
    "svm":           "#E9C46A",
    "random_forest": "#F4A261",
    "lightgbm":      "#264653",
    "autoencoder":   "#A8DADC",
    "xgboost":       "#457B9D",
    "catboost":      "#E63946",
    "spatial_knn":   "#90BE6D",
    "kriging":       "#43AA8B",
}
LABELS_GNN = {"spire": "SPIRE", "gcn": "GCN", "gat": "GAT"}
LABELS_BASE = {
    "mlp": "MLP", "svm": "SVM", "random_forest": "Random Forest",
    "lightgbm": "LightGBM", "autoencoder": "Autoencoder",
    "xgboost": "XGBoost", "catboost": "CatBoost",
    "spatial_knn": "Spatial kNN", "kriging": "Kriging",
}
ALL_MODELS = list(COLORS_GNN) + list(COLORS_BASE)
METRICS    = ["mae", "rmse"]

REGIONS = [
    "Africa", "China", "East Asia", "Europe", "Middle East",
    "North America", "Oceania", "Russian Federation",
    "South America", "South and Central Asia",
]
REG_SHORT = {
    "Africa": "Africa", "China": "China", "East Asia": "E. Asia",
    "Europe": "Europe", "Middle East": "Mid. East",
    "North America": "N. America", "Oceania": "Oceania",
    "Russian Federation": "Russia", "South America": "S. America",
    "South and Central Asia": "S.C. Asia",
}


def _safe(v):
    return float(v) if v is not None else float("nan")


def _mean(vals):
    clean = [v for v in vals if not np.isnan(v)]
    return np.mean(clean) if clean else float("nan")


def _std(vals):
    clean = [v for v in vals if not np.isnan(v)]
    return np.std(clean) if len(clean) > 1 else 0.0


# ── Data loaders ──────────────────────────────────────────────────────────────

def _parse_per_region(pr_data):
    per_region = {}
    for r in REGIONS:
        if r not in pr_data:
            continue
        entry = pr_data[r]
        per_region[r] = {"actual_proportion": _safe(entry.get("actual_proportion"))}
        pm = entry.get("per_model", {})
        for m in ALL_MODELS:
            per_region[r][f"{m}_predicted"] = _safe(pm[m].get("predicted_proportion")) if m in pm else float("nan")
            per_region[r][f"{m}_abs_error"]  = _safe(pm[m].get("abs_error"))           if m in pm else float("nan")
    return per_region


def load_exp4():
    with open("result/exp4_proportion.json") as f:
        d = json.load(f)
    runs = list(d["runs"].values())   # dict → list of run dicts
    avg, std = {}, {}
    for m in ALL_MODELS:
        vals = {mt: [_safe(r["aggregated_metrics"][m].get(mt))
                     for r in runs if m in r.get("aggregated_metrics", {})]
                for mt in METRICS}
        avg[m] = {mt: _mean(vals[mt]) for mt in METRICS}
        std[m] = {mt: _std(vals[mt])  for mt in METRICS}
    per_region = {}
    for r in REGIONS:
        per_region[r] = {
            "actual_proportion": _mean([
                _safe(run["per_region"][r]["actual_proportion"])
                for run in runs if r in run.get("per_region", {})
            ])
        }
        for m in ALL_MODELS:
            preds = [_safe(run["per_region"][r]["per_model"][m]["predicted_proportion"])
                     for run in runs
                     if r in run.get("per_region", {})
                     and m in run["per_region"][r].get("per_model", {})]
            errs  = [_safe(run["per_region"][r]["per_model"][m]["abs_error"])
                     for run in runs
                     if r in run.get("per_region", {})
                     and m in run["per_region"][r].get("per_model", {})]
            per_region[r][f"{m}_predicted"] = _mean(preds)
            per_region[r][f"{m}_abs_error"]  = _mean(errs)
    return avg, std, per_region


def _load_runs(fname):
    with open(fname) as f:
        d = json.load(f)
    runs = d["runs"]
    avg, std = {}, {}
    for m in ALL_MODELS:
        vals = {mt: [_safe(r["aggregated_metrics"][m].get(mt))
                     for r in runs if m in r.get("aggregated_metrics", {})]
                for mt in METRICS}
        avg[m] = {mt: _mean(vals[mt]) for mt in METRICS}
        std[m] = {mt: _std(vals[mt])  for mt in METRICS}
    per_region = {}
    for r in REGIONS:
        per_region[r] = {
            "actual_proportion": _mean([_safe(run["per_region"][r]["actual_proportion"])
                                        for run in runs if r in run.get("per_region", {})])
        }
        for m in ALL_MODELS:
            preds = [_safe(run["per_region"][r]["per_model"][m]["predicted_proportion"])
                     for run in runs
                     if r in run.get("per_region", {}) and m in run["per_region"][r].get("per_model", {})]
            errs  = [_safe(run["per_region"][r]["per_model"][m]["abs_error"])
                     for run in runs
                     if r in run.get("per_region", {}) and m in run["per_region"][r].get("per_model", {})]
            per_region[r][f"{m}_predicted"] = _mean(preds)
            per_region[r][f"{m}_abs_error"]  = _mean(errs)
    return avg, std, per_region


def load_exp5():
    return _load_runs("result/exp5_proportion_per_region.json")


def load_exp6():
    return _load_runs("result/exp6_proportion_inductive_sample.json")


# ── Panel drawers ─────────────────────────────────────────────────────────────

def draw_panel_a(ax, avg, title, std=None):
    x       = np.arange(len(ALL_MODELS))
    w       = 0.35
    offsets = np.array([-w / 2, w / 2])
    alphas  = [0.88, 0.60]
    for metric, offset, alpha in zip(METRICS, offsets, alphas):
        vals   = [avg.get(m, {}).get(metric, float("nan")) for m in ALL_MODELS]
        errs   = [std[m][metric] for m in ALL_MODELS] if std else None
        colors = [COLORS_GNN.get(m, COLORS_BASE.get(m, "#999")) for m in ALL_MODELS]
        ax.bar(x + offset, vals, width=w, color=colors,
               alpha=alpha, edgecolor="white", linewidth=0.5, zorder=3,
               yerr=errs, error_kw=dict(ecolor="#444", capsize=2,
                                        linewidth=0.7, zorder=4))
    ax.set_ylim(0, 0.3)
    model_labels = [LABELS_GNN.get(m, LABELS_BASE.get(m, m)) for m in ALL_MODELS]
    ax.set_xticks(np.arange(len(ALL_MODELS)))
    ax.set_xticklabels(model_labels, fontsize=9.5, rotation=25, ha="right")
    ax.set_ylabel("Error (lower is better)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(x=2.5, color="#AAAAAA", linestyle=":", linewidth=1.3, zorder=4)
    ax.text(1.0, 0.995, "GNN", ha="center", va="top",
            fontsize=9, color="#555", fontstyle="italic",
            transform=ax.get_xaxis_transform())
    ax.text(7.0, 0.995, "Baselines", ha="center", va="top",
            fontsize=9, color="#555", fontstyle="italic",
            transform=ax.get_xaxis_transform())
    ax.legend(
        handles=[mpatches.Patch(facecolor="#888", alpha=0.88, label="MAE"),
                 mpatches.Patch(facecolor="#888", alpha=0.60, label="RMSE")],
        loc="lower right", fontsize=9, framealpha=0.8,
        title="Metric", title_fontsize=9,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


def draw_panel_b(ax, avg, per_region, title):
    reg_list = [r for r in REGIONS if r in per_region]
    actual   = [per_region[r]["actual_proportion"] for r in reg_list]

    # top-2 baselines by MAE
    base_keys   = list(COLORS_BASE.keys())
    top2_bases  = sorted(base_keys, key=lambda m: avg.get(m, {}).get("mae", float("inf")))[:2]

    scatter_models = [("spire", "SPIRE", COLORS_GNN["spire"], "o")]
    for m, marker in zip(top2_bases, ["s", "^"]):
        scatter_models.append((m, LABELS_BASE[m], COLORS_BASE[m], marker))

    for m, label, color, marker in scatter_models:
        pred = [per_region[r][f"{m}_predicted"] for r in reg_list]
        ax.scatter(actual, pred, color=color, marker=marker,
                   s=55, zorder=3, label=label, alpha=0.85)

    for r, a in zip(reg_list, actual):
        p = per_region[r]["spire_predicted"]
        ax.annotate(REG_SHORT[r], (a, p), fontsize=6.5,
                    color=COLORS_GNN["spire"],
                    xytext=(3, 2), textcoords="offset points")

    ax.plot([0, 1.05], [0, 1.05], color="#999", linewidth=1.2,
            linestyle="--", zorder=2, label="y = x")
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Actual Proportion", fontsize=10)
    ax.set_ylabel("Predicted Proportion", fontsize=10)
    ax.set_facecolor("#FAFAFA")
    ax.grid(linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=24)


def draw_panel_c(ax, per_region, title, show_xlabel=True):
    reg_list   = [r for r in REGIONS if r in per_region]
    reg_labels = [REG_SHORT[r] for r in reg_list]
    x2       = np.arange(len(reg_list))
    gnn_list = [
        ("spire", COLORS_GNN["spire"], "SPIRE"),
        ("gcn",       COLORS_GNN["gcn"],       "GCN"),
        ("gat",       COLORS_GNN["gat"],       "GAT"),
    ]
    w2       = 0.26
    offsets2 = np.arange(3) * w2 - w2

    for (m, color, label), off in zip(gnn_list, offsets2):
        errs = [per_region[r][f"{m}_abs_error"] for r in reg_list]
        ax.bar(x2 + off, errs, width=w2, color=color,
               alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3, label=label)

    avg_sage = np.nanmean([per_region[r]["spire_abs_error"] for r in reg_list])
    ax.axhline(avg_sage, color=COLORS_GNN["spire"],
               linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(len(reg_list) - 0.4, avg_sage + 0.003,
            f"avg SPIRE={avg_sage:.3f}",
            fontsize=7.5, color=COLORS_GNN["spire"], va="bottom", ha="right")

    ax.set_xticks(x2)
    if show_xlabel:
        ax.set_xticklabels(reg_labels, fontsize=8.5, rotation=30, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_ylim(0, 0.4)
    ax.set_ylabel("|Predicted − Actual|", fontsize=10)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold")


# ── Figure A — Panel A × 3 ───────────────────────────────────────────────────

def make_fig_a(data_list):
    fig, axes = plt.subplots(3, 1, figsize=(18, 21))
    fig.patch.set_facecolor("#F8F9FA")
    fig.subplots_adjust(hspace=0.3)
    for ax, (avg, std, _, title) in zip(axes, data_list):
        ax.set_facecolor("#F8F9FA")
        draw_panel_a(ax, avg, title, std=std)
    for ax, lbl in zip(axes, ["(1)", "(2)", "(3)"]):
        ax.annotate(lbl, xy=(-0.02, 1.03), xycoords="axes fraction",
                    fontsize=12, fontweight="bold", color="#1A1A2E",
                    annotation_clip=False)
    plt.savefig("figures/fig_A_panel_A.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → figures/fig_A_panel_A.png")
    plt.close()


# ── Figure B — Panel B × 3 ───────────────────────────────────────────────────

def make_fig_b(data_list):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#F8F9FA")
    fig.subplots_adjust(wspace=0.55)
    for ax, (avg, _, pr, title) in zip(axes, data_list):
        ax.set_facecolor("#F8F9FA")
        draw_panel_b(ax, avg, pr, title)
    for ax, lbl in zip(axes, ["(1)", "(2)", "(3)"]):
        ax.annotate(lbl, xy=(-0.08, 1.02), xycoords="axes fraction",
                    fontsize=12, fontweight="bold", color="#1A1A2E",
                    annotation_clip=False)
    plt.savefig("figures/fig_B_panel_B.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → figures/fig_B_panel_B.png")
    plt.close()


# ── Figure C — Panel C × 3 ───────────────────────────────────────────────────

def make_fig_c(data_list):
    fig, axes = plt.subplots(3, 1, figsize=(18, 18))
    fig.patch.set_facecolor("#F8F9FA")
    fig.subplots_adjust(hspace=0.25)
    for i, (ax, (_, _std, pr, title)) in enumerate(zip(axes, data_list)):
        ax.set_facecolor("#F8F9FA")
        draw_panel_c(ax, pr, title, show_xlabel=(i == 2))
    for ax, lbl in zip(axes, ["(1)", "(2)", "(3)"]):
        ax.annotate(lbl, xy=(-0.02, 1.03), xycoords="axes fraction",
                    fontsize=12, fontweight="bold", color="#1A1A2E",
                    annotation_clip=False)
    plt.savefig("figures/fig_C_panel_C.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → figures/fig_C_panel_C.png")
    plt.close()


# ── Run ───────────────────────────────────────────────────────────────────────

avg4,  std4,  pr4  = load_exp4()
avg5, std5, pr5 = load_exp5()
avg6, std6, pr6 = load_exp6()

data_list = [
    (avg4,  std4,  pr4,  "Exp 4 — Global Proportion: Error Metrics"),
    (avg5, std5, pr5, "Exp 5 — Per-Region Proportion: Error Metrics"),
    (avg6, std6, pr6, "Exp 6 — Global Inductive Proportion: Error Metrics"),
]

make_fig_a(data_list)
make_fig_b(data_list)
make_fig_c(data_list)
