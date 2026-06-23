"""
Combined figures across Exp 1, 2a, 3a (classification).
Produces:
  figures/fig_X_panel_A.png  — Panel A (avg metrics bar)     × 3 experiments
  figures/fig_Y_panel_B.png  — Panel B (radar)               × 3 experiments
  figures/fig_Z_panel_C.png  — Panel C_a (3-GNN F1 per-region) × 3 experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# ── Constants ─────────────────────────────────────────────────────────────────

COLORS_GNN = {
    "spire": "#7B2D8B",
    "gcn":       "#1A6FA8",
    "gat":       "#2A9D5C",
}
COLORS_BASE = {
    "mlp":                    "#E76F51",
    "svm":                    "#E9C46A",
    "random_forest":          "#F4A261",
    "lightgbm":               "#264653",
    "autoencoder_predictor":  "#A8DADC",
    "xgboost":                "#457B9D",
    "catboost":               "#E63946",
    "spatial_knn_regression": "#90BE6D",
    "kriging":                "#43AA8B",
}
LABELS_GNN = {"spire": "SPIRE", "gcn": "GCN", "gat": "GAT"}
LABELS_BASE = {
    "mlp": "MLP", "svm": "SVM", "random_forest": "Random Forest",
    "lightgbm": "LightGBM", "autoencoder_predictor": "Autoencoder",
    "xgboost": "XGBoost", "catboost": "CatBoost",
    "spatial_knn_regression": "Spatial kNN", "kriging": "Kriging",
}
ALL_MODELS    = list(COLORS_GNN) + list(COLORS_BASE)
METRICS_CLS   = ["f1", "roc_auc", "pr_auc"]
RADAR_METRICS = ["accuracy", "f1", "roc_auc", "pr_auc", "precision", "recall"]
RADAR_LABELS  = ["Acc", "F1", "ROC\nAUC", "PR\nAUC", "Prec", "Recall"]

REGIONS = [
    "Africa", "China", "East Asia", "Europe", "Middle East",
    "North America", "Oceania", "Russian Federation",
    "South America", "South and Central Asia",
]
REG_SHORT = {
    "Africa": "Africa", "China": "China", "East Asia": "East Asia",
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

def load_exp1():
    with open("result/baseline.json") as f:
        b = json.load(f)
    with open("result/gnn_inductive.json") as f:
        g = json.load(f)
    all_mt = METRICS_CLS + ["accuracy", "precision", "recall"]

    def _avg(runs_dict, model, metric):
        vals = [(v.get("avg_metrics", {}).get(model) or {}).get(metric)
                for v in runs_dict.values()]
        vals = [x for x in vals if x is not None]
        return _mean(vals)

    def _std_r(runs_dict, model, metric):
        vals = [(v.get("avg_metrics", {}).get(model) or {}).get(metric)
                for v in runs_dict.values()]
        vals = [x for x in vals if x is not None]
        return _std(vals)

    avg, std = {}, {}
    for m in ["spire", "gcn", "gat"]:
        avg[m] = {mt: _avg(g["runs"], m, mt) for mt in all_mt}
        std[m] = {mt: _std_r(g["runs"], m, mt) for mt in all_mt}
    for m in COLORS_BASE:
        avg[m] = {mt: _avg(b["runs"], m, mt) for mt in all_mt}
        std[m] = {mt: _std_r(b["runs"], m, mt) for mt in all_mt}

    pr = {}
    for r in REGIONS:
        pr[r] = {}
        for m in ["spire", "gcn", "gat"]:
            f1s  = [_safe(v["per_region"][r]["test_metrics"][m]["f1"])
                    for v in g["runs"].values()
                    if r in v.get("per_region", {})]
            rocs = [_safe(v["per_region"][r]["test_metrics"][m].get("roc_auc"))
                    for v in g["runs"].values()
                    if r in v.get("per_region", {})]
            pr[r][f"{m}_f1"]      = _mean(f1s)
            pr[r][f"{m}_roc_auc"] = _mean(rocs)
    return avg, std, pr


def load_exp2():
    with open("result/baseline_per_region.json") as f:
        b = json.load(f)
    with open("result/gnn_per_region.json") as f:
        g = json.load(f)
    all_mt = METRICS_CLS + ["accuracy", "precision", "recall"]
    avg, std = {}, {}
    for m in COLORS_GNN:
        vals = {mt: [_safe(r["avg_metrics"][m].get(mt)) for r in g["runs"]] for mt in all_mt}
        avg[m] = {mt: _mean(vals[mt]) for mt in all_mt}
        std[m] = {mt: _std(vals[mt])  for mt in all_mt}
    for m in COLORS_BASE:
        vals = {mt: [_safe(r["avg_metrics"][m].get(mt)) for r in b["runs"]] for mt in all_mt}
        avg[m] = {mt: _mean(vals[mt]) for mt in all_mt}
        std[m] = {mt: _std(vals[mt])  for mt in all_mt}
    pr = {}
    for r in REGIONS:
        pr[r] = {}
        for m in ["spire", "gcn", "gat"]:
            f1s  = [_safe(run["per_region"][r]["test_metrics"][m]["f1"])          for run in g["runs"]]
            rocs = [_safe(run["per_region"][r]["test_metrics"][m].get("roc_auc")) for run in g["runs"]]
            pr[r][f"{m}_f1"]      = _mean(f1s)
            pr[r][f"{m}_roc_auc"] = _mean(rocs)
    return avg, std, pr


def load_exp3():
    with open("result/inductive_sample.json") as f:
        s = json.load(f)
    all_mt = METRICS_CLS + ["accuracy", "precision", "recall"]
    avg, std = {}, {}
    for m in COLORS_GNN:
        vals = {mt: [_safe(r[m]["avg_metrics"].get(mt)) for r in s["runs"]] for mt in all_mt}
        avg[m] = {mt: _mean(vals[mt]) for mt in all_mt}
        std[m] = {mt: _std(vals[mt])  for mt in all_mt}
    for m in COLORS_BASE:
        vals = {mt: [_safe(r["baselines"]["avg_metrics"][m].get(mt)) for r in s["runs"]] for mt in all_mt}
        avg[m] = {mt: _mean(vals[mt]) for mt in all_mt}
        std[m] = {mt: _std(vals[mt])  for mt in all_mt}
    pr = {}
    for r in REGIONS:
        pr[r] = {}
        for m in ["spire", "gcn", "gat"]:
            f1s, rocs = [], []
            for run in s["runs"]:
                tm = run[m]["per_region"][r].get("test_metrics", run[m]["per_region"][r])
                f1s.append(_safe(tm.get("f1")))
                rocs.append(_safe(tm.get("roc_auc")))
            pr[r][f"{m}_f1"]      = _mean(f1s)
            pr[r][f"{m}_roc_auc"] = _mean(rocs)
    return avg, std, pr


# ── Panel drawers ─────────────────────────────────────────────────────────────

def draw_panel_a(ax, avg, title, std=None):
    x       = np.arange(len(ALL_MODELS))
    w       = 0.26
    offsets = np.array([-w, 0, w])
    alphas  = [0.88, 0.70, 0.55]
    for metric, offset, alpha in zip(METRICS_CLS, offsets, alphas):
        vals   = [avg.get(m, {}).get(metric, float("nan")) for m in ALL_MODELS]
        errs   = [std[m][metric] for m in ALL_MODELS] if std else None
        colors = [COLORS_GNN.get(m, COLORS_BASE.get(m, "#999")) for m in ALL_MODELS]
        ax.bar(x + offset, vals, width=w, color=colors,
               alpha=alpha, edgecolor="white", linewidth=0.5, zorder=3,
               yerr=errs, error_kw=dict(ecolor="#444", capsize=2,
                                        linewidth=0.7, zorder=4))
    model_labels = [LABELS_GNN.get(m, LABELS_BASE.get(m, m)) for m in ALL_MODELS]
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9.5, rotation=25, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
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
        handles=[mpatches.Patch(facecolor="#888", alpha=a, label=l)
                 for l, a in [("F1", 0.88), ("ROC-AUC", 0.70), ("PR-AUC", 0.55)]],
        loc="lower right", fontsize=9, framealpha=0.8,
        title="Metric", title_fontsize=9,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


def draw_panel_b(ax, avg, title):
    N      = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    radar_models = {
        "SPIRE":     ("spire",     COLORS_GNN["spire"],      "-",  2.2),
        "GCN":           ("gcn",           COLORS_GNN["gcn"],            "-",  1.4),
        "GAT":           ("gat",           COLORS_GNN["gat"],            "-",  1.4),
        "XGBoost":       ("xgboost",       COLORS_BASE["xgboost"],       "--", 1.4),
        "MLP":           ("mlp",           COLORS_BASE["mlp"],           "--", 1.4),
        "Random Forest": ("random_forest", COLORS_BASE["random_forest"], "--", 1.4),
    }
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0.5, 1.0)
    ax.set_rticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=7, color="#777")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, fontsize=9)
    ax.set_facecolor("#FAFAFA")
    ax.grid(color="#CCCCCC", linestyle="--", alpha=0.5)
    for name, (key, color, ls, lw) in radar_models.items():
        vals = [avg.get(key, {}).get(mt, float("nan")) for mt in RADAR_METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=lw, linestyle=ls, label=name)
        ax.fill(angles, vals, color=color, alpha=0.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30),
              fontsize=7.5, framealpha=0.85, ncol=2)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=18)


def draw_panel_c_gnn(ax, per_region, title, show_xlabel=True):
    reg_labels = [REG_SHORT[r] for r in REGIONS]
    x2       = np.arange(len(REGIONS))
    gnn_list = [("spire", "SPIRE"), ("gcn", "GCN"), ("gat", "GAT")]
    w2       = 0.26
    offsets2 = np.array([-w2, 0, w2])
    for (m, label), off in zip(gnn_list, offsets2):
        f1_vals = [per_region[r][f"{m}_f1"] for r in REGIONS]
        bars = ax.bar(x2 + off, f1_vals, width=w2,
                      color=COLORS_GNN[m], alpha=0.85,
                      edgecolor="white", linewidth=0.5, zorder=3, label=label)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.006,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6.5,
                        color=COLORS_GNN[m], fontweight="bold")
    for m, _ in gnn_list:
        avg_f1 = np.nanmean([per_region[r][f"{m}_f1"] for r in REGIONS])
        ax.axhline(avg_f1, color=COLORS_GNN[m], linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(len(REGIONS) - 0.4, avg_f1 + 0.006,
                f"avg {LABELS_GNN[m]}={avg_f1:.3f}",
                fontsize=7, color=COLORS_GNN[m], va="bottom", ha="right")
    ax.set_xticks(x2)
    if show_xlabel:
        ax.set_xticklabels(reg_labels, fontsize=8.5, rotation=30, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold")


# ── Figure X — Panel A × 3 ───────────────────────────────────────────────────

def make_fig_x(data_list):
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
    plt.savefig("figures/fig_X_panel_A.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → figures/fig_X_panel_A.png")
    plt.close()


# ── Figure Y — Panel B × 3 ───────────────────────────────────────────────────

def make_fig_y(data_list):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8),
                             subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#F8F9FA")
    fig.subplots_adjust(wspace=0.55)
    for ax, (avg, _, __, title) in zip(axes, data_list):
        ax.set_facecolor("#FAFAFA")
        draw_panel_b(ax, avg, title)
    for ax, lbl in zip(axes, ["(1)", "(2)", "(3)"]):
        ax.annotate(lbl, xy=(-0.08, 1.10), xycoords="axes fraction",
                    fontsize=12, fontweight="bold", color="#1A1A2E",
                    annotation_clip=False)
    plt.savefig("figures/fig_Y_panel_B.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → figures/fig_Y_panel_B.png")
    plt.close()


# ── Figure Z — Panel C_a × 3 ─────────────────────────────────────────────────

def make_fig_z(data_list):
    fig, axes = plt.subplots(3, 1, figsize=(18, 18))
    fig.patch.set_facecolor("#F8F9FA")
    fig.subplots_adjust(hspace=0.25)
    for i, (ax, (_, _std, pr, title)) in enumerate(zip(axes, data_list)):
        ax.set_facecolor("#F8F9FA")
        draw_panel_c_gnn(ax, pr, title, show_xlabel=(i == 2))
    for ax, lbl in zip(axes, ["(1)", "(2)", "(3)"]):
        ax.annotate(lbl, xy=(-0.02, 1.03), xycoords="axes fraction",
                    fontsize=12, fontweight="bold", color="#1A1A2E",
                    annotation_clip=False)
    plt.savefig("figures/fig_Z_panel_C.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → figures/fig_Z_panel_C.png")
    plt.close()


# ── Run ───────────────────────────────────────────────────────────────────────

avg1,  std1,  pr1  = load_exp1()
avg2, std2, pr2 = load_exp2()
avg3, std3, pr3 = load_exp3()

data_list = [
    (avg1,  std1,  pr1,  "Exp 1 — Leave-one-region-out: Average Metrics"),
    (avg2, std2, pr2, "Exp 2 — Per-Region, Random Split: Average Metrics"),
    (avg3, std3, pr3, "Exp 3 — Global Inductive, Random Split: Average Metrics"),
]

make_fig_x(data_list)
make_fig_y(data_list)
make_fig_z(data_list)
