"""
visualize_case_study.py

World scatter map for Exp 1 (leave-one-region-out inductive).
Compares SPIRE vs XGBoost F1, annotating regions where SPIRE wins.

Output:
  figures/fig_world_overview_exp1.png

Usage:
    python visualize_case_study.py [--save-dir figures]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from config import OUTPUT_PATH
from feature_engineering import create_spatial_split_indices


REGION_COLORS = {
    "Africa":                   "#E41A1C",
    "China":                    "#FF7F00",
    "East Asia":                "#4DAF4A",
    "Europe":                   "#377EB8",
    "Middle East":              "#984EA3",
    "North America":            "#A65628",
    "Oceania":                  "#F781BF",
    "Russian Federation":       "#888888",
    "South America":            "#BCBD22",
    "South and Central Asia":   "#17BECF",
}

ANNOT_OFFSETS = {
    "China":              (  90,  -25),
    "East Asia":          (  60,   30),
    "Europe":             (  -80,  -20),
    "Middle East":        (  60,   20),
    "Russian Federation": (   0,   50),
    "South America":      ( -90,   25),
    "Africa":             ( -60,   20),
    "North America":      ( -60,   40),
    "Oceania":            (  60,  -40),
    "South and Central Asia": (60, -30),
}


def build_country_to_region(df):
    """Majority-vote Country -> Region, from the same `Region` field used
    everywhere else. A few countries (China, Canada, Russian Federation)
    have a handful of records assigned to a neighboring region; the
    country as a whole is shaded by whichever region holds most of its
    records (e.g. Canada: 251 North America vs 3 Europe)."""
    counts = df.assign(Country=df["Country"].str.strip()).groupby(
        ["Country", "Region"]
    ).size()
    return counts.groupby(level=0).idxmax().apply(lambda t: t[1]).to_dict()


def add_region_country_shading(ax, country_to_region):
    """Fill each country polygon with its assigned region's color, giving
    a country-outline/legend-by-region view (Comment 5-B) instead of only
    scatter points. Natural Earth's `NAME` field matches our `Country`
    values directly for nearly every country; `NAME_LONG` is used as a
    fallback for the handful that don't (e.g. long-form country names)."""
    shp_path = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    matched, unmatched = 0, []
    for record in shpreader.Reader(shp_path).records():
        name = record.attributes.get("NAME", "").strip()
        name_long = record.attributes.get("NAME_LONG", "").strip()
        region = country_to_region.get(name) or country_to_region.get(name_long)
        if region is None:
            continue
        matched += 1
        ax.add_geometries(
            [record.geometry], crs=ccrs.PlateCarree(),
            facecolor=REGION_COLORS.get(region, "#999999"),
            edgecolor="white", linewidth=0.3, alpha=0.30, zorder=0.5,
        )
    print(f"  Country shading: {matched}/{len(country_to_region)} countries matched to a region")


def get_test_coords(df, n_sample=5, seed=42):
    """For Exp 1: ALL nodes in each region are test nodes. Sample n_sample for display."""
    rng = np.random.default_rng(seed)
    out = {}
    for region in sorted(df["Region"].dropna().unique()):
        split = create_spatial_split_indices(df, test_region=region, val_ratio=0.2, random_seed=seed)
        test_idx = split["test_idx"]
        if len(test_idx) == 0:
            continue
        sample = rng.choice(test_idx, size=min(n_sample, len(test_idx)), replace=False)
        out[region] = df.iloc[sample][["Latitude", "Longitude", "has_ree"]].reset_index(drop=True)
    return out


def compute_metrics(gnn, bl):
    """Avg SPIRE vs XGBoost F1 per region across all seeds."""
    seeds = list(gnn["runs"].keys())
    out = {}
    for region in gnn["regions"]:
        spire_f1s, xgb_f1s = [], []
        for seed in seeds:
            try:
                spire_f1 = gnn["runs"][seed]["per_region"][region]["test_metrics"]["spire"]["f1"]
                xgb_f1   = bl["runs"][seed]["per_region"][region]["xgboost"]["test_metrics"]["f1"]
                spire_f1s.append(spire_f1)
                xgb_f1s.append(xgb_f1)
            except Exception:
                pass
        if spire_f1s:
            out[region] = {
                "spire_f1": float(np.mean(spire_f1s)),
                "xgb_f1":   float(np.mean(xgb_f1s)),
                "f1_diff":  float(np.mean(spire_f1s) - np.mean(xgb_f1s)),
            }
    return out


def make_world_overview(df, test_coords, metrics, save_path):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([-180, 180, -75, 85], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN,     facecolor="#AED6EF", zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor="#F5F0E8", zorder=0)
    add_region_country_shading(ax, build_country_to_region(df))
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#999999", zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#BBBBBB", zorder=1)
    ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                 alpha=0.5, linestyle="--", x_inline=False, y_inline=False)

    for region in sorted(df["Region"].dropna().unique()):
        rdf = df[df["Region"] == region]
        ax.scatter(rdf["Longitude"], rdf["Latitude"],
                   c=REGION_COLORS.get(region, "#999999"),
                   s=6, alpha=0.35, linewidths=0, zorder=2,
                   transform=ccrs.PlateCarree())

    for region, tc in test_coords.items():
        ax.scatter(tc["Longitude"], tc["Latitude"],
                   s=90, c="white", edgecolors="black", linewidths=1.2,
                   zorder=4, alpha=0.95, transform=ccrs.PlateCarree())
        ax.scatter(tc["Longitude"], tc["Latitude"],
                   s=35,
                   c=tc["has_ree"].map({1: "#FFD700", 0: "#222222"}),
                   edgecolors="none", zorder=5, transform=ccrs.PlateCarree())

    for region, m in metrics.items():
        tc = test_coords.get(region)
        if tc is None:
            continue
        if m["f1_diff"] <= 0.02:
            continue
        clat = tc["Latitude"].median()
        clon = tc["Longitude"].median()
        dx, dy = ANNOT_OFFSETS.get(region, (0, 30))
        label = (
            f"{region}\n"
            f"SPIRE F1={m['spire_f1']:.2f}\n"
            f"XGBoost  F1={m['xgb_f1']:.2f}  (Δ={m['f1_diff']:+.2f})"
        )
        color = "#003399"
        fc    = "#DDE8FF"
        ax.annotate(
            label,
            xy=(clon, clat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8, color=color, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.35", facecolor=fc,
                      edgecolor=color, alpha=0.93),
        )

    ax.set_title(
        "Global REE Deposit Distribution — Test Nodes Highlighted (Exp 1)\n"
        "Annotated: regions where SPIRE F1 > XGBoost  (avg over 5 seeds)",
        fontsize=11, fontweight="bold",
    )

    region_patches = [
        mpatches.Patch(facecolor=REGION_COLORS.get(r, "#999999"), label=r)
        for r in sorted(REGION_COLORS)
    ]
    node_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFD700",
               markeredgecolor="black", markersize=9, label="Test node — REE = 1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#222222",
               markeredgecolor="black", markersize=9, label="Test node — REE = 0"),
    ]
    legend1 = ax.legend(handles=region_patches, loc="lower left", fontsize=7.5,
                        ncol=2, framealpha=0.9, title="Region", title_fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=node_handles, loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="figures")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("Loading processed dataset...")
    df = pd.read_excel(OUTPUT_PATH)
    df = df[df["Region"].notna() & (df["Region"] != "Antarctica")].reset_index(drop=True)

    print("Loading Exp 1 results...")
    gnn = json.load(open("result/gnn_inductive.json"))
    bl  = json.load(open("result/baseline.json"))

    metrics     = compute_metrics(gnn, bl)
    test_coords = get_test_coords(df, seed=42)

    print("\nGenerating world overview (Exp 1)...")
    make_world_overview(df, test_coords, metrics,
                        f"{args.save_dir}/fig_world_overview_exp1.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
