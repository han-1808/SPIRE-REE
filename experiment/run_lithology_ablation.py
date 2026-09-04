"""
experiment/run_lithology_ablation.py

Comment (2)-D: lithology-inference ablation.

Two parts, per the solution text:

  1. Tag each record with a confidence tier (Direct / Tier1 / Tier2 /
     Unclassified) right at the inference step -- reuses
     `preprocessing/host_lith.classify_host_lith_confidence()`, computed
     purely from raw inputs + `infer_host_lith`'s own output (does not
     duplicate or alter its branching logic, so `Host_Lith`/`is_group`
     values downstream are unaffected). "Report F1 by tier" needs
     per-node predictions from Exp1, not yet saved (same gap as Comment 7
     and (18)-B) -- deferred; this script only builds and saves the
     per-record tier here.

  2. Compare the full model vs. one without inferred-lithology features
     (the 9 `is_group` indicators, Comment 1) to quantify circularity
     risk -- reuses `run_ablation.py`'s new `exclude_lithology` flag
     (mirrors the `exclude_minerals`/`exclude_coords` ablation pattern
     already built for (1)-A and (2)-B). Unlike (2)-B, no dimension
     compensation is called for here.

Usage:
    python experiment/run_lithology_ablation.py
    python experiment/run_lithology_ablation.py --seeds 42 1 2
    python experiment/run_lithology_ablation.py --regions Oceania "South and Central Asia"

Results: result/lithology_confidence_tiers.json (part 1),
result/ablation_lithology.json (part 2).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json

import pandas as pd

from config import OUTPUT_PATH
from host_lith import classify_host_lith_confidence
from run_ablation import _DEFAULT_SEEDS, run_ablation_sweep

RAW_PATH = "data/Global_REE_occurrence_database.xlsx"
_TIER_RESULTS_PATH = "result/lithology_confidence_tiers.json"
_ABLATION_RESULTS_PATH = "result/ablation_lithology.json"
_MODE = "full"
_CONDITIONS = {
    "with_lithology": {"exclude_lithology": False},
    "no_lithology":   {"exclude_lithology": True},
}


def build_confidence_tiers(processed_df: pd.DataFrame) -> pd.DataFrame:
    """Part 1: per-record confidence tier, joined onto the processed
    dataset via ID_No (Host_Lith/Dep_Type/Sig_Mins nullness only survives
    in the raw file -- `Host_Lith` in the processed file is already filled)."""
    raw = pd.read_excel(RAW_PATH, usecols=["ID_No", "Host_Lith", "Dep_Type", "Sig_Mins"])
    raw = raw.drop_duplicates(subset="ID_No", keep="first")
    raw["confidence_tier"] = raw.apply(classify_host_lith_confidence, axis=1)

    merged = processed_df[["ID_No"]].merge(
        raw[["ID_No", "confidence_tier"]], on="ID_No", how="left"
    )
    if merged["confidence_tier"].isna().any():
        n_bad = int(merged["confidence_tier"].isna().sum())
        raise ValueError(
            f"{n_bad} processed row(s) had no matching raw record via ID_No "
            "-- join is likely broken."
        )
    return merged


def run_lithology_ablation_comparison(
    processed_df: pd.DataFrame,
    seeds: list[int],
    regions: list[str] | None = None,
) -> dict:
    results: dict = {}
    for cond_name, cond_kwargs in _CONDITIONS.items():
        print(f"\n{'#'*60}\n  Condition: {cond_name}  ({cond_kwargs})\n{'#'*60}")
        cond_results, cond_summary = run_ablation_sweep(
            processed_df, modes=[_MODE], seeds=seeds, regions=regions, **cond_kwargs,
        )
        results[cond_name] = {
            "runs": cond_results[_MODE],
            "summary": cond_summary[_MODE],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Comment (2)-D: lithology-inference ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--regions", nargs="+", type=str, default=None,
                         help="Restrict to a subset of regions (default: all)")
    parser.add_argument("--skip-ablation", action="store_true",
                         help="Only build confidence tiers (part 1), skip the F1 comparison (part 2)")
    args = parser.parse_args()

    print("Loading processed data...")
    processed_df = pd.read_excel(OUTPUT_PATH)

    print("\n[Part 1] Building host-lithology confidence tiers...")
    tiers = build_confidence_tiers(processed_df)
    print(tiers["confidence_tier"].value_counts())

    Path("result").mkdir(exist_ok=True)
    tiers.to_json(_TIER_RESULTS_PATH, orient="records", indent=2)
    print(f"  Saved → {_TIER_RESULTS_PATH} (per-record tier, for the F1-by-tier "
          f"split once Exp1 per-node predictions are available, jointly with Comment 7-A)")

    if args.skip_ablation:
        print("\n--skip-ablation set, done.")
        return

    print("\n[Part 2] with_lithology vs no_lithology F1 comparison...")
    results = run_lithology_ablation_comparison(processed_df, args.seeds, args.regions)

    payload = {
        "experiment": "ablation_lithology",
        "description": (
            "Comment (2)-D: with_lithology (current, 9 is_group indicators "
            "available to SHAP selection) vs no_lithology (excluded from "
            "the SHAP candidate pool, no dimension compensation) on Exp 1 "
            "(leave-one-region-out inductive), mode=full, SPIRE."
        ),
        "model": "spire",
        "conditions": _CONDITIONS,
        "seeds": args.seeds,
        "regions": args.regions,
        "results": results,
    }
    with open(_ABLATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {_ABLATION_RESULTS_PATH}")

    print("\n── Comment (2)-D summary (avg ± std across seeds) ──────────")
    print(f"{'Condition':<16} {'F1 (avg±std)':>18}")
    for cond in _CONDITIONS:
        avg = results[cond]["summary"]["avg_metrics"].get("f1")
        std = results[cond]["summary"]["std_metrics"].get("f1")
        print(f"{cond:<16} {avg:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
