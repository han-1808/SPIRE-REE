import re

import pandas as pd

from config import HOST_LITH_GROUPS


def normalize_dep(dep) -> str:
    if pd.isna(dep):
        return ""
    dep = dep.lower()
    dep = dep.replace("?", "")
    dep = dep.replace("(", "").replace(")", "")
    return dep


def extract_primary_dep(dep) -> list:
    dep = normalize_dep(dep)
    for term in ["with residual enrichment", "supergene", "tailings"]:
        dep = dep.replace(term, "")
    parts = re.split("[;,]", dep)
    return [p.strip() for p in parts if p.strip()]


def infer_host_lith(row) -> str:
    if not pd.isna(row["Host_Lith"]):
        return row["Host_Lith"]

    deps = extract_primary_dep(row["Dep_Type"])
    sig = row["Sig_Mins"].lower() if isinstance(row["Sig_Mins"], str) else ""

    # ========= Dep_Type lost after cleaning → treat as unclassified =========
    # NOTE: REE_Mins is deliberately not used here (Comment 11 fix) — it is the
    # exact field that defines the target `has_ree`, so using it to infer
    # Host_Lith (which feeds is_group / node features / graph edges) leaks the
    # target into the input. Records only inferable via REE_Mins fall through
    # to the sig-based or final "unclassified" fallback below.
    if len(deps) == 0:
        if sig:
            if "apatite" in sig and "magnetite" in sig:
                return "igneous rock (inferred, low confidence)"
            if "fluorite" in sig:
                return "igneous or hydrothermal rock (inferred, low confidence)"
            if any(k in sig for k in ["pyrite", "chalcopyrite"]):
                return "igneous-related hydrothermal rock (inferred, low confidence)"
            if any(k in sig for k in ["calcite", "dolomite"]):
                return "carbonate rock (inferred, low confidence)"
            if "garnet" in sig:
                return "metamorphic rock (inferred, low confidence)"
            if any(k in sig for k in ["quartz", "feldspar", "zircon"]):
                return "felsic igneous or sedimentary rock (inferred, low confidence)"

    # ========= 1. CARBONATITE =========
    for d in deps:
        if "carbonatite" in d:
            return "carbonatite"

    # ========= 2. ALKALINE IGNEOUS =========
    for d in deps:
        if "alkaline igneous" in d:
            return "syenite / nepheline syenite"

    # ========= 3. IRON OXIDE–APATITE =========
    for d in deps:
        if "iron oxide-apatite" in d:
            return "igneous rock (felsic to intermediate)"

    # ========= 4. OTHER IGNEOUS =========
    for d in deps:
        if "other igneous" in d:
            if "apatite" in sig or "magnetite" in sig:
                return "igneous rock (inferred, low confidence)"
            return "igneous rock"

    # ========= 5. METAMORPHIC =========
    for d in deps:
        if "metamorphic" in d:
            return "gneiss / schist"

    # ========= 6. PHOSPHORITE =========
    for d in deps:
        if "phosphorite" in d:
            return "phosphorite"

    # ========= 6b. BLACK SHALE =========
    for d in deps:
        if "black shale" in d:
            return "black shale"

    # ========= 7. SEDIMENTARY =========
    for d in deps:
        if any(k in d for k in ["sedimentary", "coal"]):
            return "sedimentary rock"

    # ========= 8. BAUXITE =========
    for d in deps:
        if "bauxite" in d:
            return "laterite"

    # ========= 9. PLACER / PALEOPLACER =========
    for d in deps:
        if "placer" in d:
            return "sand, gravel"

    # ========= 10. FLUORITE DEPOSIT =========
    for d in deps:
        if "fluorite deposit" in d:
            if "apatite" in sig or "magnetite" in sig:
                return "igneous-related hydrothermal rock (inferred, low confidence)"
            return "unclassified"

    # ========= 11. URANIUM DEPOSIT =========
    for d in deps:
        if "uranium deposit" in d:
            return "sandstone / igneous (unspecified)"

    # ========= 12. HYDROTHERMAL Fe-OXIDE =========
    for d in deps:
        if "hydrothermal fe-oxide" in d or "iron oxide" in d:
            return "igneous-related hydrothermal rock"

    # ========= 13. FINAL FALLBACK: unclassified =========
    if "unclassified" in deps:
        if sig:
            if "apatite" in sig and "magnetite" in sig:
                return "igneous rock (inferred, low confidence)"
            if "fluorite" in sig:
                return "igneous or hydrothermal rock (inferred, low confidence)"
            if any(k in sig for k in ["pyrite", "chalcopyrite"]):
                return "igneous-related hydrothermal rock (inferred, low confidence)"
            if any(k in sig for k in ["calcite", "dolomite"]):
                return "carbonate rock (inferred, low confidence)"
            if "garnet" in sig:
                return "metamorphic rock (inferred, low confidence)"
            if any(k in sig for k in ["quartz", "feldspar", "zircon"]):
                return "felsic igneous or sedimentary rock (inferred, low confidence)"
        return "unclassified"

    return "unclassified"


def classify_host_lith_confidence(row) -> str:
    """
    Comment (2)-D: confidence tier for `infer_host_lith(row)`'s result,
    determined purely from the raw inputs + the function's own output --
    does not duplicate or alter `infer_host_lith`'s branching logic, so
    there is zero risk of this changing any `Host_Lith`/`is_group` value
    downstream (a real concern raised earlier, since `is_group` also
    drives LODTO's splits and the graph's cosine-similarity term).

    Tiers
    -----
    Direct       : `Host_Lith` was reported directly in the raw record,
                   no inference needed at all.
    Tier1        : inferred from `Dep_Type` deposit-type keywords (the
                   function's main keyword-matched branches).
    Tier2        : inferred from `Sig_Mins` alone, a weaker signal --
                   `infer_host_lith` marks every such case with the
                   literal substring "(inferred, low confidence)" itself,
                   regardless of which branch produced it (the `deps==0`
                   block, the "OTHER IGNEOUS"/"FLUORITE DEPOSIT"
                   sub-branches, or the final fallback), so checking for
                   that marker captures all of them uniformly and matches
                   the function's own self-labeling.
    Unclassified : no `Dep_Type` or `Sig_Mins` signal at all.
    """
    if not pd.isna(row["Host_Lith"]):
        return "Direct"
    inferred = infer_host_lith(row)
    if inferred == "unclassified":
        return "Unclassified"
    if "(inferred, low confidence)" in inferred:
        return "Tier2"
    return "Tier1"


def fill_host_lith(df: pd.DataFrame) -> pd.DataFrame:
    df["Host_Lith_filled"] = df.apply(infer_host_lith, axis=1)
    df["Host_Lith_inferred"] = df["Host_Lith"].isna() & df["Host_Lith_filled"].notna()
    print(f"Host_Lith filled: {df['Host_Lith_inferred'].sum()} records")
    df["Host_Lith"] = df["Host_Lith_filled"]
    df = df.drop(["Host_Lith_filled", "Host_Lith_inferred"], axis=1)
    return df


def compute_host_lith_stats(df: pd.DataFrame) -> pd.DataFrame:
    s = df["Host_Lith"].dropna().str.split(",")
    s_exploded = s.explode().str.strip()
    stats = s_exploded.value_counts().reset_index()
    stats.columns = ["Host_Lith", "Count"]
    return stats


def _normalize(text: str) -> str:
    return text.lower().strip()


def _text_matches_group(text: str, keywords: list) -> bool:
    text = _normalize(text)
    for kw in keywords:
        if re.search(rf"\b{re.escape(_normalize(kw))}\b", text):
            return True
    return False


def create_lith_group_features(df: pd.DataFrame) -> pd.DataFrame:
    for group in HOST_LITH_GROUPS:
        df[f"is_{group}"] = 0

    for idx, row in df.iterrows():
        if pd.isna(row["Host_Lith"]):
            continue
        lith_list = [x.strip() for x in row["Host_Lith"].split(",")]
        for lith in lith_list:
            for group, keywords in HOST_LITH_GROUPS.items():
                if _text_matches_group(lith, keywords):
                    df.at[idx, f"is_{group}"] = 1

    is_cols = [c for c in df.columns if c.startswith("is_")]
    #df["num_groups"] = df[is_cols].sum(axis=1)
    return df
