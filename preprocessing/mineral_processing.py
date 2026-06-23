import re

import pandas as pd

from cleaning import normalize_text
from config import DASH_RECORDS_PATH, REE_ELEMENTS, TOP_N_MINERALS


def contains_ree(sig_mins: str, ree_dict: dict = REE_ELEMENTS) -> bool:
    if pd.isna(sig_mins):
        return False
    text = sig_mins.lower()
    if re.search(r"\b(lree|hree|ree)\b", text):
        return True
    for el in ree_dict.keys():
        pattern = rf"""
            (?<![a-z]){el}(?![a-z])
            |
            {el}-(rich|bearing|oxide|enriched)
        """
        if re.search(pattern, text, re.VERBOSE):
            return True
    return False


def list_ree_elements(sig_mins: str, ree_dict: dict = REE_ELEMENTS) -> list:
    if pd.isna(sig_mins):
        return []
    text = sig_mins.lower()
    found = []
    for el in ree_dict.keys():
        pattern = rf"(?<![a-z]){el}(?![a-z])|{el}-(rich|bearing|oxide|enriched)"
        if re.search(pattern, text):
            found.append(el.upper())
    if re.search(r"\b(lree|hree|ree)\b", text):
        found.append("REE_GENERIC")
    return list(set(found))


def remove_ree_from_sig_mins_inplace(sig_mins: str, ree_dict: dict = REE_ELEMENTS):
    if pd.isna(sig_mins):
        return sig_mins
    mins = [m.strip() for m in sig_mins.split(",")]
    filtered = [m for m in mins if not contains_ree(m, ree_dict)]
    return ", ".join(filtered) if filtered else None


def _has_dash(sig_mins: str) -> bool:
    if pd.isna(sig_mins):
        return False
    return "-" in sig_mins


def process_minerals(df: pd.DataFrame, n: int = TOP_N_MINERALS) -> pd.DataFrame:
    # Normalize REE_Mins
    df["REE_Mins"] = df["REE_Mins"].apply(normalize_text)

    # Binary REE presence flag
    df["has_ree"] = df["REE_Mins"].notnull().astype(int)

    # Detect REE elements in Sig_Mins, then remove them
    df["has_REE_in_Sig_Mins"] = df["Sig_Mins"].apply(contains_ree)
    df["REE_elements"] = df["Sig_Mins"].apply(list_ree_elements)
    df["Sig_Mins"] = df["Sig_Mins"].apply(remove_ree_from_sig_mins_inplace)

    # Export hyphenated mineral records for manual review
    df["has_dash"] = df["Sig_Mins"].apply(_has_dash)
    dash_records = df[df["has_dash"]][["ID_No", "Name", "Commods", "REE_Mins", "Sig_Mins", "Host_Lith"]]
    dash_records.to_excel(DASH_RECORDS_PATH, index=False)
    print(f"Exported {len(dash_records)} has-dash records to: {DASH_RECORDS_PATH}")

    # Build top-N significant mineral binary features
    s_exploded = df["Sig_Mins"].dropna().str.split(",").explode().str.strip()
    top_n_mins = s_exploded.value_counts().head(n).index.tolist()
    for m in top_n_mins:
        col_name = f"has_{m.replace('-', '_').replace('(', '').replace(')', '')}"
        df[col_name] = df["Sig_Mins"].fillna("").str.contains(
            rf"\b{re.escape(m)}\b", regex=True
        ).astype(int)

    # Drop intermediate columns
    df = df.drop(["has_REE_in_Sig_Mins", "has_dash", "REE_elements", "Sig_Mins"], axis=1)
    return df
