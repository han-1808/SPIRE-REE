DATA_PATH = "data/Global_REE_occurrence_database.xlsx"
OUTPUT_PATH = "data/Global_REE_occurrence_database_processed.xlsx"
DASH_RECORDS_PATH = "data/has_dash_records.xlsx"
HOST_LITH_STATS_PATH = "data/host_lith_stats.xlsx"

TOP_N_MINERALS = 180

COLUMNS_TO_DROP = [
    'OBJECTID', 'Name_Other', 'Company', 'Comments', 'Components', 'Stat_Note',
    'Oth_Mins', 'Ref_List', 'Discov_Yr', 'Expl_Note', 'Mine_Meth', 'P_Status',
    'PStat_Note', 'P_Years', 'P_refs', 'P_Note', 'RR_Note', 'RR_RegCode', 'RR_Refs',
    'RR_Yr_Est', 'RR_Cutoff', 'RR_oth_grd', 'RR_Ore_Mt', 'RR_TREO_Mt', 'RR_HM_Mt',
    'RR_HM_pct', 'RR_min_Mt', 'RR_min_pct', 'RR_mon_Mt', 'RR_mon_pct', 'RR_TREOgrd',
    'RR_REE_grd', 'Dep_Note', 'Dep_Form', 'Part_of', 'Rec_Note', 'REE_Ratio',
    'Loc_Note', 'Rec_Type', 'LREE_Note', 'HREE_Note', 'REE',
]

REE_ELEMENTS = {
    "ce": "cerium",
    "dy": "dysprosium",
    "er": "erbium",
    "eu": "europium",
    "gd": "gadolinium",
    "ho": "holmium",
    "la": "lanthanum",
    "lu": "lutetium",
    "nd": "neodymium",
    "pr": "praseodymium",
    "pm": "promethium",
    "sm": "samarium",
    "sc": "scandium",
    "tb": "terbium",
    "tm": "thulium",
    "yb": "ytterbium",
    "y":  "yttrium",
}

# Known REE-bearing mineral names (used for exploratory analysis)
REE_MINERALS_REF = [
    "Aeschynite", "Allanite (orthite)", "Anatase", "Ancylite-(Ce)", "Apatite",
    "Bastnäsite-(Ce)", "Brannerite", "Britholite-(Ce)", "Brockite",
    "Calcio-ancylite-(Ce)", "Cerianite-(Ce)", "Cerite-(Ce)", "Cheralite-(Ce)",
    "Chevkinite", "Churchite-(Y)", "Crandallite", "Doverite", "Synchysite-(Y)",
    "Eudialyte", "Euxenite-(Y)", "Fergusonite-(Ce)", "Fergusonite-(Y)",
    "Florencite-(Ce)", "Fluocerite-(Ce)", "Fluocerite-(La)", "Fluorapatite-(Ce)",
    "Fluorite", "Gadolinite", "Gagarinite-(Y)", "Gerenite-(Y)", "Gorceixite",
    "Goyazite", "Hingganite-(Y)", "Huanghoite-(Ce)", "Hydroxylbastnäsite-(Ce)",
    "Limonite-(Y)", "Kainosite-(Y)", "Loparite-(Ce)", "Monazite-(Ce)", "Mosandrite",
    "Parisite-(Ce)", "Perovskite", "Pyrochlore", "Rhabdophane-(Ce)",
    "Rhabdophane-(La)", "Rinkite", "Samarskite-(Y)", "Steenstrupine-(Ce)",
    "Synchysite-(Ce)", "Thalénite-(Y)", "Titanite (sphene)", "Uraninite",
    "Vitusite-(Ce)", "Xenotime-(Y)", "Yttrofluorite", "Yttrotantalite-(Y)", "Zircon",
]

# ── GNN artifact paths ──────────────────────────────────────────────────────
GNN_FEATURES_PATH   = "data/gnn_features.parquet"
GNN_TARGET_PATH     = "data/gnn_target.npy"
GNN_EDGE_INDEX_PATH = "data/gnn_edge_index.npy"
GNN_EDGE_WEIGHT_PATH = "data/gnn_edge_weight.npy"
BASELINE_METRICS_PATH       = "result/baseline.json"
GNN_INDUCTIVE_METRICS_PATH  = "result/gnn_inductive.json"
SAMPLE_INDUCTIVE_METRICS_PATH = "result/inductive_sample.json"

# ── Graph construction ───────────────────────────────────────────────────────
KNN_K = 15
TOP_K_SHAP_FEATURES = 10
DEFAULT_GNN_REGION_SWEEP_CONFIG = (32, 2, 0.5, 0.001)

# ── Commodity element regex patterns (REE-safe flags from Commods column) ───
COMMODITY_ELEMENTS = {
    "has_Nb": r"\bNb\b",
    "has_Ta": r"\bTa\b",
    "has_Th": r"\bTh\b",
    "has_U":  r"\bU\b",
    "has_P":  r"\bP\b",
    "has_F":  r"\bF\b",
    "has_Zr": r"\bZr\b",
}

# ── Ordered rules for REE-sanitised deposit classification ───────────────────
# (label, keywords) — first match wins; mirrors classify_system() in REE_detector.ipynb
SYSTEM_CLASS_RULES = [
    ("carbonatite",        ["carbonatite"]),
    ("alkaline_intrusive", ["alkaline", "alkali"]),
    ("placer",             ["placer"]),
    ("clay_laterite",      ["clay", "laterite", "ion adsorption", "ion-adsorption"]),
    ("pegmatite",          ["pegmatite"]),
    ("vein",               ["vein"]),
    ("skarn",              ["skarn", "contact"]),
    ("hydrothermal",       ["hydrothermal"]),
]
# Fallback: "other" if Dep_Type is meaningful, "unknown" otherwise

# ── REE terms to scrub before system classification (prevents leakage) ───────
SYSTEM_CLASS_REE_PATTERN = (
    r"(REE|rare earth|lanthanide|monazite|bastn[aä]site|xenotime|loparite|eudialyte)"
)

# ── Fixed category lists — guarantee consistent dummy column counts ───────────
REGION_CATEGORIES = [
    "Africa", "China", "East Asia", "Europe", "Middle East",
    "North America", "Oceania", "Russian Federation",
    "South America", "South and Central Asia",
]
REC_TYPE_CATEGORIES = [
    "district or area", "district or area(?)",
    "intrusion or complex", "intrusion or complex(?)",
    "site", "site(?)",
]

HOST_LITH_GROUPS = {
    "carbonatite_system": [
        "carbonatite", "magnetite carbonatite", "dolomitic carbonatite",
        "calcite carbonatite", "calcite carbonatite lava",
        "dolomite carbonatite", "sövite", "beforsite",
        "rauhaugite", "weathered carbonatite",
    ],
    "alkaline_intrusive": [
        "syenite", "nepheline syenite", "syenite / nepheline syenite",
        "alkaline syenite", "quartz syenite", "peralkaline syenite",
        "alkaline igneous rock", "igneous rock (inferred alkaline)",
        "foyaite", "urtite", "ijolite", "melteigite", "phonolite",
        "alkaline granite", "peralkaline granite",
    ],
    "mafic_ultramafic": [
        "pyroxenite", "gabbro", "diorite", "monzonite",
        "intermediate intrusions", "phoscorite",
    ],
    "felsic_pegmatite": [
        "granite", "biotite granite", "monzogranite",
        "gneissic granite", "pegmatite", "albitite",
    ],
    "metamorphic_metasomatic": [
        "gneiss", "gneiss / schist", "schist",
        "mica schist", "amphibolite", "migmatite",
        "quartzite", "fenite",
    ],
    "sedimentary": [
        "limestone", "dolomite", "shale", "black shale",
        "sandstone", "siltstone", "mudstone", "conglomerate",
        "sedimentary rock", "sandstone / igneous (unspecified)",
        "phosphorite",
    ],
    "placer": [
        "sand", "gravel", "beach sand", "dune sand",
        "beach and dune sand", "dune and beach sand",
        "sand gravel", "alluvium", "alluvial sediment",
    ],
    "supergene_lateritic": [
        "laterite", "bauxite", "clay",
    ],
    "unclassified": ["unclassified"],
}
