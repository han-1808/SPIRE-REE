import pandas as pd

from config import COLUMNS_TO_DROP


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(COLUMNS_TO_DROP, axis=1, errors='ignore')


def normalize_text(x) -> str:
    if not isinstance(x, str):
        return x
    return x.replace(" ", "").lower()
