import pandas as pd
import re
import math


def read_csv_with_fallback(path, **kwargs):
    """Try several encodings/engines until the CSV loads."""
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1", "cp1252"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except Exception:
            continue

    return pd.read_csv(
        path,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip",
        **kwargs,
    )


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(str(x).strip())
    except Exception:
        return None


def map_first_existing(df, candidates_dict):
    """Map each canonical column to the first matching source column."""
    mapping = {}
    for new, cands in candidates_dict.items():
        for c in cands:
            if c in df.columns:
                mapping[new] = c
                break
    return mapping
