import pandas as pd
import re, math, os
from pathlib import Path

import kagglehub
from kagglehub.datasets import KaggleDatasetAdapter


def read_csv_with_fallback(path, **kwargs):
    """여러 인코딩/파서를 시도하며 CSV 읽기"""
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1", "cp1252"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except Exception:
            continue

    return pd.read_csv(
        path, encoding="utf-8", engine="python", on_bad_lines="skip", **kwargs
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
    """가능한 컬럼명 후보들 중 데이터프레임에 존재하는 첫 컬럼으로 매핑"""
    mapping = {}
    for new, cands in candidates_dict.items():
        for c in cands:
            if c in df.columns:
                mapping[new] = c
                break
    return mapping
