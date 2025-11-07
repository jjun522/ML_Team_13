#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beer review & recipe filtering + linkage (exact + optional fuzzy)

Input:
  - /data/beer_reviews.csv
  - /data/recipeData.csv

Output:
  - /data/beer_reviews_clean.csv
  - /data/recipes_clean.csv
  - /data/beer_name_linkage.csv
  - /data/filtering_report.txt
"""

import os, re, math, json, time
import pandas as pd
from difflib import get_close_matches

# =============================
# Config (경로/옵션/임계값)
# =============================
REVIEWS_PATH = "C:/Users/ATIV/Desktop/vscode/CV/beer_reviews.csv"
RECIPES_PATH = "C:/Users/ATIV/Desktop/vscode/CV/recipeData.csv"

OUT_REVIEWS_CLEAN = "C:/Users/ATIV/Desktop/vscode/CV/data/beer_reviews_clean.csv"
OUT_RECIPES_CLEAN = "C:/Users/ATIV/Desktop/vscode/CV/data/recipes_clean.csv"
OUT_LINKAGE       = "C:/Users/ATIV/Desktop/vscode/CV/data/beer_name_linkage.csv"
OUT_REPORT        = "C:/Users/ATIV/Desktop/vscode/CV/data/filtering_report.txt"

# 수치 필터 범위(필요 시 조정)
ABV_MIN, ABV_MAX = 0.0, 20.0
IBU_MIN, IBU_MAX = 0.0, 150.0
OG_MIN,  OG_MAX  = 1.000, 1.150
FG_MIN,  FG_MAX  = 0.980, 1.060

# 문자열 정규화 관련
MIN_NAME_LEN = 3

# 퍼지 매칭 옵션(기본 False → 느림 방지)
USE_FUZZY         = False
FUZZY_CUTOFF      = 0.92   # 0~1, 높을수록 보수적
MAX_FUZZY_CAND    = 1      # 후보 수 (1 권장)
# 후보 풀 제한: 같은 스타일 우선 + 같은 첫 글자 동일 시도
LIMIT_TO_SAME_FIRST_LETTER = True

# 대용량일 때 퍼지 매칭 속도/안정성 튜닝
MAX_REVIEWS_FOR_FUZZY = 50_000   # 리뷰 고유 맥주명 상한 (넘으면 상위 N개만)
MAX_RECIPES_FOR_POOL  = 50_000   # 레시피 고유 맥주명 상한 (넘으면 샘플링)


# =============================
# 유틸
# =============================
def read_csv_with_fallback(path, **kwargs):
    """여러 인코딩/파서를 시도하며 CSV 읽기"""
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except Exception as e:
            last_err = e
            continue
    # 최후: 손상 라인 스킵
    return pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip", low_memory=False, **kwargs)

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


# =============================
# 로드
# =============================
t0 = time.time()
reviews = read_csv_with_fallback(REVIEWS_PATH)
recipes = read_csv_with_fallback(RECIPES_PATH)

orig_counts = {"reviews_rows": len(reviews), "recipes_rows": len(recipes)}

# =============================
# 컬럼 표준화(유연 매핑)
# =============================
rev_colmap_candidates = {
    "beer_name": ["beer_name", "Beer Name", "beer"],
    "beer_style": ["beer_style", "Beer Style", "style"],
    "beer_abv": ["beer_abv", "ABV", "beer_abv_percent"],
    "brewery_name": ["brewery_name", "Brewery Name"],
    "review_profilename": ["review_profilename", "user", "reviewer"],
    "review_overall": ["review_overall", "overall"],
    "review_aroma": ["review_aroma", "aroma"],
    "review_appearance": ["review_appearance", "appearance"],
    "review_palate": ["review_palate", "palate"],
    "review_taste": ["review_taste", "taste"],
    "review_time": ["review_time", "time"],
    "review_text": ["review_text", "text", "review"],
}
rev_map = map_first_existing(reviews, rev_colmap_candidates)
reviews = reviews.rename(columns={v: k for k, v in rev_map.items()})[list(rev_map.keys())]

rec_colmap_candidates = {
    "Name": ["Name", "RecipeName", "BeerName"],
    "Style": ["Style"],
    "ABV": ["ABV"],
    "IBU": ["IBU"],
    "OG": ["OG", "OriginalGravity"],
    "FG": ["FG", "FinalGravity"],
    "BrewMethod": ["BrewMethod", "Method"],
}
rec_map = map_first_existing(recipes, rec_colmap_candidates)
recipes = recipes.rename(columns={v: k for k, v in rec_map.items()})[list(rec_map.keys())]

# =============================
# 기본 클리닝
# =============================
# 이름 정규화 + 너무 짧은 이름 제거
reviews["beer_name_norm"] = reviews["beer_name"].apply(normalize_name)
recipes["beer_name_norm"] = recipes["Name"].apply(normalize_name)

rev_before = len(reviews)
reviews = reviews[reviews["beer_name_norm"].str.len() >= MIN_NAME_LEN].copy()
rev_dropped_names = rev_before - len(reviews)

rec_before = len(recipes)
recipes = recipes[recipes["beer_name_norm"].str.len() >= MIN_NAME_LEN].copy()
rec_dropped_names = rec_before - len(recipes)

# 수치 필터링(있을 때만)
if "beer_abv" in reviews.columns:
    reviews["beer_abv"] = reviews["beer_abv"].apply(safe_float)
    reviews = reviews[(reviews["beer_abv"].isna()) | (reviews["beer_abv"].between(ABV_MIN, ABV_MAX))]

for col, (lo, hi) in {"ABV": (ABV_MIN, ABV_MAX), "IBU": (IBU_MIN, IBU_MAX),
                      "OG": (OG_MIN, OG_MAX), "FG": (FG_MIN, FG_MAX)}.items():
    if col in recipes.columns:
        recipes[col] = recipes[col].apply(safe_float)
        recipes = recipes[(recipes[col].isna()) | (recipes[col].between(lo, hi))]

# 중복 제거
key_cols = [c for c in ["review_profilename", "beer_name_norm", "review_time"] if c in reviews.columns]
if key_cols:
    reviews = reviews.sort_values(by=key_cols).drop_duplicates(
        subset=[c for c in key_cols if c != "review_time"],
        keep="last"
    )

dedup_subset = ["beer_name_norm"] + (["Style"] if "Style" in recipes.columns else [])
recipes = recipes.drop_duplicates(subset=dedup_subset, keep="first")

# =============================
# 1) 정확 매칭(exact)
# =============================
exact = pd.merge(
    reviews[["beer_name", "beer_name_norm", "beer_style"]].drop_duplicates(),
    recipes[["Name", "beer_name_norm", "Style"]].drop_duplicates(),
    on="beer_name_norm",
    how="inner",
    suffixes=("_rev", "_rec")
)
exact["match_type"] = "exact"
exact["match_score"] = 1.0

# =============================
# 2) 퍼지 매칭(fuzzy) — 옵션
# =============================
linkage = exact.rename(columns={
    "beer_name": "beer_name_rev",
    "Name": "Name_rec"
})[["beer_name_rev","beer_name_norm","beer_style","Name_rec","Style","match_type","match_score"]]

if USE_FUZZY:
    # 후보 풀 만들기: 스타일별 인덱스 + 전체 풀
    if "Style" in recipes.columns:
        rec_by_style = {}
        for style, sub in recipes.groupby(recipes["Style"].fillna("").str.lower().str.strip()):
            rec_by_style[style] = list(sub["beer_name_norm"].unique())
    else:
        rec_by_style = {}

    all_recipe_names = list(recipes["beer_name_norm"].unique())

    # 대용량 보호: 필요시 샘플링
    if len(all_recipe_names) > MAX_RECIPES_FOR_POOL:
        all_recipe_names = all_recipe_names[:MAX_RECIPES_FOR_POOL]

    # 아직 매칭 안 된 리뷰측 맥주명
    unmatched = (reviews[["beer_name", "beer_name_norm", "beer_style"]]
                 .drop_duplicates())
    unmatched = unmatched[~unmatched["beer_name_norm"].isin(exact["beer_name_norm"])]

    # 대용량 보호: 리뷰 쪽도 상위 N개만 시도 (원하면 조건/정렬 로직에 맞게 교체)
    if len(unmatched) > MAX_REVIEWS_FOR_FUZZY:
        unmatched = unmatched.head(MAX_REVIEWS_FOR_FUZZY)

    fuzzy_rows = []
    for _, row in unmatched.iterrows():
        q = row["beer_name_norm"]
        if not q or len(q) < MIN_NAME_LEN:
            continue
        style_key = (row.get("beer_style") or "").lower().strip()
        pool = rec_by_style.get(style_key, all_recipe_names)

        # 첫 글자 제한(속도/정확도 향상)
        if LIMIT_TO_SAME_FIRST_LETTER and len(q) > 0:
            first = q[0]
            pool = [p for p in pool if len(p) > 0 and p[0] == first]
            if not pool:  # 비면 전체 풀로 fallback
                pool = rec_by_style.get(style_key, all_recipe_names)

        candidates = get_close_matches(q, pool, n=MAX_FUZZY_CAND, cutoff=FUZZY_CUTOFF)
        for cand in candidates:
            rec_rows = recipes[recipes["beer_name_norm"] == cand].head(1)
            if not rec_rows.empty:
                fuzzy_rows.append({
                    "beer_name_rev": row["beer_name"],
                    "beer_name_norm": q,
                    "beer_style": row.get("beer_style"),
                    "Name_rec": rec_rows["Name"].iloc[0],
                    "Style": rec_rows["Style"].iloc[0] if "Style" in rec_rows.columns else None,
                    "match_type": "fuzzy",
                    "match_score": 0.95  # difflib의 간단 점수(placeholder)
                })

    if fuzzy_rows:
        fuzzy = pd.DataFrame(fuzzy_rows)
        linkage = pd.concat([linkage, fuzzy], ignore_index=True)

# 중복 제거(동일 조합 1개만)
linkage = linkage.drop_duplicates(subset=["beer_name_norm","Name_rec","match_type"], keep="first")

# =============================
# 저장 + 리포트
# =============================
reviews.to_csv(OUT_REVIEWS_CLEAN, index=False)
recipes.to_csv(OUT_RECIPES_CLEAN, index=False)
linkage.to_csv(OUT_LINKAGE, index=False)

stats = {
    "input": orig_counts,
    "after_name_drop": {
        "reviews_dropped_empty_names": int(rev_dropped_names),
        "recipes_dropped_empty_names": int(rec_dropped_names),
    },
    "cleaned": {
        "reviews_rows": int(len(reviews)),
        "recipes_rows": int(len(recipes)),
    },
    "linkage": {
        "exact_matches": int((linkage["match_type"] == "exact").sum()),
        "fuzzy_matches": int((linkage["match_type"] == "fuzzy").sum()) if USE_FUZZY and not linkage.empty else 0,
        "total_linked_pairs": int(len(linkage)),
    },
    "paths": {
        "beer_reviews_clean": OUT_REVIEWS_CLEAN,
        "recipes_clean": OUT_RECIPES_CLEAN,
        "beer_name_linkage": OUT_LINKAGE,
        "report": OUT_REPORT
    },
    "runtime_sec": round(time.time() - t0, 2)
}

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("# Data Filtering & Linkage Report\n")
    f.write(json.dumps(stats, indent=2, ensure_ascii=False))

print(json.dumps(stats, indent=2, ensure_ascii=False))
print(f"Saved:\n  {OUT_REVIEWS_CLEAN}\n  {OUT_RECIPES_CLEAN}\n  {OUT_LINKAGE}\n  {OUT_REPORT}")

