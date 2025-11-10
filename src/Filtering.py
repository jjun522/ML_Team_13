#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beer review & recipe filtering + linkage (exact + optional fuzzy)

Input:
  - data/beer_reviews.csv
  - data/recipeData.csv

Output:
  - result/beer_reviews_clean.csv
  - result/recipes_clean.csv
  - result/beer_name_linkage.csv
  - result/filtering_report.txt
"""

from utils.helper import (
    read_csv_with_fallback,
    normalize_name,
    safe_float,
    map_first_existing,
)
from utils.paths import (
    RAW_REVIEWS_CSV,
    RAW_RECIPES_CSV,
    OUT_REVIEWS_CLEAN,
    OUT_RECIPES_CLEAN,
    OUT_LINKAGE,
    OUT_REPORT,
    ensure_dirs,
)
import json, time
import pandas as pd
from difflib import get_close_matches

# Ensure standard folders exist before reading/writing data.
ensure_dirs()

# =============================
# Config (경로/옵션/임계값)
# =============================

# 수치 필터 범위(필요 시 조정)
ABV_MIN, ABV_MAX = 0.0, 20.0
IBU_MIN, IBU_MAX = 0.0, 150.0
OG_MIN, OG_MAX = 1.000, 1.150
FG_MIN, FG_MAX = 0.980, 1.060
COLOR_MIN, COLOR_MAX = 0.0, 60.0

# 문자열 정규화 관련
MIN_NAME_LEN = 3

# 퍼지 매칭 옵션(기본 False → 느림 방지)
USE_FUZZY = False
FUZZY_CUTOFF = 0.92  # 0~1, 높을수록 보수적
MAX_FUZZY_CAND = 1  # 후보 수 (1 권장)
# 후보 풀 제한: 같은 스타일 우선 + 같은 첫 글자 동일 시도
LIMIT_TO_SAME_FIRST_LETTER = True

# 대용량일 때 퍼지 매칭 속도/안정성 튜닝
MAX_REVIEWS_FOR_FUZZY = 50_000  # 리뷰 고유 맥주명 상한 (넘으면 상위 N개만)
MAX_RECIPES_FOR_POOL = 50_000  # 레시피 고유 맥주명 상한 (넘으면 샘플링)


# =============================
# 로드
# =============================
t0 = time.time()
reviews = read_csv_with_fallback(RAW_REVIEWS_CSV)
recipes = read_csv_with_fallback(RAW_RECIPES_CSV)

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
reviews = reviews.rename(columns={v: k for k, v in rev_map.items()})[
    list(rev_map.keys())
]

rec_colmap_candidates = {
    "Name": ["Name", "RecipeName", "BeerName"],
    "Style": ["Style"],
    "ABV": ["ABV"],
    "IBU": ["IBU"],
    "OG": ["OG", "OriginalGravity"],
    "FG": ["FG", "FinalGravity"],
    "Color": ["Color", "SRM"],  # Color 컬럼 후보 추가
    "BrewMethod": ["BrewMethod", "Method"],
}
rec_map = map_first_existing(recipes, rec_colmap_candidates)
recipes = recipes.rename(columns={v: k for k, v in rec_map.items()})[
    list(rec_map.keys())
]

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
    reviews = reviews[
        (reviews["beer_abv"].isna()) | (reviews["beer_abv"].between(ABV_MIN, ABV_MAX))
    ]

for col, (lo, hi) in {
    "ABV": (ABV_MIN, ABV_MAX),
    "IBU": (IBU_MIN, IBU_MAX),
    "OG": (OG_MIN, OG_MAX),
    "FG": (FG_MIN, FG_MAX),
    "Color": (COLOR_MIN, COLOR_MAX),
}.items():
    if col in recipes.columns:
        recipes[col] = recipes[col].apply(safe_float)
        recipes = recipes[(recipes[col].isna()) | (recipes[col].between(lo, hi))]

rec_dropped_style_nan = 0
if "Style" in recipes.columns:
    rec_before_style_drop = len(recipes)
    print("Filtering recipes with missing 'Style'...")
    recipes = recipes[
        recipes["Style"].notna() & (recipes["Style"].str.strip() != "")
    ].copy()
    rec_dropped_style_nan = rec_before_style_drop - len(recipes)
else:
    print("Warning: 'Style' column not found in recipes. Skipping style filter.")

# 중복 제거
key_cols = [
    c
    for c in ["review_profilename", "beer_name_norm", "review_time"]
    if c in reviews.columns
]
if key_cols:
    reviews = reviews.sort_values(by=key_cols).drop_duplicates(
        subset=[c for c in key_cols if c != "review_time"], keep="last"
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
    suffixes=("_rev", "_rec"),
)
exact["match_type"] = "exact"
exact["match_score"] = 1.0

# =============================
# 2) 퍼지 매칭(fuzzy) — 옵션
# =============================
linkage = exact.rename(columns={"beer_name": "beer_name_rev", "Name": "Name_rec"})[
    [
        "beer_name_rev",
        "beer_name_norm",
        "beer_style",
        "Name_rec",
        "Style",
        "match_type",
        "match_score",
    ]
]

if USE_FUZZY:
    # 후보 풀 만들기: 스타일별 인덱스 + 전체 풀
    if "Style" in recipes.columns:
        rec_by_style = {}
        for style, sub in recipes.groupby(
            recipes["Style"].fillna("").str.lower().str.strip()
        ):
            rec_by_style[style] = list(sub["beer_name_norm"].unique())
    else:
        rec_by_style = {}

    all_recipe_names = list(recipes["beer_name_norm"].unique())

    # 대용량 보호: 필요시 샘플링
    if len(all_recipe_names) > MAX_RECIPES_FOR_POOL:
        all_recipe_names = all_recipe_names[:MAX_RECIPES_FOR_POOL]

    # 아직 매칭 안 된 리뷰측 맥주명
    unmatched = reviews[["beer_name", "beer_name_norm", "beer_style"]].drop_duplicates()
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
                fuzzy_rows.append(
                    {
                        "beer_name_rev": row["beer_name"],
                        "beer_name_norm": q,
                        "beer_style": row.get("beer_style"),
                        "Name_rec": rec_rows["Name"].iloc[0],
                        "Style": (
                            rec_rows["Style"].iloc[0]
                            if "Style" in rec_rows.columns
                            else None
                        ),
                        "match_type": "fuzzy",
                        "match_score": 0.95,  # difflib의 간단 점수(placeholder)
                    }
                )

    if fuzzy_rows:
        fuzzy = pd.DataFrame(fuzzy_rows)
        linkage = pd.concat([linkage, fuzzy], ignore_index=True)

# 중복 제거(동일 조합 1개만)
linkage = linkage.drop_duplicates(
    subset=["beer_name_norm", "Name_rec", "match_type"], keep="first"
)

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
        "fuzzy_matches": (
            int((linkage["match_type"] == "fuzzy").sum())
            if USE_FUZZY and not linkage.empty
            else 0
        ),
        "total_linked_pairs": int(len(linkage)),
    },
    "paths": {
        "beer_reviews_clean": str(OUT_REVIEWS_CLEAN),
        "recipes_clean": str(OUT_RECIPES_CLEAN),
        "beer_name_linkage": str(OUT_LINKAGE),
        "report": str(OUT_REPORT),
    },
    "runtime_sec": round(time.time() - t0, 2),
}

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("# Data Filtering & Linkage Report\n")
    f.write(json.dumps(stats, indent=2, ensure_ascii=False))

print(json.dumps(stats, indent=2, ensure_ascii=False))
print(
    f"Saved:\n  {OUT_REVIEWS_CLEAN}\n  {OUT_RECIPES_CLEAN}\n  {OUT_LINKAGE}\n  {OUT_REPORT}"
)
