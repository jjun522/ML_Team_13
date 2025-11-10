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
# Config (paths/options/thresholds)
# =============================

# Numeric filter ranges (adjust if needed)
ABV_MIN, ABV_MAX = 0.0, 20.0
IBU_MIN, IBU_MAX = 0.0, 150.0
OG_MIN, OG_MAX = 1.000, 1.150
FG_MIN, FG_MAX = 0.980, 1.060
COLOR_MIN, COLOR_MAX = 0.0, 60.0

# Text normalization
MIN_NAME_LEN = 3

# Optional fuzzy matching (disabled by default to keep runtime low)
USE_FUZZY = False
FUZZY_CUTOFF = 0.92  # 0~1, higher = stricter
MAX_FUZZY_CAND = 1  # number of candidates (1 is recommended)
# Limit candidate pool: favor same style and same first letter when possible
LIMIT_TO_SAME_FIRST_LETTER = True

# Fuzzy matching guards for large datasets
MAX_REVIEWS_FOR_FUZZY = 50_000  # cap on unique beers from reviews
MAX_RECIPES_FOR_POOL = 50_000  # cap on unique beers from recipes


# =============================
# Load data
# =============================
t0 = time.time()
reviews = read_csv_with_fallback(RAW_REVIEWS_CSV)
recipes = read_csv_with_fallback(RAW_RECIPES_CSV)

orig_counts = {"reviews_rows": len(reviews), "recipes_rows": len(recipes)}

# =============================
# Column standardization (flexible mapping)
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
    "Color": ["Color", "SRM"],  # recognize SRM as color
    "BrewMethod": ["BrewMethod", "Method"],
}
rec_map = map_first_existing(recipes, rec_colmap_candidates)
recipes = recipes.rename(columns={v: k for k, v in rec_map.items()})[
    list(rec_map.keys())
]

# =============================
# Basic cleaning
# =============================
# Normalize names and drop very short ones
reviews["beer_name_norm"] = reviews["beer_name"].apply(normalize_name)
recipes["beer_name_norm"] = recipes["Name"].apply(normalize_name)

rev_before = len(reviews)
reviews = reviews[reviews["beer_name_norm"].str.len() >= MIN_NAME_LEN].copy()
rev_dropped_names = rev_before - len(reviews)

rec_before = len(recipes)
recipes = recipes[recipes["beer_name_norm"].str.len() >= MIN_NAME_LEN].copy()
rec_dropped_names = rec_before - len(recipes)

# Numeric filters (only if present)
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

# Drop duplicates
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
# 1) Exact matches
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
# 2) Optional fuzzy matches
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
    # Build candidate pool: per-style index + full pool
    if "Style" in recipes.columns:
        rec_by_style = {}
        for style, sub in recipes.groupby(
            recipes["Style"].fillna("").str.lower().str.strip()
        ):
            rec_by_style[style] = list(sub["beer_name_norm"].unique())
    else:
        rec_by_style = {}

    all_recipe_names = list(recipes["beer_name_norm"].unique())

    # Protect against huge pools by sampling
    if len(all_recipe_names) > MAX_RECIPES_FOR_POOL:
        all_recipe_names = all_recipe_names[:MAX_RECIPES_FOR_POOL]

    # Reviews (normalized names) still unresolved
    unmatched = reviews[["beer_name", "beer_name_norm", "beer_style"]].drop_duplicates()
    unmatched = unmatched[~unmatched["beer_name_norm"].isin(exact["beer_name_norm"])]

    # Optional review-side sampling if needed
    if len(unmatched) > MAX_REVIEWS_FOR_FUZZY:
        unmatched = unmatched.head(MAX_REVIEWS_FOR_FUZZY)

    fuzzy_rows = []
    for _, row in unmatched.iterrows():
        q = row["beer_name_norm"]
        if not q or len(q) < MIN_NAME_LEN:
            continue
        style_key = (row.get("beer_style") or "").lower().strip()
        pool = rec_by_style.get(style_key, all_recipe_names)

        # First-letter filter for speed/precision
        if LIMIT_TO_SAME_FIRST_LETTER and len(q) > 0:
            first = q[0]
            pool = [p for p in pool if len(p) > 0 and p[0] == first]
            if not pool:  # Fall back to the global pool
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
                        "match_score": 0.95,  # simple difflib score placeholder
                    }
                )

    if fuzzy_rows:
        fuzzy = pd.DataFrame(fuzzy_rows)
        linkage = pd.concat([linkage, fuzzy], ignore_index=True)

# Remove duplicate pairs (keep first)
linkage = linkage.drop_duplicates(
    subset=["beer_name_norm", "Name_rec", "match_type"], keep="first"
)

# =============================
# Save outputs and report
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
