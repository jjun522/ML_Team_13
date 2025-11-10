"""
Beer Recommendation System Model (Final Version)
(CF, CBF, Hybrid)
"""

import json
from datetime import datetime

import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from utils.paths import (
    OUT_REVIEWS_CLEAN,
    OUT_RECIPES_CLEAN,
    TRAIN_REVIEWS_CSV,
    TEST_REVIEWS_CSV,
    RECOMMENDATIONS_JSON,
    ensure_dirs,
)

# ---
# 0. Load data
# ---
ensure_dirs()

try:
    reviews_df = pd.read_csv(OUT_REVIEWS_CLEAN, low_memory=False)
    recipes_df = pd.read_csv(OUT_RECIPES_CLEAN, low_memory=False)
except FileNotFoundError:
    print("Error: CSV file not found.")
    exit()

if 'review_time' not in reviews_df.columns:
    print("Error: 'review_time' column is required for the split.")
    exit()

print("\n--- 0. Chronological split ---")
reviews_df['review_time'] = pd.to_datetime(reviews_df['review_time'], unit='s')
reviews_df = reviews_df.sort_values(by='review_time')

split_point = int(len(reviews_df) * 0.8)
train_reviews_df = reviews_df.iloc[:split_point]
test_reviews_df = reviews_df.iloc[split_point:]

print(f"Train set (80%): {len(train_reviews_df)} rows")
print(f"Test set (20%): {len(test_reviews_df)} rows")

train_reviews_df.to_csv(TRAIN_REVIEWS_CSV, index=False)
test_reviews_df.to_csv(TEST_REVIEWS_CSV, index=False)
print(f"Saved split to {TRAIN_REVIEWS_CSV} and {TEST_REVIEWS_CSV}")

if 'review_time' not in reviews_df.columns:
    print("Error: 'review_time' column is required for the split.")
    exit()

print("\n--- 0. Chronological split ---")
reviews_df['review_time'] = pd.to_datetime(reviews_df['review_time'], unit='s')
reviews_df = reviews_df.sort_values(by='review_time')

split_point = int(len(reviews_df) * 0.8)
train_reviews_df = reviews_df.iloc[:split_point]
test_reviews_df = reviews_df.iloc[split_point:]

print(f"Train set (80%): {len(train_reviews_df)} rows")
print(f"Test set (20%): {len(test_reviews_df)} rows")

train_reviews_df.to_csv(TRAIN_REVIEWS_CSV, index=False)
test_reviews_df.to_csv(TEST_REVIEWS_CSV, index=False)
print(f"Saved split to {TRAIN_REVIEWS_CSV} and {TEST_REVIEWS_CSV}")

# ---
# 1. Model-based collaborative filtering (CF)
# [scikit-surprise] SVD reference  https://westlife0615.tistory.com/858
# ---
print("\n--- 1. Train collaborative filtering (CF) model ---")
reader = Reader(rating_scale=(1, 5))
full_train_data = Dataset.load_from_df(train_reviews_df[['review_profilename', 'beer_name_norm', 'review_overall']], reader)
full_trainset = full_train_data.build_full_trainset()
algo_svd = SVD(n_factors=50, n_epochs=20, random_state=42)
algo_svd.fit(full_trainset)

# ---
# 2. Content-based filtering (CBF)
# ---
print("\n--- 2. Prepare content-based (CBF) features ---")

SAMPLING_SIZE = 20000
if len(recipes_df) > SAMPLING_SIZE:
    print(f"Original recipes: {len(recipes_df)} rows")
    recipes_df = recipes_df.sample(n=SAMPLING_SIZE, random_state=42)

recipes_features = recipes_df.drop_duplicates(subset=['beer_name_norm']).set_index('beer_name_norm')

# Feature definitions
numerical_features = ['ABV', 'IBU', 'OG', 'FG', 'Color']

for col in numerical_features:
    if col in recipes_features.columns:
        recipes_features[col] = recipes_features[col].fillna(recipes_features[col].mean())
    else:
        print(f"Warning: column '{col}' not found in recipes_df. Dropping it.")
        numerical_features.remove(col)

# Scale numeric features
scaler = MinMaxScaler()
features_scaled_df = pd.DataFrame(
    scaler.fit_transform(recipes_features[numerical_features]),
    columns=numerical_features,
    index=recipes_features.index
)

# One-hot encode categorical features
if 'Style' in recipes_features.columns:
    style_dummies = pd.get_dummies(recipes_features['Style'], prefix='Style')
    final_features_df = pd.concat([features_scaled_df, style_dummies], axis=1)
    print("CBF feature matrix ready.")
else:
    print("Warning: 'Style' column is missing. Using numeric features only.")
    final_features_df = features_scaled_df

print("Starting cosine similarity computation...")
item_similarity_df = pd.DataFrame(
    cosine_similarity(final_features_df),
    index=final_features_df.index,
    columns=final_features_df.index
)
print("Cosine similarity ready.")


def get_content_based_recommendations(beer_norm_name, top_n=5):
    if beer_norm_name not in item_similarity_df:
        return pd.Series(dtype='float64')

    similar_scores = item_similarity_df[beer_norm_name]

    # Drop near-perfect matches (self similarity)
    similar_scores = similar_scores[similar_scores < 0.99]

    similar_scores = similar_scores.drop(beer_norm_name, errors='ignore').sort_values(ascending=False)

    return similar_scores.head(top_n)


print("CBF model prep complete.")

# ---
# 3. Hybrid filtering (CF + CBF)
# ---
print("\n--- 3. Build recommendation lists (CF, CBF, Hybrid) ---")

# Weights (tune as needed)
W_CF = 1.0  # CF weight
W_CBF = 0.5  # CBF weight


def get_all_recommendations(user_id, top_n=10):
    """
    Return three recommendation lists for a user:
      1. CF list
      2. CBF list
      3. Hybrid list (CF * CBF)
    """

    user_reviews = train_reviews_df[train_reviews_df['review_profilename'] == user_id]
    user_rated_beers = set(user_reviews['beer_name_norm'])
    top_user_beers = user_reviews[user_reviews['review_overall'] >= 4.0]['beer_name_norm'].unique()

    if len(top_user_beers) == 0:
        print(f"User '{user_id}' has no beers rated ≥ 4.0 (cold start).")
        return None, None, None

    candidate_beers = {}
    for beer_norm in top_user_beers:
        similar_beers = get_content_based_recommendations(beer_norm, top_n=10)
        for beer_name, score in similar_beers.items():
            candidate_beers[beer_name] = max(candidate_beers.get(beer_name, 0), score)

    all_scores = []
    for beer_norm, cbf_max_score in candidate_beers.items():
        if beer_norm in user_rated_beers:
            continue

        cf_score = algo_svd.predict(uid=user_id, iid=beer_norm).est

        # Weighted hybrid score
        hybrid_score = (cf_score ** W_CF) * (cbf_max_score ** W_CBF)

        all_scores.append((beer_norm, cf_score, cbf_max_score, hybrid_score))

    cf_recs = sorted(all_scores, key=lambda x: x[1], reverse=True)[:top_n]
    cbf_recs = sorted(all_scores, key=lambda x: x[2], reverse=True)[:top_n]
    hybrid_recs = sorted(all_scores, key=lambda x: x[3], reverse=True)[:top_n]

    return cf_recs, cbf_recs, hybrid_recs


# ---
# 4. Example recommendation output
# ---
print("\n--- Top 5 reviewers (for inspection) ---")
top_reviewers = train_reviews_df['review_profilename'].value_counts().head(5)
print(top_reviewers)
print("------------------------------")

TEST_USER = 'stakem'
cf_recs, cbf_recs, hybrid_recs = get_all_recommendations(TEST_USER, top_n=10)

print(f"\n--- [Sample recommendations for {TEST_USER}] ---")

if cf_recs:
    # CF list
    print("\n[Feed 1: collaborative filtering]")
    for beer, cf, cbf, hy in cf_recs:
        print(f"  - {beer} (CF score: {cf:.2f})")

    # CBF list
    print("\n[Feed 2: content-based filtering]")
    for beer, cf, cbf, hy in cbf_recs:
        print(f"  - {beer} (CBF similarity: {cbf:.2f})")

    # Hybrid list
    print("\n[Feed 3: hybrid (CF × CBF)]")
    for beer, cf, cbf, hy in hybrid_recs:
        print(f"  - {beer} (Hybrid: {hy:.2f} | CF {cf:.2f}, CBF {cbf:.2f})")
else:
    print(f"Could not generate recommendations for '{TEST_USER}'.")

def _format_recommendations(recs):
    if not recs:
        return []
    formatted = []
    for idx, (beer, cf, cbf, hy) in enumerate(recs, start=1):
        formatted.append(
            {
                "rank": idx,
                "beer_name": beer,
                "cf_score": round(float(cf), 4),
                "cbf_score": round(float(cbf), 4),
                "hybrid_score": round(float(hy), 4),
            }
        )
    return formatted


recommendation_payload = {
    "user": TEST_USER,
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "cf": _format_recommendations(cf_recs),
    "cbf": _format_recommendations(cbf_recs),
    "hybrid": _format_recommendations(hybrid_recs),
}

RECOMMENDATIONS_JSON.write_text(
    json.dumps(recommendation_payload, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nSaved recommendation JSON: {RECOMMENDATIONS_JSON}")
