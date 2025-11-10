import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import json
from datetime import datetime
from scipy.sparse import csr_matrix
# Surprise is used for CF modeling/evaluation
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from utils.paths import (
    TRAIN_REVIEWS_CSV,
    TEST_REVIEWS_CSV,
    OUT_RECIPES_CLEAN,
    EVAL_METRICS_JSON,
    ensure_dirs,
)

ensure_dirs()

# 0. Settings and hyperparameters

# Tuned hyperparameters
K = 10 # number of items in each list (Top-K)
RELEVANCE_THRESHOLD = 4.0 # label items as relevant when rating >= threshold
W_CF = 1.0 # weight for CF score
W_CBF = 0.8 # weight for CBF score
N_FACTORS = 50 # number of latent factors for SVD
MIN_USER_REVIEWS = 10 # drop users with fewer reviews
MIN_BEER_REVIEWS = 10 # drop beers with fewer reviews
MIN_CBF_SIMILARITY = 0.4 # minimum similarity to keep an item in CBF pool
RANDOM_STATE = 42

# Limit CBF calculations to avoid memory spikes
MAX_CBF_ITEMS = 20000

# Load data
print("Loading datasets...")
try:
    train_reviews_df = pd.read_csv(TRAIN_REVIEWS_CSV, low_memory=False)
    test_reviews_df = pd.read_csv(TEST_REVIEWS_CSV, low_memory=False)
    recipes_df = pd.read_csv(OUT_RECIPES_CLEAN, low_memory=False)
except FileNotFoundError as e:
    print("Error: file not found.", e)
    exit()

print(f"Loaded train={len(train_reviews_df)}, test={len(test_reviews_df)}, recipes={len(recipes_df)}.")

# Filter users/beers with too few reviews
print("\nFiltering by minimum review counts...")
user_counts = train_reviews_df['review_profilename'].value_counts()
active_users = user_counts[user_counts >= MIN_USER_REVIEWS].index
train_reviews_df = train_reviews_df[train_reviews_df['review_profilename'].isin(active_users)]
beer_counts = train_reviews_df['beer_name_norm'].value_counts()
popular_beers = beer_counts[beer_counts >= MIN_BEER_REVIEWS].index
train_reviews_df = train_reviews_df[train_reviews_df['beer_name_norm'].isin(popular_beers)]
print(f"Train rows after filtering: {len(train_reviews_df)}")

# 1. Train CF model (Surprise SVD)
print("\n 1) Training CF model (Surprise SVD)")
# Reader tells Surprise the rating scale (1-5)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    # Convert into Surprise format: user, item, rating
    train_reviews_df[['review_profilename', 'beer_name_norm', 'review_overall']],
    reader
)

# Initialize SVD (matrix factorization) model
model_svd = SVD(n_factors=N_FACTORS, random_state=RANDOM_STATE, n_epochs=25, lr_all=0.005, reg_all=0.02)
trainset = data.build_full_trainset()
model_svd.fit(trainset)
print(" CF model training complete.")

# 2. CBF model (similarity matrix)
print("\n2) Building CBF similarity matrix")

# Keep only items that appear in the training set
item_indices = [trainset.to_raw_iid(i) for i in trainset.all_items()]

filtered_beer_names = set(item_indices)
recipes_features_filtered = recipes_df[recipes_df['beer_name_norm'].isin(filtered_beer_names)].copy()
recipes_features = recipes_features_filtered.drop_duplicates(subset=['beer_name_norm']).set_index('beer_name_norm')

# Reindex using the trained item order (missing items become NaN)
recipes_features = recipes_features.reindex(item_indices)
# Drop rows where every feature is NaN
recipes_features = recipes_features.dropna(how='all')

# Limit how many items we keep for CBF
original_cbf_count = len(recipes_features.index)
print(f"CBF candidate items: {original_cbf_count}")

if original_cbf_count > MAX_CBF_ITEMS:
    # Keep only the most popular MAX_CBF_ITEMS recipes
    top_cbf_items = popular_beers.head(MAX_CBF_ITEMS).index
    recipes_features = recipes_features.loc[recipes_features.index.intersection(top_cbf_items)]
    print(f"-> Trimmed to {len(recipes_features.index)} recipes for stability.")

# Continue feature prep
numerical_features = [col for col in ['ABV', 'IBU', 'OG', 'FG', 'Color'] if col in recipes_features.columns]
for col in numerical_features:
    # Replace NaNs with the column mean
    recipes_features[col] = recipes_features[col].fillna(recipes_features[col].mean())

# Scale numeric features with MinMaxScaler to avoid bias during similarity
scaler = MinMaxScaler()
features_scaled_df = pd.DataFrame(
    scaler.fit_transform(recipes_features[numerical_features]),
    columns=numerical_features,
    index=recipes_features.index
)
if 'Style' in recipes_features.columns:
    style_dummies = pd.get_dummies(recipes_features['Style'], prefix='Style')
    # Align indices after one-hot encoding and fill missing rows with 0
    style_dummies = style_dummies.reindex(recipes_features.index, fill_value=0)
    final_features_df = pd.concat([features_scaled_df, style_dummies], axis=1)
else:
    final_features_df = features_scaled_df

# Fill any remaining NaNs with 0 and drop rows containing inf
final_features_df = final_features_df.fillna(0).astype(np.float64)
final_features_df = final_features_df[np.isfinite(final_features_df).all(axis=1)]

# Convert to a sparse matrix for efficiency
features_sparse = csr_matrix(final_features_df.values)

try:
    # Cosine similarity between every pair of item vectors
    item_similarity_df = pd.DataFrame(
        cosine_similarity(features_sparse),
        index=final_features_df.index,
        columns=final_features_df.index
    )
    print(f"CBF similarity matrix built ({item_similarity_df.shape})")
except MemoryError:
    print("Failed to build similarity matrix: out of memory.")
except Exception as e:
    print(f"Similarity calculation failed: {e}")
    item_similarity_df = pd.DataFrame()

# 3. Helper Functions
def calculate_ndcg_at_k(recs, truth, k):
    """Compute NDCG@K."""
    # NDCG rewards relevant items that appear near the top of the list
    dcg = 0.0
    for i, item in enumerate(recs[:k]):
        # Relevance is 1 when the item is in the ground truth set
        relevance = 1.0 if item in truth else 0.0
        # Divide by log discount to penalize lower ranks
        dcg += relevance / np.log2(i + 2)
    n_relevant = len(truth)
    if n_relevant == 0:
        return 0.0
    idcg = 0.0
    # Ideal DCG (perfectly ordered list)
    for i in range(min(k, n_relevant)):
        idcg += 1.0 / np.log2(i + 2)
    if idcg == 0.0:
        return 0.0
    # Normalize DCG by IDCG (0..1)
    return dcg / idcg

def predict_cf(user_id, item_id):
    """Predict a rating with the trained SVD model."""
    prediction = model_svd.predict(str(user_id), str(item_id))
    return prediction.est

def get_content_based_recommendations(beer_norm_name, top_n=1000):
    """Return the Top-N beers closest to the given beer (CBF)."""
    if item_similarity_df.empty or beer_norm_name not in item_similarity_df.columns:
        return pd.Series(dtype='float64')
    if beer_norm_name not in item_similarity_df.index:
        return pd.Series(dtype='float64')

    # Fetch similarity scores against every other beer
    sims = item_similarity_df.loc[beer_norm_name].drop(beer_norm_name, errors='ignore')
    # Drop items that fall below the similarity threshold
    sims = sims[sims >= MIN_CBF_SIMILARITY]
    # Sort by similarity descending
    return sims.sort_values(ascending=False).head(top_n)

def get_hybrid_recommendations_for_user(user_id, k):
    """Generate a Top-K hybrid list for a user."""
    if item_similarity_df.empty: return []

    try:
        # Ensure the user exists within the training set
        if trainset.to_inner_uid(str(user_id)) not in trainset.all_users():
            return []
    except ValueError:
        return []

    user_reviews = train_reviews_df[train_reviews_df['review_profilename'] == user_id]
    user_rated = set(user_reviews['beer_name_norm'])
    # Positive beers (ratings >= threshold) act as seeds
    top_user_beers = user_reviews[user_reviews['review_overall'] >= RELEVANCE_THRESHOLD]['beer_name_norm'].unique()

    # Drop seeds that are missing from the CBF matrix
    top_user_beers = [b for b in top_user_beers if b in item_similarity_df.index]
    if not top_user_beers: return []

    candidate_beers = {}
    # Collect candidate beers via CBF and keep the max similarity score
    for beer_norm in top_user_beers:
        similar_beers = get_content_based_recommendations(beer_norm, top_n=k * 10)
        for beer_name, score in similar_beers.items():
            candidate_beers[beer_name] = max(candidate_beers.get(beer_name, 0), score)

    all_scores = []
    for beer_norm, cbf_score in candidate_beers.items():
        if beer_norm in user_rated: continue  # skip already-rated beers

        # Predict CF rating
        cf_score = predict_cf(user_id, beer_norm)

        # Skip if CF score is low
        if cf_score < 4.0:
            continue

        if cbf_score < MIN_CBF_SIMILARITY: continue

        # Hybrid score = (CF^W_CF) * (CBF^W_CBF)
        hybrid_score = (cf_score ** W_CF) * (cbf_score ** W_CBF)
        all_scores.append((beer_norm, hybrid_score))

    hybrid_recs = sorted(all_scores, key=lambda x: x[1], reverse=True)[:k]
    return [beer_name for beer_name, _ in hybrid_recs]

# 4. Quantitative metrics (RMSE, Precision, Recall, NDCG)
print("\n 4) Evaluating RMSE with the Surprise test split")
test_set_df = test_reviews_df[['review_profilename', 'beer_name_norm', 'review_overall']].copy()

def get_surprise_testset(df, trainset):
    """Keep only (user, item) pairs that exist in the trainset."""
    test_set = []
    valid_uids = set(trainset.to_raw_uid(i) for i in trainset.all_users())
    valid_iids = set(trainset.to_raw_iid(i) for i in trainset.all_items())

    for _, row in df.iterrows():
        user = str(row['review_profilename'])
        item = str(row['beer_name_norm'])
        rating = row['review_overall']

        # Skip pairs that never appeared in training
        if user in valid_uids and item in valid_iids:
            test_set.append((user, item, rating))
    return test_set

testset_surprise = get_surprise_testset(test_set_df, trainset)

rmse = None
if testset_surprise:
    predictions = model_svd.test(testset_surprise)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f" RMSE: {rmse:.4f} (n={len(predictions)})")
else:
    print(" Unable to compute RMSE (no overlapping users/items).")

print(f"\n 5) Precision@{K}, Recall@{K}, NDCG@{K}")
test_users_in_train = set(test_reviews_df['review_profilename']).intersection(set(train_reviews_df['review_profilename']))
test_users_sample = list(test_users_in_train)

# Build ground truth from test-set ratings >= threshold
ground_truth_map = test_reviews_df[test_reviews_df['review_overall'] >= RELEVANCE_THRESHOLD].groupby('review_profilename')['beer_name_norm'].apply(set)

precisions, recalls, ndcgs = [], [], []
for user_id in test_users_sample:
    truth = ground_truth_map.get(user_id, set())
    if not truth:
        continue  # skip users without relevant items

    # Generate Top-K hybrid list
    recs_list = get_hybrid_recommendations_for_user(user_id, K)

    if not recs_list:
        continue  # skip if we could not build a list

    recs_set = set(recs_list)

    hits = len(truth.intersection(recs_set))

    if len(truth) > 0:
        precisions.append(hits / K)
        recalls.append(hits / len(truth))
        ndcgs.append(calculate_ndcg_at_k(recs_list, truth, K))

print(f"Evaluated users: {len(test_users_sample)}")

mean_precision = float(np.mean(precisions)) if precisions else None
mean_recall = float(np.mean(recalls)) if recalls else None
mean_ndcg = float(np.mean(ndcgs)) if ndcgs else None
evaluated_users = len(precisions)

if precisions:
    print(f" Precision@{K}: {mean_precision:.4f}")
    print(f" Recall@{K}: {mean_recall:.4f}")
    print(f" NDCG@{K}: {mean_ndcg:.4f}")
else:
    print(" Not enough users/lists to compute P/R/NDCG.")

# 5. Qualitative check (IPA persona)

print(f"\n 6) IPA persona analysis")
# Use beer_style to find IPA lovers
ipa_reviews = train_reviews_df[train_reviews_df['beer_style'].str.contains("IPA", case=False, na=False)]
ipa_lover_counts = ipa_reviews.groupby('review_profilename').size()
ipa_lover_ratings = ipa_reviews.groupby('review_profilename')['review_overall'].mean()
# Pick users with ≥5 IPA reviews and mean IPA rating ≥4.5
persona_candidates = ipa_lover_counts[ipa_lover_counts >= 5].index.intersection(ipa_lover_ratings[ipa_lover_ratings >= 4.5].index)
persona_summary = None

if not persona_candidates.empty:
    PERSONA_USER = persona_candidates[0]
    print(f" Persona user: '{PERSONA_USER}'")

    # Align with recipes data
    recs_map = recipes_df.drop_duplicates(subset=['beer_name_norm']).set_index('beer_name_norm')

    persona_recs = get_hybrid_recommendations_for_user(PERSONA_USER, k=5)
    persona_summary = {
        "user": PERSONA_USER,
        "recommendations": persona_recs,
    }

    print(f"\n[Top 5 hybrid recommendations]")
    if persona_recs:
        recs_in_recipes_df = recs_map.reindex(persona_recs).reset_index().dropna(subset=['Name'])

        if not recs_in_recipes_df.empty:
            recs_to_print = recs_in_recipes_df[['Name', 'Style', 'ABV', 'IBU']].copy()
            recs_to_print['ABV'] = recs_to_print['ABV'].round(2)
            recs_to_print['IBU'] = recs_to_print['IBU'].round(2)
            recs_to_print.index = range(1, len(recs_to_print) + 1)

            print(recs_to_print)
        else:
            print(" Recommended beers not found in recipe table.")
    else:
        print(" Failed to generate persona recommendations (CBF or CF score too low).")

    # Compare average features of liked vs recommended beers
    liked_beers = train_reviews_df[(train_reviews_df['review_profilename'] == PERSONA_USER) & (train_reviews_df['review_overall'] >= RELEVANCE_THRESHOLD)]['beer_name_norm']
    liked_beers_in_recipes = liked_beers[liked_beers.isin(recs_map.index)]
    recs_in_recipes_names = [r for r in persona_recs if r in recs_map.index]

    if not liked_beers_in_recipes.empty and recs_in_recipes_names:
        numerical_features_all = [col for col in ['ABV', 'IBU', 'OG', 'FG', 'Color'] if col in recs_map.columns]

        # Fill NaNs with column means so we can compare
        temp_map = recs_map.copy()
        for col in numerical_features_all:
            temp_map[col] = temp_map[col].fillna(temp_map[col].mean())

        liked_stats = temp_map.loc[liked_beers_in_recipes][numerical_features_all].mean()
        recs_stats = temp_map.loc[recs_in_recipes_names][numerical_features_all].mean()

        comparison_df = pd.DataFrame({"Liked (Train set)": liked_stats, "Recommended (Hybrid)": recs_stats})
        comparison_df = comparison_df.T[['ABV', 'IBU', 'Color']].round(2)
        print("\n[Recipe cross-check: mean ABV/IBU/Color]")
        print(comparison_df)
    else:
        print("\n [Not enough data for recipe comparison].")
else:
    print(" Could not find an IPA persona user.")

metrics_payload = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "k": K,
    "rmse": rmse,
    "precision_at_k": mean_precision,
    "recall_at_k": mean_recall,
    "ndcg_at_k": mean_ndcg,
    "evaluated_users": evaluated_users,
    "persona": persona_summary,
}

EVAL_METRICS_JSON.write_text(
    json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"\nSaved evaluation metrics JSON: {EVAL_METRICS_JSON}")
print("\n Evaluation script finished")
