import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import os
import math

# 0. 설정 및 하이퍼파라미터 

FOLDER_PATH = r"C:\data"
TRAIN_REVIEWS_PATH = os.path.join(FOLDER_PATH, "train_reviews.csv")
TEST_REVIEWS_PATH = os.path.join(FOLDER_PATH, "test_reviews (1).csv")
RECIPES_PATH = os.path.join(FOLDER_PATH, "recipes_clean.csv")

# 튜닝된 하이퍼파라미터
K = 10
RELEVANCE_THRESHOLD = 4.0
W_CF = 1.0
W_CBF = 0.8         
N_FACTORS = 50
MIN_USER_REVIEWS = 10
MIN_BEER_REVIEWS = 10
MIN_CBF_SIMILARITY = 0.4 

# 데이터 로드
print("데이터 로드 중")
try:
    train_reviews_df = pd.read_csv(TRAIN_REVIEWS_PATH, low_memory=False)
    test_reviews_df = pd.read_csv(TEST_REVIEWS_PATH, low_memory=False)
    recipes_df = pd.read_csv(RECIPES_PATH, low_memory=False)
except FileNotFoundError as e:
    print("오류: 파일을 찾을 수 없습니다. 경로를 다시 확인하세요:", e)
    exit()

print(f"훈련셋 {len(train_reviews_df)}, 테스트셋 {len(test_reviews_df)}, 레시피 {len(recipes_df)} 로드 완료.")

# 데이터 필터링
print("\n 최소 리뷰 개수 기준으로 데이터 필터링")
user_counts = train_reviews_df['review_profilename'].value_counts()
active_users = user_counts[user_counts >= MIN_USER_REVIEWS].index
train_reviews_df = train_reviews_df[train_reviews_df['review_profilename'].isin(active_users)]
beer_counts = train_reviews_df['beer_name_norm'].value_counts()
popular_beers = beer_counts[beer_counts >= MIN_BEER_REVIEWS].index
train_reviews_df = train_reviews_df[train_reviews_df['beer_name_norm'].isin(popular_beers)]
print(f"필터링 후 훈련셋 크기: {len(train_reviews_df)}")

# 1. CF 모델 학습 (NMF - Scikit-learn)

print("\n 1. CF 모델 (NMF) 학습 중")
R_df = train_reviews_df.pivot(index='review_profilename', columns='beer_name_norm', values='review_overall').fillna(0)
R = R_df.values
user_indices = R_df.index
item_indices = R_df.columns
model_nmf = NMF(n_components=N_FACTORS, init='random', random_state=42, max_iter=200, tol=1e-3, l1_ratio=0.1)
W = model_nmf.fit_transform(R)
H = model_nmf.components_
R_hat = np.dot(W, H)
R_hat_df = pd.DataFrame(R_hat, index=user_indices, columns=item_indices)
print(" CF 모델 학습 완료 (NMF)")

# 2. CBF 모델 (유사도 행렬)

print("\n2. CBF 유사도 행렬 생성 중")
filtered_beer_names = set(item_indices)
recipes_features_filtered = recipes_df[recipes_df['beer_name_norm'].isin(filtered_beer_names)].copy()
recipes_features = recipes_features_filtered.drop_duplicates(subset=['beer_name_norm']).set_index('beer_name_norm')
recipes_features = recipes_features.reindex(item_indices).dropna(how='all')
numerical_features = [col for col in ['ABV', 'IBU', 'OG', 'FG', 'Color'] if col in recipes_features.columns]
for col in numerical_features:
    recipes_features[col] = recipes_features[col].fillna(recipes_features[col].mean())
scaler = MinMaxScaler()
features_scaled_df = pd.DataFrame(
    scaler.fit_transform(recipes_features[numerical_features]),
    columns=numerical_features,
    index=recipes_features.index
)
if 'Style' in recipes_features.columns:
    style_dummies = pd.get_dummies(recipes_features['Style'], prefix='Style')
    style_dummies = style_dummies.reindex(recipes_features.index, fill_value=0)
    final_features_df = pd.concat([features_scaled_df, style_dummies], axis=1)
else:
    final_features_df = features_scaled_df
try:
    item_similarity_df = pd.DataFrame(
        cosine_similarity(final_features_df),
        index=final_features_df.index,
        columns=final_features_df.index
    )
    print(f" CBF 유사도 행렬 생성 완료 ({item_similarity_df.shape})")
except MemoryError:
    print(" 메모리 부족으로 유사도 행렬 생성 실패")
    item_similarity_df = pd.DataFrame()

# 3. Helper Functions

def calculate_ndcg_at_k(recs, truth, k):
    """Normalized Discounted Cumulative Gain (NDCG) @ K 계산"""
    dcg = 0.0
    for i, item in enumerate(recs[:k]):
        relevance = 1.0 if item in truth else 0.0
        dcg += relevance / np.log2(i + 2)
    n_relevant = len(truth)
    if n_relevant == 0:
        return 0.0
    idcg = 0.0
    for i in range(min(k, n_relevant)):
        idcg += 1.0 / np.log2(i + 2)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def predict_cf(user_id, item_id):
    """NMF 예측 평점 반환 (R_hat_df에서 직접 조회)"""
    try:
        return R_hat_df.loc[user_id, item_id]
    except KeyError:
        return 0.0 

def get_content_based_recommendations(beer_norm_name, top_n=1000):
    """특정 맥주와 유사한 맥주 Top-N 반환 (CBF)"""
    if item_similarity_df.empty or beer_norm_name not in item_similarity_df.columns:
        return pd.Series(dtype='float64')
    sims = item_similarity_df[beer_norm_name].drop(beer_norm_name, errors='ignore')
    sims = sims[sims >= MIN_CBF_SIMILARITY] 
    return sims.sort_values(ascending=False).head(top_n)

def get_hybrid_recommendations_for_user(user_id, k):
    """유저에게 하이브리드 추천 목록 Top-K를 생성"""
    if item_similarity_df.empty: return []
    user_reviews = train_reviews_df[train_reviews_df['review_profilename'] == user_id]
    user_rated = set(user_reviews['beer_name_norm'])
    top_user_beers = user_reviews[user_reviews['review_overall'] >= RELEVANCE_THRESHOLD]['beer_name_norm'].unique()
    top_user_beers = [b for b in top_user_beers if b in item_similarity_df.index]
    if not top_user_beers: return []
    
    candidate_beers = {}
    for beer_norm in top_user_beers:
        similar_beers = get_content_based_recommendations(beer_norm, top_n=k * 10)
        for beer_name, score in similar_beers.items():
            candidate_beers[beer_name] = max(candidate_beers.get(beer_name, 0), score)

    all_scores = []
    for beer_norm, cbf_score in candidate_beers.items():
        if beer_norm in user_rated: continue
        cf_score = predict_cf(user_id, beer_norm)
        
        # CF 예측 점수가 0보다 크면 허용 (NMF가 0으로 예측한 항목은 제외)
        if cf_score <= 0.0: 
            continue
            
        if cbf_score < MIN_CBF_SIMILARITY: continue
        
        # 튜닝된 W_CBF(0.8) 사용
        hybrid_score = (cf_score ** W_CF) * (cbf_score ** W_CBF)
        all_scores.append((beer_norm, hybrid_score))
        
    hybrid_recs = sorted(all_scores, key=lambda x: x[1], reverse=True)[:k]
    return [beer_name for beer_name, _ in hybrid_recs]

# 4. 정량적 평가

print("\n 3. RMSE 계산 중")
preds, actuals = [], []
for _, row in test_reviews_df.iterrows():
    user, item, actual = row['review_profilename'], row['beer_name_norm'], row['review_overall']
  
    if user in R_hat_df.index and item in R_hat_df.columns:
        pred = R_hat_df.loc[user, item]
        
        # NMF는 예측 평점이 0보다 작을 수 없으므로, 유의미한 예측(0보다 큰 값)만 사용
        if pred > 0.0: 
            preds.append(pred)
            actuals.append(actual)

if preds:
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"RMSE: {rmse:.4f} (총 {len(actuals)}건 사용)")
else:
    print(" RMSE 계산 불가: 테스트셋 유저/아이템 불일치 (CF 모델 평가 실패)")

print(f"\n 4. Precision@{K}, Recall@{K}, NDCG@{K} 계산 중")

# 평가 샘플 유저 수 확장: 전체 테스트셋 유저를 대상으로 평가
test_users_in_train = set(test_reviews_df['review_profilename']).intersection(set(train_reviews_df['review_profilename']))
test_users_sample = list(test_users_in_train) 

# Ground Truth 생성
ground_truth_map = test_reviews_df[test_reviews_df['review_overall'] >= RELEVANCE_THRESHOLD].groupby('review_profilename')['beer_name_norm'].apply(set)

precisions, recalls, ndcgs = [], [], []
for user_id in test_users_sample:
    truth = ground_truth_map.get(user_id, set())
    if not truth: continue
    
    recs_list = get_hybrid_recommendations_for_user(user_id, K)
    
    if not recs_list: continue
    
    recs_set = set(recs_list)
    hits = len(truth.intersection(recs_set))
    
    precisions.append(hits / K)
    recalls.append(hits / len(truth))
    ndcgs.append(calculate_ndcg_at_k(recs_list, truth, K))

print(f"평가 대상 유저 수: {len(test_users_sample)}명")

if precisions:
    print(f" Precision@{K}: {np.mean(precisions):.4f}")
    print(f" Recall@{K}: {np.mean(recalls):.4f}")
    print(f" NDCG@{K}: {np.mean(ndcgs):.4f}")
else:
    print(" P@K, R@K, NDCG@K 계산 불가")

# 5. 정성적 평가 (IPA 페르소나)

print(f"\n 5. 정성적 평가 (IPA 애호가)")
ipa_reviews = train_reviews_df[train_reviews_df['beer_style'].str.contains("IPA", case=False, na=False)]
ipa_lover_counts = ipa_reviews.groupby('review_profilename').size()
ipa_lover_ratings = ipa_reviews.groupby('review_profilename')['review_overall'].mean()
persona_candidates = ipa_lover_counts[ipa_lover_counts >= 5].index.intersection(ipa_lover_ratings[ipa_lover_ratings >= 4.5].index)

if not persona_candidates.empty:
    PERSONA_USER = persona_candidates[0]
    print(f"페르소나 선정: '{PERSONA_USER}'")
    
    persona_recs = get_hybrid_recommendations_for_user(PERSONA_USER, k=5)
    
    print(f"\n[Top 5 추천 맥주 목록]")
    if persona_recs:
        recs_in_recipes_df = recipes_df[recipes_df['beer_name_norm'].isin(persona_recs)].copy()
        recs_in_recipes_df = (recs_in_recipes_df.drop_duplicates(subset=['beer_name_norm']).set_index('beer_name_norm').reindex(persona_recs).reset_index().dropna(subset=['Name']))
        
        if not recs_in_recipes_df.empty:
        
            recs_to_print = recs_in_recipes_df[['Name', 'Style', 'ABV', 'IBU']].copy()
            recs_to_print.index = range(1, len(recs_to_print) + 1)
            
            print(recs_to_print)
        else:
            print(" 추천된 맥주가 레시피 DB에 존재하지 않음.")
    else:
        print(" 추천 생성 실패 (CBF 유사도 부족 또는 CF 평점 낮음).")

    liked_beers = train_reviews_df[(train_reviews_df['review_profilename'] == PERSONA_USER) & (train_reviews_df['review_overall'] >= RELEVANCE_THRESHOLD)]['beer_name_norm']
    liked_beers_in_recipes = liked_beers[liked_beers.isin(recipes_df['beer_name_norm'])]
    recs_in_recipes_names = [r for r in persona_recs if r in recipes_df['beer_name_norm'].values]
    
    if not liked_beers_in_recipes.empty and recs_in_recipes_names:
        liked_stats = recipes_df[recipes_df['beer_name_norm'].isin(liked_beers_in_recipes)][numerical_features].mean()
        recs_stats = recipes_df[recipes_df['beer_name_norm'].isin(recs_in_recipes_names)][numerical_features].mean()
        comparison_df = pd.DataFrame({"Liked (Train set)": liked_stats, "Recommended (Hybrid)": recs_stats})
        comparison_df = comparison_df.T[['ABV', 'IBU', 'Color']].round(2)
        print("\n[레시피 교차 검증 (평균 ABV/IBU)]")
        print(comparison_df)
    else:
        print("\n [레시피 교차 검증 불가] 데이터 부족.")
else:
    print(" IPA 애호가 페르소나를 찾지 못했습니다.")

print("\n 평가 스크립트 종료")