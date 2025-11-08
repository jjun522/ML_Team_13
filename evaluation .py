import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import math
import json
from scipy.sparse import csr_matrix 
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

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
RANDOM_STATE = 42

# 메모리 문제 해결을 위해 CBF 계산에 사용할 아이템 최대 개수 지정
MAX_CBF_ITEMS = 20000 

# 데이터 로드
print("데이터 로드 중")
try:
    train_reviews_df = pd.read_csv(TRAIN_REVIEWS_PATH, low_memory=False)
    test_reviews_df = pd.read_csv(TEST_REVIEWS_PATH, low_memory=False)
    recipes_df = pd.read_csv(RECIPES_PATH, low_memory=False)
except FileNotFoundError as e:
    print("오류: 파일을 찾을 수 없습니다.", e)
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

# 1. CF 모델 학습 (Surprise SVD)
print("\n 1. CF 모델 (Surprise SVD) 학습 중")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    train_reviews_df[['review_profilename', 'beer_name_norm', 'review_overall']], 
    reader
)

model_svd = SVD(n_factors=N_FACTORS, random_state=RANDOM_STATE, n_epochs=25, lr_all=0.005, reg_all=0.02)
trainset = data.build_full_trainset()
model_svd.fit(trainset)
print(" CF 모델 학습 완료 (Surprise SVD)")

# 2. CBF 모델 (유사도 행렬)
print("\n2. CBF 유사도 행렬 생성 중")

# 훈련셋에 있는 아이템만 추출
item_indices = [trainset.to_raw_iid(i) for i in trainset.all_items()] 

filtered_beer_names = set(item_indices)
recipes_features_filtered = recipes_df[recipes_df['beer_name_norm'].isin(filtered_beer_names)].copy()
recipes_features = recipes_features_filtered.drop_duplicates(subset=['beer_name_norm']).set_index('beer_name_norm')

# 학습된 아이템 목록 기준으로 인덱스 재정렬 (매칭 안 되면 NaN)
recipes_features = recipes_features.reindex(item_indices)
# 모든 피처가 NaN인 행은 제거 
recipes_features = recipes_features.dropna(how='all')

# CBF 아이템 개수 제한 로직
original_cbf_count = len(recipes_features.index)
print(f"CBF 유사도 계산 대상 원본 아이템 개수: {original_cbf_count}개")

if original_cbf_count > MAX_CBF_ITEMS:
    # 가장 리뷰가 많은 상위 MAX_CBF_ITEMS 개만 선택
    top_cbf_items = popular_beers.head(MAX_CBF_ITEMS).index
    recipes_features = recipes_features.loc[recipes_features.index.intersection(top_cbf_items)]
    print(f"-> 메모리 제약을 위해 CBF 아이템 개수를 {len(recipes_features.index)}개로 축소했습니다.")

# 피처 전처리 계속
numerical_features = [col for col in ['ABV', 'IBU', 'OG', 'FG', 'Color'] if col in recipes_features.columns]
for col in numerical_features:
    # NaN 값은 해당 컬럼의 평균으로 대체
    recipes_features[col] = recipes_features[col].fillna(recipes_features[col].mean())
    
scaler = MinMaxScaler()
features_scaled_df = pd.DataFrame(
    scaler.fit_transform(recipes_features[numerical_features]),
    columns=numerical_features,
    index=recipes_features.index
)
if 'Style' in recipes_features.columns:
    style_dummies = pd.get_dummies(recipes_features['Style'], prefix='Style')
    # 원-핫 인코딩 후, 인덱스 재정렬 및 누락된 부분은 0으로 채움
    style_dummies = style_dummies.reindex(recipes_features.index, fill_value=0)
    final_features_df = pd.concat([features_scaled_df, style_dummies], axis=1)
else:
    final_features_df = features_scaled_df

# 최종적으로 남아있을 수 있는 NaN을 0으로 채우고, Inf 값을 가진 행을 제거
final_features_df = final_features_df.fillna(0)
final_features_df = final_features_df[np.isfinite(final_features_df).all(axis=1)]

# 메모리 효율성 및 속도 향상을 위해 희소 행렬(Sparse Matrix)로 변환
features_sparse = csr_matrix(final_features_df.values)

try:
    item_similarity_df = pd.DataFrame(
        cosine_similarity(features_sparse),
        index=final_features_df.index,
        columns=final_features_df.index
    )
    print(f" CBF 유사도 행렬 생성 완료 ({item_similarity_df.shape})")
except MemoryError:
    # 메모리 오류 발생 시 메시지 출력
    print(" 메모리 부족으로 유사도 행렬 생성 실패.")
except Exception as e:
    print(f" 유사도 계산 실패): {e}")
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
    """SVD 모델을 사용하여 예측 평점 반환"""
    prediction = model_svd.predict(str(user_id), str(item_id))
    return prediction.est

def get_content_based_recommendations(beer_norm_name, top_n=1000):
    """특정 맥주와 유사한 맥주 Top-N 반환 (CBF)"""
    if item_similarity_df.empty or beer_norm_name not in item_similarity_df.columns:
        return pd.Series(dtype='float64')
    if beer_norm_name not in item_similarity_df.index:
        return pd.Series(dtype='float64')
        
    sims = item_similarity_df.loc[beer_norm_name].drop(beer_norm_name, errors='ignore')
    sims = sims[sims >= MIN_CBF_SIMILARITY] 
    return sims.sort_values(ascending=False).head(top_n)

def get_hybrid_recommendations_for_user(user_id, k):
    """유저에게 하이브리드 추천 목록 Top-K를 생성"""
    if item_similarity_df.empty: return []
    
    try:
        if trainset.to_inner_uid(str(user_id)) not in trainset.all_users():
            return []
    except ValueError:
        return []

    user_reviews = train_reviews_df[train_reviews_df['review_profilename'] == user_id]
    user_rated = set(user_reviews['beer_name_norm'])
    top_user_beers = user_reviews[user_reviews['review_overall'] >= RELEVANCE_THRESHOLD]['beer_name_norm'].unique()
    
    # CBF 행렬에 있는 맥주만 필터링
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
        
        # CF 예측
        cf_score = predict_cf(user_id, beer_norm)
        
        # CF 예측 점수가 4.0 미만인 경우는 유효하지 않다고 간주하여 필터링
        if cf_score < 4.0: 
            continue
            
        if cbf_score < MIN_CBF_SIMILARITY: continue
        
        # Hybrid Score 계산: (CF_Score ^ W_CF) * (CBF_Score ^ W_CBF)
        hybrid_score = (cf_score ** W_CF) * (cbf_score ** W_CBF)
        all_scores.append((beer_norm, hybrid_score))
        
    hybrid_recs = sorted(all_scores, key=lambda x: x[1], reverse=True)[:k]
    return [beer_name for beer_name, _ in hybrid_recs]

# 4. 정량적 평가 (RMSE, Precision, Recall, NDCG)
print("\n 4. RMSE 계산 중 (Surprise Test Set 사용)")
test_set_df = test_reviews_df[['review_profilename', 'beer_name_norm', 'review_overall']].copy()

def get_surprise_testset(df, trainset):
    test_set = []
    valid_uids = set(trainset.to_raw_uid(i) for i in trainset.all_users()) 
    valid_iids = set(trainset.to_raw_iid(i) for i in trainset.all_items())
    
    for _, row in df.iterrows():
        user = str(row['review_profilename'])
        item = str(row['beer_name_norm'])
        rating = row['review_overall']
     
        if user in valid_uids and item in valid_iids:
            test_set.append((user, item, rating))
    return test_set

testset_surprise = get_surprise_testset(test_set_df, trainset)

rmse = None
if testset_surprise:
    predictions = model_svd.test(testset_surprise)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f" RMSE: {rmse:.4f} (총 {len(predictions)}건 사용)")
else:
    print(" RMSE 계산 불가: 테스트셋 유저/아이템 불일치 (CF 모델 평가 실패)")

print(f"\n 5. Precision@{K}, Recall@{K}, NDCG@{K} 계산 중")

# 훈련셋에 기록이 있는 유저 중 테스트셋에도 있는 유저만 대상으로 평가
test_users_in_train = set(test_reviews_df['review_profilename']).intersection(set(train_reviews_df['review_profilename']))
test_users_sample = list(test_users_in_train) 

# Ground Truth 생성: 테스트셋에서 4.0점 이상 준 맥주 목록
ground_truth_map = test_reviews_df[test_reviews_df['review_overall'] >= RELEVANCE_THRESHOLD].groupby('review_profilename')['beer_name_norm'].apply(set)

precisions, recalls, ndcgs = [], [], []
for user_id in test_users_sample:
    truth = ground_truth_map.get(user_id, set())
    if not truth: continue # Ground Truth가 없으면 평가 스킵
    
    recs_list = get_hybrid_recommendations_for_user(user_id, K)
    
    if not recs_list: continue # 추천 목록 생성 실패 시 스킵
    
    recs_set = set(recs_list)
    
    hits = len(truth.intersection(recs_set))
    
    if len(truth) > 0:
        precisions.append(hits / K)
        recalls.append(hits / len(truth))
        ndcgs.append(calculate_ndcg_at_k(recs_list, truth, K))

print(f" 평가 대상 유저 수: {len(test_users_sample)}명")

if precisions:
    print(f" Precision@{K}: {np.mean(precisions):.4f}")
    print(f" Recall@{K}: {np.mean(recalls):.4f}")
    print(f" NDCG@{K}: {np.mean(ndcgs):.4f}")
else:
    print(" P@K, R@K, NDCG@K 계산 불가 (평가 유저/추천 목록 부족)")

# 5. 정성적 평가 (IPA 페르소나)

print(f"\n 6. 정성적 평가 (IPA 애호가)")
# 'beer_style' 컬럼을 사용하여 IPA 애호가 찾기
ipa_reviews = train_reviews_df[train_reviews_df['beer_style'].str.contains("IPA", case=False, na=False)]
ipa_lover_counts = ipa_reviews.groupby('review_profilename').size()
ipa_lover_ratings = ipa_reviews.groupby('review_profilename')['review_overall'].mean()
# 5개 이상 리뷰 작성 및 평균 평점 4.5점 이상인 IPA 애호가 선택
persona_candidates = ipa_lover_counts[ipa_lover_counts >= 5].index.intersection(ipa_lover_ratings[ipa_lover_ratings >= 4.5].index)

if not persona_candidates.empty:
    PERSONA_USER = persona_candidates[0]
    print(f" 페르소나 선정: '{PERSONA_USER}'")
    
    # 레시피 DB에 있는 이름 정보를 가져오기 위한 필터링
    recs_map = recipes_df.drop_duplicates(subset=['beer_name_norm']).set_index('beer_name_norm')
    
    persona_recs = get_hybrid_recommendations_for_user(PERSONA_USER, k=5)
    
    print(f"\n[Top 5 추천 맥주 목록 (Hybrid)]")
    if persona_recs:
        recs_in_recipes_df = recs_map.reindex(persona_recs).reset_index().dropna(subset=['Name'])
        
        if not recs_in_recipes_df.empty:
            recs_to_print = recs_in_recipes_df[['Name', 'Style', 'ABV', 'IBU']].copy()
            recs_to_print['ABV'] = recs_to_print['ABV'].round(2)
            recs_to_print['IBU'] = recs_to_print['IBU'].round(2)
            recs_to_print.index = range(1, len(recs_to_print) + 1)
            
            print(recs_to_print)
        else:
            print(" 추천된 맥주가 레시피 DB에 존재하지 않음.")
    else:
        print(" 추천 생성 실패 (CBF 유사도 부족 또는 CF 평점 낮음).")
    
    # 선호 맥주와 추천 맥주의 평균 피처 비교
    liked_beers = train_reviews_df[(train_reviews_df['review_profilename'] == PERSONA_USER) & (train_reviews_df['review_overall'] >= RELEVANCE_THRESHOLD)]['beer_name_norm']
    liked_beers_in_recipes = liked_beers[liked_beers.isin(recs_map.index)]
    recs_in_recipes_names = [r for r in persona_recs if r in recs_map.index]
    
    if not liked_beers_in_recipes.empty and recs_in_recipes_names:
        numerical_features_all = [col for col in ['ABV', 'IBU', 'OG', 'FG', 'Color'] if col in recs_map.columns]
        
        # NaN 값을 평균으로 채워 통계 계산을 가능하게 합니다.
        temp_map = recs_map.copy()
        for col in numerical_features_all:
             temp_map[col] = temp_map[col].fillna(temp_map[col].mean())
        
        liked_stats = temp_map.loc[liked_beers_in_recipes][numerical_features_all].mean()
        recs_stats = temp_map.loc[recs_in_recipes_names][numerical_features_all].mean()
        
        comparison_df = pd.DataFrame({"Liked (Train set)": liked_stats, "Recommended (Hybrid)": recs_stats})
        comparison_df = comparison_df.T[['ABV', 'IBU', 'Color']].round(2)
        print("\n[레시피 교차 검증 (평균 ABV/IBU/Color)]")
        print(comparison_df)
    else:
        print("\n [레시피 교차 검증 불가] 데이터 부족.")
else:
    print(" IPA 애호가 페르소나를 찾지 못했습니다.")

print("\n 평가 스크립트 종료")