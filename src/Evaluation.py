import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import json
from datetime import datetime
from scipy.sparse import csr_matrix
# Surprise 라이브러리: 추천 시스템 모델 구축 및 평가에 특화된 파이썬 라이브러리
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

# 0. 설정 및 하이퍼파라미터

# 튜닝된 하이퍼파라미터
K = 10 # 추천할 아이템 개수 (Top-K)
RELEVANCE_THRESHOLD = 4.0 # 관련 항목(Ground Truth)을 정의하는 평점 임계값
W_CF = 1.0 # CF 점수에 부여할 가중치
W_CBF = 0.8 # CBF 점수에 부여할 가중치
N_FACTORS = 50 # SVD 모델의 잠재 요인(Latent Factors) 개수
MIN_USER_REVIEWS = 10 # 최소 리뷰 개수 미만인 유저는 제외
MIN_BEER_REVIEWS = 10 # 최소 리뷰 개수 미만인 맥주는 제외
MIN_CBF_SIMILARITY = 0.4 # CBF 추천 후보군에 포함되기 위한 최소 콘텐츠 유사도
RANDOM_STATE = 42

# 메모리 문제 해결을 위해 CBF 계산에 사용할 아이템 최대 개수 지정
MAX_CBF_ITEMS = 20000

# 데이터 로드
print("데이터 로드 중")
try:
    train_reviews_df = pd.read_csv(TRAIN_REVIEWS_CSV, low_memory=False)
    test_reviews_df = pd.read_csv(TEST_REVIEWS_CSV, low_memory=False)
    recipes_df = pd.read_csv(OUT_RECIPES_CLEAN, low_memory=False)
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
# Reader 객체 생성: 평점의 스케일(1점부터 5점까지)을 Surprise에 알려줌
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    # Surprise 데이터셋 형식으로 변환: 사용자, 아이템, 평점 컬럼만 사용
    train_reviews_df[['review_profilename', 'beer_name_norm', 'review_overall']],
    reader
)

# SVD(Singular Value Decomposition) 모델 초기화
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

# MinMaxScaler를 사용하여 수치형 피처 정규화
# 각 피처의 값을 0과 1 사이로 스케일링하여 유사도 계산 시 특정 피처의 값 크기에 의한 편향을 방지
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
final_features_df = final_features_df.fillna(0).astype(np.float64)
final_features_df = final_features_df[np.isfinite(final_features_df).all(axis=1)]

# 메모리 효율성 및 속도 향상을 위해 희소 행렬(Sparse Matrix)로 변환 (csr_matrix 사용)
features_sparse = csr_matrix(final_features_df.values)

try:
    # 코사인 유사도 계산: 두 벡터(아이템의 피처) 간의 각도 코사인 값을 측정하여 유사도 정의
    item_similarity_df = pd.DataFrame(
        cosine_similarity(features_sparse),
        index=final_features_df.index,
        columns=final_features_df.index
    )
    print(f" CBF 유사도 행렬 생성 완료 ({item_similarity_df.shape})")
except MemoryError:
    print(" 메모리 부족으로 유사도 행렬 생성 실패.")
except Exception as e:
    print(f" 유사도 계산 실패): {e}")
    item_similarity_df = pd.DataFrame()

# 3. Helper Functions
def calculate_ndcg_at_k(recs, truth, k):
    """Normalized Discounted Cumulative Gain (NDCG) @ K 계산"""
    # NDCG는 추천 목록의 순서까지 고려하여 랭킹 정확도를 측정하는 지표
    dcg = 0.0
    for i, item in enumerate(recs[:k]):
        # 관련성(relevance)은 추천된 항목이 Ground Truth에 있으면 1.0, 아니면 0.0
        relevance = 1.0 if item in truth else 0.0
        # DCG 계산: 관련성을 순위 디스카운트(log2(i+2))로 나누어 합산
        dcg += relevance / np.log2(i + 2)
    n_relevant = len(truth)
    if n_relevant == 0:
        return 0.0
    idcg = 0.0
    # IDCG 계산: 최적의 순서일 때의 DCG 값 (정규화 기준)
    for i in range(min(k, n_relevant)):
        idcg += 1.0 / np.log2(i + 2)
    if idcg == 0.0:
        return 0.0
    # NDCG는 DCG를 IDCG로 정규화한 값 (0과 1 사이)
    return dcg / idcg

def predict_cf(user_id, item_id):
    """SVD 모델을 사용하여 예측 평점 반환"""
    # Surprise SVD 모델을 사용하여 특정 유저(user_id)가 특정 아이템(item_id)에 줄 평점을 예측
    prediction = model_svd.predict(str(user_id), str(item_id))
    return prediction.est

def get_content_based_recommendations(beer_norm_name, top_n=1000):
    """특정 맥주와 유사한 맥주 Top-N 반환 (CBF)"""
    if item_similarity_df.empty or beer_norm_name not in item_similarity_df.columns:
        return pd.Series(dtype='float64')
    if beer_norm_name not in item_similarity_df.index:
        return pd.Series(dtype='float64')

    # 해당 맥주와 다른 모든 맥주 간의 유사도 점수 추출
    sims = item_similarity_df.loc[beer_norm_name].drop(beer_norm_name, errors='ignore')
    # 최소 유사도 임계값(MIN_CBF_SIMILARITY) 미만의 항목은 제외
    sims = sims[sims >= MIN_CBF_SIMILARITY]
    # 유사도 순으로 정렬하여 Top-N 반환
    return sims.sort_values(ascending=False).head(top_n)

def get_hybrid_recommendations_for_user(user_id, k):
    """유저에게 하이브리드 추천 목록 Top-K를 생성"""
    if item_similarity_df.empty: return []

    try:
        # 유저가 훈련셋에 존재하는지 확인
        if trainset.to_inner_uid(str(user_id)) not in trainset.all_users():
            return []
    except ValueError:
        return []

    user_reviews = train_reviews_df[train_reviews_df['review_profilename'] == user_id]
    user_rated = set(user_reviews['beer_name_norm'])
    # 유저가 긍정적으로 평가한(RELEVANCE_THRESHOLD 이상) 맥주 목록 추출 (CBF 추천의 시드)
    top_user_beers = user_reviews[user_reviews['review_overall'] >= RELEVANCE_THRESHOLD]['beer_name_norm'].unique()

    # CBF 행렬에 있는 맥주만 필터링
    top_user_beers = [b for b in top_user_beers if b in item_similarity_df.index]
    if not top_user_beers: return []

    candidate_beers = {}
    # 긍정 평가 맥주 기반으로 유사한 맥주를 CBF 방식으로 찾고, 가장 높은 유사도를 후보 점수로 저장
    for beer_norm in top_user_beers:
        similar_beers = get_content_based_recommendations(beer_norm, top_n=k * 10)
        for beer_name, score in similar_beers.items():
            candidate_beers[beer_name] = max(candidate_beers.get(beer_name, 0), score)

    all_scores = []
    for beer_norm, cbf_score in candidate_beers.items():
        if beer_norm in user_rated: continue # 이미 평가한 맥주는 추천에서 제외

        # CF 예측 평점 계산 (Surprise SVD 모델 사용)
        cf_score = predict_cf(user_id, beer_norm)

        # CF 예측 점수가 4.0 미만인 경우는 유효하지 않다고 간주하여 필터링 (품질 필터링)
        if cf_score < 4.0:
            continue

        if cbf_score < MIN_CBF_SIMILARITY: continue

        # Hybrid Score 계산: (CF_Score ^ W_CF) * (CBF_Score ^ W_CBF)
        # CF와 CBF 점수에 가중치를 적용하여 최종 하이브리드 추천 점수를 산출
        hybrid_score = (cf_score ** W_CF) * (cbf_score ** W_CBF)
        all_scores.append((beer_norm, hybrid_score))

    hybrid_recs = sorted(all_scores, key=lambda x: x[1], reverse=True)[:k]
    return [beer_name for beer_name, _ in hybrid_recs]

# 4. 정량적 평가 (RMSE, Precision, Recall, NDCG)
print("\n 4. RMSE 계산 중 (Surprise Test Set 사용)")
test_set_df = test_reviews_df[['review_profilename', 'beer_name_norm', 'review_overall']].copy()

def get_surprise_testset(df, trainset):
    """평가 대상 유저/아이템이 훈련셋에 있는지 확인하여 유효한 Surprise 테스트셋 생성"""
    test_set = []
    valid_uids = set(trainset.to_raw_uid(i) for i in trainset.all_users())
    valid_iids = set(trainset.to_raw_iid(i) for i in trainset.all_items())

    for _, row in df.iterrows():
        user = str(row['review_profilename'])
        item = str(row['beer_name_norm'])
        rating = row['review_overall']

        # 훈련셋에 있는 사용자(UID)와 아이템(IID)만 포함하여 평가 일관성 유지
        if user in valid_uids and item in valid_iids:
            test_set.append((user, item, rating))
    return test_set

testset_surprise = get_surprise_testset(test_set_df, trainset)

rmse = None
if testset_surprise:
    # SVD 모델의 예측 평점 계산
    predictions = model_svd.test(testset_surprise)
    # RMSE (제곱 평균 제곱근 오차) 계산: 예측 정확도의 주 지표
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

    # Hybrid 추천 목록 Top-K 생성
    recs_list = get_hybrid_recommendations_for_user(user_id, K)

    if not recs_list: continue # 추천 목록 생성 실패 시 스킵

    recs_set = set(recs_list)

    hits = len(truth.intersection(recs_set))

    if len(truth) > 0:
        # Precision@K: 추천 목록 중 관련 항목의 비율
        precisions.append(hits / K)
        # Recall@K: 관련 항목 중 추천된 항목의 비율
        recalls.append(hits / len(truth))
        # NDCG@K: 순서 가중치를 고려한 랭킹 정확도
        ndcgs.append(calculate_ndcg_at_k(recs_list, truth, K))

print(f" 평가 대상 유저 수: {len(test_users_sample)}명")

mean_precision = float(np.mean(precisions)) if precisions else None
mean_recall = float(np.mean(recalls)) if recalls else None
mean_ndcg = float(np.mean(ndcgs)) if ndcgs else None
evaluated_users = len(precisions)

if precisions:
    print(f" Precision@{K}: {mean_precision:.4f}")
    print(f" Recall@{K}: {mean_recall:.4f}")
    print(f" NDCG@{K}: {mean_ndcg:.4f}")
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
persona_summary = None

if not persona_candidates.empty:
    PERSONA_USER = persona_candidates[0]
    print(f" 페르소나 선정: '{PERSONA_USER}'")

    # 레시피 DB에 있는 이름 정보를 가져오기 위한 필터링
    recs_map = recipes_df.drop_duplicates(subset=['beer_name_norm']).set_index('beer_name_norm')

    persona_recs = get_hybrid_recommendations_for_user(PERSONA_USER, k=5)
    persona_summary = {
        "user": PERSONA_USER,
        "recommendations": persona_recs,
    }

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
print(f"\n평가 지표 JSON 저장 완료: {EVAL_METRICS_JSON}")
print("\n 평가 스크립트 종료")
