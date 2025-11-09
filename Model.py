"""
Beer Recommendation System Model (Final Version)
(CF, CBF, Hybrid)
"""

import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ---
# 0. 데이터 로드
# ---
REVIEWS_CLEAN_PATH = "/beer_reviews_clean.csv"
RECIPES_CLEAN_PATH = "/recipes_clean.csv"

OUT_TRAIN_REVIEWS = "train_reviews.csv"
OUT_TEST_REVIEWS = "test_reviews.csv"

try:
    reviews_df = pd.read_csv(REVIEWS_CLEAN_PATH, low_memory=False)
    recipes_df = pd.read_csv(RECIPES_CLEAN_PATH, low_memory=False)
except FileNotFoundError:
    print("오류: CSV 파일을 찾을 수 없습니다.")
    exit()

if 'review_time' not in reviews_df.columns:
    print("오류: 'review_time' 컬럼이 없어 분할이 불가능합니다.")
    exit()

print("\n--- 0. 시간순 데이터 분할 시작 ---")
reviews_df['review_time'] = pd.to_datetime(reviews_df['review_time'], unit='s')
reviews_df = reviews_df.sort_values(by='review_time')

split_point = int(len(reviews_df) * 0.8)
train_reviews_df = reviews_df.iloc[:split_point]
test_reviews_df = reviews_df.iloc[split_point:]

print(f"훈련셋 (80%): {len(train_reviews_df)}개")
print(f"테스트셋 (20%): {len(test_reviews_df)}개")

train_reviews_df.to_csv(OUT_TRAIN_REVIEWS, index=False)
test_reviews_df.to_csv(OUT_TEST_REVIEWS, index=False)
print(f"-> {OUT_TRAIN_REVIEWS}, {OUT_TEST_REVIEWS} 파일 저장 완료.")

if 'review_time' not in reviews_df.columns:
    print("오류: 'review_time' 컬럼이 없어 분할이 불가능합니다.")
    exit()

print("\n--- 0. 시간순 데이터 분할 시작 ---")
reviews_df['review_time'] = pd.to_datetime(reviews_df['review_time'], unit='s')
reviews_df = reviews_df.sort_values(by='review_time')

split_point = int(len(reviews_df) * 0.8)
train_reviews_df = reviews_df.iloc[:split_point]
test_reviews_df = reviews_df.iloc[split_point:]

print(f"훈련셋 (80%): {len(train_reviews_df)}개")
print(f"테스트셋 (20%): {len(test_reviews_df)}개")

train_reviews_df.to_csv(OUT_TRAIN_REVIEWS, index=False)
test_reviews_df.to_csv(OUT_TEST_REVIEWS, index=False)
print(f"-> {OUT_TRAIN_REVIEWS}, {OUT_TEST_REVIEWS} 파일 저장 완료.")

# ---
# 1. 모델 기반 협업 필터링 (CF)
# ---
print("\n--- 1. 협업 필터링(CF) 모델 학습 ---")
reader = Reader(rating_scale=(1, 5))
full_train_data = Dataset.load_from_df(train_reviews_df[['review_profilename', 'beer_name_norm', 'review_overall']], reader)
full_trainset = full_train_data.build_full_trainset()
algo_svd = SVD(n_factors=50, n_epochs=20, random_state=42)
algo_svd.fit(full_trainset)

# ---
# 2. 콘텐츠 기반 필터링 (CBF)
# ---
print("\n--- 2. 콘텐츠 기반 필터링(CBF) 모델 준비 ---")

SAMPLING_SIZE = 20000
if len(recipes_df) > SAMPLING_SIZE:
    print(f"원본 레시피 {len(recipes_df)}개")
    recipes_df = recipes_df.sample(n=SAMPLING_SIZE, random_state=42)

recipes_features = recipes_df.drop_duplicates(subset=['beer_name_norm']).set_index('beer_name_norm')

# 피처 정의
numerical_features = ['ABV', 'IBU', 'OG', 'FG', 'Color']

for col in numerical_features:
    if col in recipes_features.columns:
        recipes_features[col] = recipes_features[col].fillna(recipes_features[col].mean())
    else:
        print(f"경고: '{col}' 컬럼이 recipes_df에 없습니다. 해당 컬럼을 제외합니다.")
        numerical_features.remove(col)

# 수치형 피처 스케일링
scaler = MinMaxScaler()
features_scaled_df = pd.DataFrame(
    scaler.fit_transform(recipes_features[numerical_features]),
    columns=numerical_features,
    index=recipes_features.index
)

# 범주형 피처 원-핫 인코딩
if 'Style' in recipes_features.columns:
    style_dummies = pd.get_dummies(recipes_features['Style'], prefix='Style')
    # 수치형 + 범주형 피처 결합
    final_features_df = pd.concat([features_scaled_df, style_dummies], axis=1)
    print(f"CBF 특성 생성 완료. (수치형 {len(numerical_features)}개 + 스타일 {len(style_dummies.columns)}개)")
else:
    print("경고: 'Style' 컬럼이 없습니다. 수치형 특성만 사용합니다.")
    final_features_df = features_scaled_df

print("코사인 유사도 계산 시작...")
item_similarity_df = pd.DataFrame(
    cosine_similarity(final_features_df),
    index=final_features_df.index,
    columns=final_features_df.index
)
print("코사인 유사도 계산 완료.")


def get_content_based_recommendations(beer_norm_name, top_n=5):
    """
    특정 맥주와 모든 특성이 가장 유사한 맥주 Top-N 반환
    (단, 100% 동일 및 99% 이상 '복제 맥주'는 제외)
    """
    if beer_norm_name not in item_similarity_df:
        return pd.Series(dtype='float64')

    similar_scores = item_similarity_df[beer_norm_name]

    # 1.00 (완벽히 일치) 및 0.99 이상 (사실상 복제) 맥주를 필터링
    similar_scores = similar_scores[similar_scores < 0.99]

    similar_scores = similar_scores.drop(beer_norm_name, errors='ignore').sort_values(ascending=False)

    return similar_scores.head(top_n)


print("CBF 모델 준비 완료.")

# ---
# 3. 하이브리드 필터링 (Hybrid) - 3가지 추천 동시 생성
# ---
print("\n--- 3. 3가지 추천 목록 생성 (CF, CBF, Hybrid) ---")

# 가중치 (조절 가능)
W_CF = 1.0  # 협업 필터링(개인 취향) 가중치
W_CBF = 0.5  # 콘텐츠 기반(유사도) 가중치


def get_all_recommendations(user_id, top_n=10):
    """
    한 명의 유저에 대해 3가지 추천 목록을 모두 반환
    1. CF 기반 추천
    2. CBF 기반 추천
    3. Hybrid (CF * CBF) 추천
    """

    user_reviews = train_reviews_df[train_reviews_df['review_profilename'] == user_id]
    user_rated_beers = set(user_reviews['beer_name_norm'])
    top_user_beers = user_reviews[user_reviews['review_overall'] >= 4.0]['beer_name_norm'].unique()

    if len(top_user_beers) == 0:
        print(f"'{user_id}'님은 평점 4.0 이상인 맥주가 없습니다. (콜드 스타트)")
        return None, None, None

    # CBF 후보군 점수 저장
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

        # Hybrid 점수 계산
        hybrid_score = (cf_score ** W_CF) * (cbf_max_score ** W_CBF)

        all_scores.append((beer_norm, cf_score, cbf_max_score, hybrid_score))

    # 4. 3가지 리스트 생성
    cf_recs = sorted(all_scores, key=lambda x: x[1], reverse=True)[:top_n]
    cbf_recs = sorted(all_scores, key=lambda x: x[2], reverse=True)[:top_n]
    hybrid_recs = sorted(all_scores, key=lambda x: x[3], reverse=True)[:top_n]

    return cf_recs, cbf_recs, hybrid_recs


# ---
# 4. 최종 추천 예시
# ---
print("\n--- 테스트용 Top 5 리뷰어 ---")
top_reviewers = train_reviews_df['review_profilename'].value_counts().head(5)
print(top_reviewers)
print("------------------------------")

TEST_USER = 'stakem'
cf_recs, cbf_recs, hybrid_recs = get_all_recommendations(TEST_USER, top_n=10)

print(f"\n--- [최종 추천 결과 (for {TEST_USER})] ---")

if cf_recs:
    # 1. CF 추천 (개인 취향 기반)
    print("\n[추천 1: 협업 필터링 (CF) - 당신의 취향과 유사한 추천]")
    for beer, cf, cbf, hy in cf_recs:
        print(f"  - {beer} (예상CF점수: {cf:.2f})")

    # 2. CBF 추천 (콘텐츠 유사도 기반)
    print("\n[추천 2: 콘텐츠 기반 (CBF) - 당신이 좋아한 맥주와 '성분'이 유사한 추천]")
    for beer, cf, cbf, hy in cbf_recs:
        print(f"  - {beer} (유사도(CBF): {cbf:.2f})")

    # 3. Hybrid 추천 (가중치 곱셈)
    print("\n[추천 3: 하이브리드 (Hybrid) - CF와 CBF를 모두 고려한 추천]")
    for beer, cf, cbf, hy in hybrid_recs:
        print(f"  - {beer} (최종점수: {hy:.2f} | CF {cf:.2f}, CBF {cbf:.2f})")
else:
    print(f"'{TEST_USER}'님에 대한 추천 목록을 생성할 수 없습니다.")