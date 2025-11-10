"""
Beer Recommendation System Model (Final Version)
(CF, CBF, Hybrid)
"""

import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from utils.paths import (
    OUT_REVIEWS_CLEAN,
    OUT_RECIPES_CLEAN,
    TRAIN_REVIEWS_CSV,
    TEST_REVIEWS_CSV,
    ensure_dirs,
)

# ---
# 0. 데이터 로드
# ---
ensure_dirs()

try:
    reviews_df = pd.read_csv(OUT_REVIEWS_CLEAN, low_memory=False)
    recipes_df = pd.read_csv(OUT_RECIPES_CLEAN, low_memory=False)
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

train_reviews_df.to_csv(TRAIN_REVIEWS_CSV, index=False)
test_reviews_df.to_csv(TEST_REVIEWS_CSV, index=False)
print(f"-> {TRAIN_REVIEWS_CSV}, {TEST_REVIEWS_CSV} 파일 저장 완료.")

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

train_reviews_df.to_csv(TRAIN_REVIEWS_CSV, index=False)
test_reviews_df.to_csv(TEST_REVIEWS_CSV, index=False)
print(f"-> {TRAIN_REVIEWS_CSV}, {TEST_REVIEWS_CSV} 파일 저장 완료.")

# ---
# 1. 모델 기반 협업 필터링 (CF)
# [scikit-surprise] SVD 모델 생성하기  https://westlife0615.tistory.com/858
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
    final_features_df = pd.concat([features_scaled_df, style_dummies], axis=1)
    print(f"CBF 특성 생성 완료)")
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
    if beer_norm_name not in item_similarity_df:
        return pd.Series(dtype='float64')

    similar_scores = item_similarity_df[beer_norm_name]

    # 복제 레시피 필터링
    similar_scores = similar_scores[similar_scores < 0.99]

    similar_scores = similar_scores.drop(beer_norm_name, errors='ignore').sort_values(ascending=False)

    return similar_scores.head(top_n)


print("CBF 모델 준비 완료.")

# ---
# 3. 하이브리드 필터링 (Hybrid) - 3가지 추천 동시 생성
# ---
print("\n--- 3. 3가지 추천 목록 생성 (CF, CBF, Hybrid) ---")

# 가중치 (조절 가능)
W_CF = 1.0  # 협업 필터링 가중치
W_CBF = 0.5  # 콘텐츠 기반 가중치


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

        # 가중치 계산
        hybrid_score = (cf_score ** W_CF) * (cbf_max_score ** W_CBF)

        all_scores.append((beer_norm, cf_score, cbf_max_score, hybrid_score))

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
    # 1. 개인 취향 기반
    print("\n[추천 1: 협업 필터링 (CF) - 당신의 취향과 유사한 추천]")
    for beer, cf, cbf, hy in cf_recs:
        print(f"  - {beer} (예상CF점수: {cf:.2f})")

    # 2. 콘텐츠 유사도 기반
    print("\n[추천 2: 콘텐츠 기반 (CBF) - 당신이 좋아한 맥주와 '성분'이 유사한 추천]")
    for beer, cf, cbf, hy in cbf_recs:
        print(f"  - {beer} (유사도(CBF): {cbf:.2f})")

    # 3. Hybrid 추천
    print("\n[추천 3: 하이브리드 (Hybrid) - CF와 CBF를 모두 고려한 추천]")
    for beer, cf, cbf, hy in hybrid_recs:
        print(f"  - {beer} (최종점수: {hy:.2f} | CF {cf:.2f}, CBF {cbf:.2f})")
else:
    print(f"'{TEST_USER}'님에 대한 추천 목록을 생성할 수 없습니다.")
