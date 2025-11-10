from pathlib import Path

# Get root path: https://moonss-0913.tistory.com/21
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "result"

RAW_REVIEWS_CSV = DATA_DIR / "beer_reviews.csv"
RAW_RECIPES_CSV = DATA_DIR / "recipeData.csv"

OUT_REVIEWS_CLEAN = RESULTS_DIR / "beer_reviews_clean.csv"
OUT_RECIPES_CLEAN = RESULTS_DIR / "recipes_clean.csv"
OUT_LINKAGE = RESULTS_DIR / "beer_name_linkage.csv"
OUT_REPORT = RESULTS_DIR / "filtering_report.txt"
TRAIN_REVIEWS_CSV = RESULTS_DIR / "train_reviews.csv"
TEST_REVIEWS_CSV = RESULTS_DIR / "test_reviews.csv"


def ensure_dirs():
    for path in (DATA_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
