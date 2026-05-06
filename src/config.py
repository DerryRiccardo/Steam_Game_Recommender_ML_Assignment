from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

ENV_DATA_PATH = os.getenv("STEAM_DATA_PATH")
PRIMARY_CSV = DATA_DIR / "games_march2025_1000_reviews.csv"
PRIMARY_ZIP = DATA_DIR / "archive (2).zip"
DATASET_META = DATA_DIR / "dataset_sample_meta.json"
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/artermiloff/steam-games-dataset"
DATASET_REVIEW_THRESHOLD = 1000
LOCAL_DATA_PATH_FILE = DATA_DIR / "local_data_path.txt"
BASE_CLEAN_PARQUET = DATA_DIR / "base_clean_games_v1.parquet"
BASE_CLEAN_META = DATA_DIR / "base_clean_games_v1.meta.json"
PROCESSED_PARQUET = DATA_DIR / "processed_games_v2.parquet"
PROCESSED_META = DATA_DIR / "processed_games_v2.meta.json"
CSV_INSIDE_ZIP = "games_march2025_cleaned.csv"
BASE_CLEAN_SCHEMA_VERSION = 2
PROCESSED_SCHEMA_VERSION = 4

DEFAULT_FEATURES = {
    "genres": True,
    "tags": True,
    "categories": True,
    "short_description": True,
    "developers": False,
    "publishers": False,
}

DEFAULT_WEIGHTS = {
    "content": 0.78,
    "rating": 0.15,
    "popularity": 0.05,
    "recency": 0.02,
}
