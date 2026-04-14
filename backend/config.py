# config.py
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "news.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# API setup
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "")
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
REQUEST_TIMEOUT = 30

# Collection settings
DATE_LOOKBACK_DAYS = 2 # set to 2 due to the API free tier having a 24 hour article delay
DEFAULT_LANGUAGE = "en"
DEFAULT_LIMIT = 100

def get_from_date(days_back: int = DATE_LOOKBACK_DAYS) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

COLLECTION_PROFILE = {
    "name": "energy",
    "base_params": {
        "language": DEFAULT_LANGUAGE,
        "searchIn": "title,description",
        "sortBy": "publishedAt",
        "pageSize": DEFAULT_LIMIT,
        "page": 1,
    },
    "query": '("clean energy" OR "renewable energy" OR hydrogen OR solar OR wind OR battery) '
         'AND (project OR plant OR investment OR funding OR partnership OR launch OR development OR deal) '
         'NOT (stocks OR stock OR shares OR market OR keyboard OR gaming OR game OR games OR medicare OR subway OR sale OR discount OR buy OR coupon OR tracklist OR shipping)',
}

# Classification / Rate limiting
CLASSIFIER_REQUEST_DELAY_SECONDS = 2
CLASSIFIER_MAX_RETRIES = 3
CLASSIFIER_RETRY_BACKOFF_SECONDS = 5.0

# Preprocessing rules
EXCLUDED_TITLE_TERMS = [
    "stocks to consider",
    "stocks to watch",
    "watchlist",
    "stock screener",
    "best green energy stocks",
    "top green energy stocks",
    "still a buy",
    "trading up",
    "promising renewable energy stocks",
]

EXCLUDED_DESCRIPTION_TERMS = [
    "marketbeat",
]

MIN_DESCRIPTION_LENGTH = 40

# Classification settings
CLASSIFIER_PROVIDER = "api"
ACTION_CATEGORIES = [
    "new invention",
    "new product",
    "partnership",
    "policy change",
    "infrastructure project",
    "funding/investment",
    "market/finance",
    "other",
]

CLASSIFICATION_SYSTEM_PROMPT = f"""
You are a news event classifier for green energy and climate technology news.
Classify each article into exactly one of these categories:

{", ".join(ACTION_CATEGORIES)}

Return only the category label.
""".strip()

# Monitoring / judge model settings
MONITOR_MODEL_NAME = "openai/gpt-oss-120b"
MONITOR_REQUEST_DELAY_SECONDS = 2.5
MONITOR_MAX_RETRIES = 3
MONITOR_RETRY_BACKOFF_SECONDS = 5.0