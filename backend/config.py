# backend/config.py
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

# Source filtering
EXCLUDED_SOURCE_DOMAINS = {
    "slickdeals.net",
}

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
    "query": (
        '("clean energy" OR "renewable energy" OR hydrogen OR solar OR wind OR battery) '
        'AND (project OR plant OR investment OR funding OR partnership OR launch OR development OR deal)'
    ),
}

# Classification / Rate limiting
CLASSIFIER_REQUEST_DELAY_SECONDS = 2
CLASSIFIER_MAX_RETRIES = 3
CLASSIFIER_RETRY_BACKOFF_SECONDS = 5.0

# Preprocessing rules
EXCLUDED_TITLE_TERMS = []
EXCLUDED_DESCRIPTION_TERMS = []

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
    "not relevant to field",
]

CLASSIFICATION_SYSTEM_PROMPT = f"""
You are a news event classifier for European green energy and climate technology news.
Classify each article into exactly one of these categories:

{", ".join(ACTION_CATEGORIES)}

Rules:
- If the article is NOT primarily about green energy, climate technology, decarbonization, or sustainability, you MUST classify it as "not relevant to field".
- If the article has no clear relevance to Europe, the EU, or a European country, classify it as "not relevant to field".
- Only choose another category if the article clearly relates to green energy or climate technology.
- If relevance is weak, indirect, or ambiguous, classify it as "not relevant to field".
- Always return exactly one category label and nothing else.
""".strip()

# Monitoring / judge model settings
MONITOR_MODEL_NAME = "openai/gpt-oss-120b"
MONITOR_REQUEST_DELAY_SECONDS = 4
MONITOR_MAX_RETRIES = 3
MONITOR_RETRY_BACKOFF_SECONDS = 5
NOT_RELEVANT_LABEL = "not relevant to field"

# Daily summary settings
DAILY_SUMMARY_MODEL = "llama-3.1-8b-instant"

DAILY_SUMMARY_SYSTEM_PROMPT = """
You write concise decision-support briefings about green energy and climate technology.
Your role is to identify important signals, emerging patterns, risks, opportunities, and strategic implications.
""".strip()

DAILY_SUMMARY_USER_PROMPT_TEMPLATE = """
You are generating a daily decision-support briefing for professionals working with green energy and climate technology.

Based on the articles below, identify:
- the most important developments
- the strongest emerging patterns
- what decision-makers should pay attention to
- which developments are most actionable or strategically important

Return ONLY valid JSON with this exact structure:
{{
  "executive_summary": "...",
  "key_signal": "...",
  "recommended_focus": "...",
  "decision_implications": [
    "...",
    "..."
  ],
  "watchlist": [
    "...",
    "..."
  ],
  "top_stories": [
    {{
      "article_id": "...",
      "title": "...",
      "label": "...",
      "source": "...",
      "url": "...",
      "published_at": "...",
      "why_it_matters": "...",
      "decision_relevance": "..."
    }}
  ]
}}

Rules:
- For every top_story, copy article_id, title, label, source, url, and published_at exactly from the input article.
- Do not shorten, rewrite, summarize, or invent titles.
- article_id must be copied exactly from the selected input article.
- executive_summary should be concise and analytical.
- key_signal should describe the strongest trend or signal emerging today.
- recommended_focus should explain what decision-makers should monitor closely.
- decision_implications should explain possible strategic or operational implications.
- watchlist should contain concrete developments, risks, markets, technologies, or policies worth monitoring.
- Include maximum 5 top_stories.
- why_it_matters should explain the broader significance.
- decision_relevance should explain why the story matters for strategic decision-making.
- Do not give financial or investment advice.
- Do not invent facts not supported by the articles.
- Focus on signals, momentum, partnerships, policy shifts, infrastructure, scaling, regulation, and market direction.

Articles:
{articles_json}
""".strip()