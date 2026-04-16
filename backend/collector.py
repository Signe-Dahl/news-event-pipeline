# collector.py
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

import requests

from config import (
    DB_PATH,
    NEWSAPI_API_KEY,
    NEWSAPI_BASE_URL,
    REQUEST_TIMEOUT,
    COLLECTION_PROFILE,
    DATE_LOOKBACK_DAYS,
    get_from_date,
)

logger = logging.getLogger(__name__)

# Create unique article ID
def make_article_id(
    url: str,
    title: str,
    source: str = "",
    published_at: str = "",
) -> str:
    """
    Create a stable unique identifier for each article.
    Prefer URL when available, otherwise fall back to title + source + timestamp.
    """
    if url:
        base = url.strip().lower()
    else:
        base = f"{title.strip().lower()}|{(source or '').strip().lower()}|{(published_at or '').strip()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

# Create data table
def init_raw_articles_table() -> None:
    """
    Create the raw_articles table if it does not already exist.
    Stores original article data and collection metadata.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_articles (
            article_id TEXT PRIMARY KEY,
            profile TEXT,
            matched_keyword TEXT,
            title TEXT,
            description TEXT,
            url TEXT,
            source TEXT,
            category TEXT,
            language TEXT,
            country TEXT,
            published_at TEXT,
            fetched_at TEXT,
            raw_json TEXT
        )
    """)

    conn.commit()
    conn.close()

# Fetch articles from NewsAPI
def fetch_articles(days_back: int = DATE_LOOKBACK_DAYS) -> List[Dict[str, Any]]:
    """
    Request articles from NewsAPI using the configured query and date range.
    """
    if not NEWSAPI_API_KEY:
        raise ValueError("NEWSAPI_API_KEY is missing")

    params = {
        "q": COLLECTION_PROFILE["query"],
        "from": get_from_date(days_back),
        **COLLECTION_PROFILE["base_params"],
    }

    headers = {
        "X-Api-Key": NEWSAPI_API_KEY
    }

    response = requests.get(
        NEWSAPI_BASE_URL,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {data}")

    logger.info(
        "NewsAPI returned %s articles out of totalResults=%s",
        len(data.get("articles", [])),
        data.get("totalResults"),
    )

    return data.get("articles", [])

# Normalize article structure
def normalize_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the raw API response into a consistent internal article structure.
    """
    title = (article.get("title") or "").strip()
    description = (article.get("description") or "").strip()
    url = (article.get("url") or "").strip()
    source = (article.get("source", {}) or {}).get("name", "").strip()
    published_at = (article.get("publishedAt") or "").strip()

    article_id = make_article_id(
        url=url,
        title=title,
        source=source,
        published_at=published_at,
    )

    return {
        "article_id": article_id,
        "profile": COLLECTION_PROFILE["name"],
        "matched_keyword": COLLECTION_PROFILE["query"],
        "title": title,
        "description": description,
        "url": url,
        "source": source,
        "category": None,
        "language": COLLECTION_PROFILE["base_params"].get("language"),
        "country": None,
        "published_at": published_at,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw": article,
    }

# Deduplicate articles within the current batch
def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate articles within the current fetched batch.
    """
    seen = set()
    unique = []

    for article in articles:
        if article["article_id"] in seen:
            continue
        seen.add(article["article_id"])
        unique.append(article)

    return unique

# Get existing article IDs from the database 
def get_existing_article_ids(article_ids: List[str]) -> Set[str]:
    """
    Return the subset of article_ids that already exist in raw_articles.
    """
    if not article_ids:
        return set()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    placeholders = ",".join("?" for _ in article_ids)
    query = f"""
        SELECT article_id
        FROM raw_articles
        WHERE article_id IN ({placeholders})
    """

    cursor.execute(query, article_ids)
    rows = cursor.fetchall()
    conn.close()

    return {row[0] for row in rows}

# Filter out articles already stored
def filter_new_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only articles that are not already stored in the database.
    This prevents reprocessing of previously seen articles across runs.
    """
    article_ids = [article["article_id"] for article in articles]
    existing_ids = get_existing_article_ids(article_ids)

    new_articles = [
        article for article in articles
        if article["article_id"] not in existing_ids
    ]

    logger.info(
        "Existing in DB: %s | New this run: %s",
        len(existing_ids),
        len(new_articles),
    )

    return new_articles

# Save the new articles
def save_raw_articles(articles: List[Dict[str, Any]]) -> None:
    """
    Save new raw articles to the database.
    INSERT OR IGNORE adds an extra safeguard against duplicate primary keys.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for article in articles:
        cursor.execute("""
            INSERT OR IGNORE INTO raw_articles (
                article_id,
                profile,
                matched_keyword,
                title,
                description,
                url,
                source,
                category,
                language,
                country,
                published_at,
                fetched_at,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article["article_id"],
            article["profile"],
            article["matched_keyword"],
            article["title"],
            article["description"],
            article["url"],
            article["source"],
            article["category"],
            article["language"],
            article["country"],
            article["published_at"],
            article["fetched_at"],
            json.dumps(article["raw"], ensure_ascii=False),
        ))

    conn.commit()
    conn.close()

# Full collection pipeline
def collect_articles(days_back: int = DATE_LOOKBACK_DAYS) -> List[Dict[str, Any]]:
    """
    Run the full collection step:
    fetch from API, normalize, deduplicate, filter existing records, and save new rows.
    Returns only new articles for downstream processing.
    """
    logger.info("Collecting articles from the last %s day(s)", days_back)

    init_raw_articles_table()

    raw_articles = fetch_articles(days_back=days_back)
    logger.info("Fetched %s raw articles", len(raw_articles))

    normalized_articles = [normalize_article(article) for article in raw_articles]
    unique_articles = deduplicate_articles(normalized_articles)
    logger.info("Unique within fetched batch: %s", len(unique_articles))

    new_articles = filter_new_articles(unique_articles)

    if new_articles:
        save_raw_articles(new_articles)
        logger.info("Saved %s new raw articles", len(new_articles))
    else:
        logger.info("No new raw articles to save")

    return new_articles

# Simple test run for debugging and validation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    articles = collect_articles()

    print("\nSample:")
    for a in articles[:5]:
        print("-" * 50)
        print(a["title"])
        print(a["published_at"])
        print(a["matched_keyword"])