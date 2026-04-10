# classifier.py
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from groq import Groq

from config import (
    ACTION_CATEGORIES,
    CLASSIFICATION_SYSTEM_PROMPT,
    DB_PATH,
    CLASSIFIER_REQUEST_DELAY_SECONDS,
    CLASSIFIER_MAX_RETRIES,
    CLASSIFIER_RETRY_BACKOFF_SECONDS,
)

logger = logging.getLogger(__name__)

client = Groq()


def init_classified_articles_table() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classified_articles (
            article_id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            clean_text TEXT,
            label TEXT,
            raw_label TEXT,
            source TEXT,
            url TEXT,
            published_at TEXT,
            classified_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def build_classification_prompt(article: Dict[str, Any]) -> str:
    clean_text = article.get("clean_text", "").strip()

    return f"""
Classify the following article:

{clean_text}
""".strip()


def normalize_label(raw_label: str) -> str:
    cleaned = raw_label.strip().lower()

    for category in ACTION_CATEGORIES:
        if cleaned == category.lower():
            return category

    return "other"


def classify_single_article(article: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_classification_prompt(article)

    for attempt in range(1, CLASSIFIER_MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": CLASSIFICATION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_completion_tokens=20,
                top_p=1,
                stream=False,
                stop=None,
            )

            raw_label = completion.choices[0].message.content.strip()
            label = normalize_label(raw_label)

            classified_article = article.copy()
            classified_article["label"] = label
            classified_article["raw_label"] = raw_label
            classified_article["classified_at"] = datetime.now(timezone.utc).isoformat()

            return classified_article

        except Exception as e:
            logger.warning(
                "Classification attempt %s/%s failed for '%s': %s",
                attempt,
                CLASSIFIER_MAX_RETRIES,
                article.get("title", ""),
                e,
            )

            if attempt < CLASSIFIER_MAX_RETRIES:
                sleep_time = CLASSIFIER_RETRY_BACKOFF_SECONDS * attempt
                logger.info("Sleeping %.1f seconds before retry", sleep_time)
                time.sleep(sleep_time)
            else:
                logger.error(
                    "Classification failed after %s attempts for '%s'",
                    CLASSIFIER_MAX_RETRIES,
                    article.get("title", ""),
                )

                failed_article = article.copy()
                failed_article["label"] = "other"
                failed_article["raw_label"] = ""
                failed_article["classification_error"] = str(e)
                failed_article["classified_at"] = datetime.now(timezone.utc).isoformat()

                return failed_article


def save_classified_articles(articles: List[Dict[str, Any]]) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for article in articles:
        cursor.execute("""
            INSERT OR REPLACE INTO classified_articles (
                article_id,
                title,
                description,
                clean_text,
                label,
                raw_label,
                source,
                url,
                published_at,
                classified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            article.get("article_id"),
            article.get("title"),
            article.get("description"),
            article.get("clean_text"),
            article.get("label"),
            article.get("raw_label"),
            article.get("source"),
            article.get("url"),
            article.get("published_at"),
            article.get("classified_at"),
        ))

    conn.commit()
    conn.close()


def classify_articles(processed_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("Starting classification for %s articles", len(processed_articles))
    logger.info(
        "Classifier settings: delay=%.1fs, retries=%s, backoff=%.1fs",
        CLASSIFIER_REQUEST_DELAY_SECONDS,
        CLASSIFIER_MAX_RETRIES,
        CLASSIFIER_RETRY_BACKOFF_SECONDS,
    )

    init_classified_articles_table()

    classified_articles = []

    for i, article in enumerate(processed_articles, start=1):
        logger.info("Classifying article %s/%s", i, len(processed_articles))

        classified_article = classify_single_article(article)
        classified_articles.append(classified_article)

        if i < len(processed_articles):
            time.sleep(CLASSIFIER_REQUEST_DELAY_SECONDS)

    save_classified_articles(classified_articles)

    logger.info("Finished classification and saved %s articles", len(classified_articles))

    return classified_articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_articles = [
        {
            "article_id": "1",
            "title": "Google’s new clean energy deal includes massive battery",
            "description": "Form Energy's iron-air batteries will store wind and solar power to keep the data center running 24/7.",
            "source": "TechCrunch",
            "url": "https://example.com/article",
            "published_at": "2026-02-24T21:32:22+00:00",
            "clean_text": "Title: Google’s new clean energy deal includes massive battery\nDescription: Form Energy's iron-air batteries will store wind and solar power to keep the data center running 24/7.",
        }
    ]

    results = classify_articles(sample_articles)

    for article in results:
        print("-" * 50)
        print("Title:", article["title"])
        print("Label:", article["label"])
        print("Raw label:", article["raw_label"])