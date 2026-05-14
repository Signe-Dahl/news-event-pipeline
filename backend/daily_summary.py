# backend/daily_summary.py
from dotenv import load_dotenv
load_dotenv()

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from groq import Groq

from config import (
    DB_PATH,
    DAILY_SUMMARY_MODEL,
    DAILY_SUMMARY_SYSTEM_PROMPT,
    DAILY_SUMMARY_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

client = Groq()


def init_daily_summary_table() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summaries (
            summary_date TEXT PRIMARY KEY,
            short_summary TEXT,
            key_focus TEXT,
            summary_json TEXT,
            generated_at TEXT
        )
    """)

    cursor.execute("PRAGMA table_info(daily_summaries)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    if "summary_json" not in existing_columns:
        cursor.execute("ALTER TABLE daily_summaries ADD COLUMN summary_json TEXT")

    conn.commit()
    conn.close()


def get_latest_classified_articles(limit: int = 15) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            article_id,
            title,
            description,
            label,
            source,
            url,
            published_at
        FROM classified_articles
        WHERE label != 'not relevant to field'
        ORDER BY published_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "article_id": row[0],
            "title": row[1],
            "description": row[2],
            "label": row[3],
            "source": row[4],
            "url": row[5],
            "published_at": row[6],
        }
        for row in rows
    ]


def normalize_title(title: str) -> str:
    return (
        str(title or "")
        .lower()
        .replace("—", "-")
        .split(" - ")[0]
        .strip()
    )


def enrich_top_stories_with_metadata(
    summary: Dict[str, Any],
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stories = summary.get("top_stories", [])

    if not isinstance(stories, list):
        summary["top_stories"] = []
        return summary

    articles_by_id = {
        str(article.get("article_id")): article
        for article in articles
        if article.get("article_id")
    }

    for story in stories:
        if not isinstance(story, dict):
            continue

        article_id = story.get("article_id")
        matched_article = articles_by_id.get(str(article_id)) if article_id else None

        if matched_article is None:
            story_title = story.get("title", "")
            normalized_story_title = normalize_title(story_title)

            for article in articles:
                normalized_article_title = normalize_title(article.get("title", ""))

                if (
                    normalized_story_title == normalized_article_title
                    or normalized_story_title in normalized_article_title
                    or normalized_article_title in normalized_story_title
                ):
                    matched_article = article
                    break

        if matched_article:
            story["article_id"] = matched_article["article_id"]
            story["title"] = matched_article["title"]
            story["label"] = matched_article["label"]
            story["source"] = matched_article["source"]
            story["url"] = matched_article["url"]
            story["published_at"] = matched_article["published_at"]

    return summary


def generate_daily_summary() -> Optional[Dict[str, Any]]:
    init_daily_summary_table()

    articles = get_latest_classified_articles()

    if not articles:
        logger.info("No relevant classified articles found. Skipping daily summary.")
        return None

    prompt = DAILY_SUMMARY_USER_PROMPT_TEMPLATE.format(
        articles_json=json.dumps(articles, ensure_ascii=False)
    )

    try:
        completion = client.chat.completions.create(
            model=DAILY_SUMMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": DAILY_SUMMARY_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_completion_tokens=2500,
        )

        raw_response = completion.choices[0].message.content.strip()
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()

        summary = json.loads(raw_response)
        summary = enrich_top_stories_with_metadata(summary, articles)

    except Exception as error:
        logger.exception("Failed to generate daily summary: %s", error)
        return None

    today = datetime.now(timezone.utc).date().isoformat()
    generated_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO daily_summaries (
            summary_date,
            short_summary,
            key_focus,
            summary_json,
            generated_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        today,
        summary.get("executive_summary"),
        summary.get("recommended_focus"),
        json.dumps(summary, ensure_ascii=False),
        generated_at,
    ))

    conn.commit()
    conn.close()

    logger.info("Daily summary generated for %s", today)

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_daily_summary()