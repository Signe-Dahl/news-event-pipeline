#backend/daily_summary.py
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
            top_stories TEXT,
            generated_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def get_latest_classified_articles(limit: int = 25) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            title,
            description,
            label,
            source,
            url,
            published_at
        FROM classified_articles
        ORDER BY published_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "title": row[0],
            "description": row[1],
            "label": row[2],
            "source": row[3],
            "url": row[4],
            "published_at": row[5],
        }
        for row in rows
    ]


def generate_daily_summary() -> Optional[Dict[str, Any]]:
    init_daily_summary_table()

    articles = get_latest_classified_articles()

    if not articles:
        logger.info("No classified articles found. Skipping daily summary.")
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
            max_completion_tokens=1200,
        )

        raw_response = completion.choices[0].message.content.strip()
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()

        summary = json.loads(raw_response)

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
            top_stories,
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