#backend/daily_summary.py
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from groq import Groq

from config import DB_PATH

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

    prompt = f"""
You are a green energy and climate tech news analyst.

Based on the articles below, write a short daily briefing.

Focus on:
- what happened today
- the most important common pattern across the articles
- what the user should pay attention to

Return ONLY valid JSON with this exact structure:
{{
  "short_summary": "...",
  "key_focus": "...",
  "top_stories": [
    {{
      "title": "...",
      "why_it_matters": "..."
    }}
  ]
}}

Rules:
- Keep the short_summary concise.
- key_focus should explain the single most important trend or topic to watch today.
- Do not give investment advice.
- Include maximum 5 top_stories.
- Explain why each top story matters.
- Do not create or return categories.
- Do not invent information that is not supported by the articles.

Articles:
{json.dumps(articles, ensure_ascii=False)}
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You write concise daily briefings about green energy and climate tech.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_completion_tokens=800,
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
        summary.get("short_summary"),
        summary.get("key_focus"),
        json.dumps(summary.get("top_stories", []), ensure_ascii=False),
        generated_at,
    ))

    conn.commit()
    conn.close()

    logger.info("Daily summary generated for %s", today)

    return summary