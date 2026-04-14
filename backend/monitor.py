import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from groq import Groq

from config import (
    DB_PATH,
    ACTION_CATEGORIES,
    MONITOR_MODEL_NAME,
    MONITOR_REQUEST_DELAY_SECONDS,
    MONITOR_MAX_RETRIES,
    MONITOR_RETRY_BACKOFF_SECONDS,
)
logger = logging.getLogger(__name__)

client = Groq()

JUDGE_MODEL_NAME = "MONITOR_MODEL_NAME"


def init_monitoring_table() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_results (
            monitoring_id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id TEXT NOT NULL,
            title TEXT,
            description TEXT,
            clean_text TEXT,
            predicted_label TEXT,
            source TEXT,
            url TEXT,
            published_at TEXT,
            classified_at TEXT,

            relevance_judgment TEXT,
            relevance_confidence TEXT,
            relevance_explanation TEXT,

            label_judgment TEXT,
            label_confidence TEXT,
            label_explanation TEXT,

            overall_status TEXT,
            requires_human_review INTEGER,

            judge_model TEXT,
            raw_judge_response TEXT,
            evaluated_at TEXT,

            UNIQUE(article_id)
        )
    """)

    conn.commit()
    conn.close()


def fetch_classified_articles(limit: int | None = None) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT
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
        FROM classified_articles
        ORDER BY published_at DESC
    """

    params: List[Any] = []
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def fetch_already_monitored_article_ids() -> set[str]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT article_id FROM monitoring_results")
    rows = cursor.fetchall()
    conn.close()

    return {row[0] for row in rows}


def filter_unmonitored_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    monitored_ids = fetch_already_monitored_article_ids()
    new_articles = [a for a in articles if a["article_id"] not in monitored_ids]

    logger.info(
        "Monitoring candidates: %s | Already monitored: %s | New to evaluate: %s",
        len(articles),
        len(monitored_ids),
        len(new_articles),
    )

    return new_articles


def build_monitor_prompt(article: Dict[str, Any]) -> str:
    categories = ", ".join(ACTION_CATEGORIES)

    title = article.get("title", "")
    description = article.get("description", "")
    predicted_label = article.get("label", "")

    return f"""
You are evaluating the output of a green energy and climate technology news classification pipeline.

Your task is to evaluate TWO things:

1. Relevance:
Decide whether the article is relevant to the intended domain of green energy and climate technology.
Allowed values:
- relevant
- not_relevant
- uncertain

2. Label quality:
Decide whether the predicted label is appropriate for the article.
Allowed values:
- correct
- incorrect
- uncertain

Possible classification labels in the pipeline are:
{categories}

Article:
Title: {title}
Description: {description}

Predicted label: {predicted_label}

Return your result as valid JSON with exactly these keys:
{{
  "relevance_judgment": "relevant | not_relevant | uncertain",
  "relevance_confidence": "high | medium | low",
  "relevance_explanation": "short explanation",
  "label_judgment": "correct | incorrect | uncertain",
  "label_confidence": "high | medium | low",
  "label_explanation": "short explanation"
}}

Return JSON only. No markdown. No extra text.
""".strip()


def parse_judge_response(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # fallback if the model returns extra whitespace or text around JSON
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(raw_text[start:end + 1])
        else:
            raise

    return {
        "relevance_judgment": parsed.get("relevance_judgment", "uncertain"),
        "relevance_confidence": parsed.get("relevance_confidence", "low"),
        "relevance_explanation": parsed.get("relevance_explanation", ""),
        "label_judgment": parsed.get("label_judgment", "uncertain"),
        "label_confidence": parsed.get("label_confidence", "low"),
        "label_explanation": parsed.get("label_explanation", ""),
    }


def derive_overall_status(result: Dict[str, Any]) -> str:
    relevance = result["relevance_judgment"]
    label = result["label_judgment"]

    if relevance == "not_relevant":
        return "collection_issue"
    if relevance == "uncertain":
        return "needs_review"
    if label == "incorrect":
        return "classification_issue"
    if label == "uncertain":
        return "needs_review"
    return "ok"


def requires_human_review(result: Dict[str, Any]) -> int:
    if result["relevance_judgment"] in {"not_relevant", "uncertain"}:
        return 1
    if result["label_judgment"] in {"incorrect", "uncertain"}:
        return 1
    if result["relevance_confidence"] == "low":
        return 1
    if result["label_confidence"] == "low":
        return 1
    return 0


def judge_single_article(article: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_monitor_prompt(article)

    for attempt in range(1, MONITOR_MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_completion_tokens=300,
                top_p=1,
                stream=False,
                stop=None,
            )

            raw_response = completion.choices[0].message.content.strip()
            parsed = parse_judge_response(raw_response)

            parsed["overall_status"] = derive_overall_status(parsed)
            parsed["requires_human_review"] = requires_human_review(parsed)
            parsed["judge_model"] = JUDGE_MODEL_NAME
            parsed["raw_judge_response"] = raw_response
            parsed["evaluated_at"] = datetime.now(timezone.utc).isoformat()

            result = article.copy()
            result.update(parsed)

            return result

        except Exception as e:
            logger.warning(
                "Monitoring attempt %s/%s failed for '%s': %s",
                attempt,
                MONITOR_MAX_RETRIES,
                article.get("title", ""),
                e,
            )

            if attempt < MONITOR_MAX_RETRIES:
                sleep_time = MONITOR_RETRY_BACKOFF_SECONDS * attempt
                logger.info("Sleeping %.1f seconds before retry", sleep_time)
                time.sleep(sleep_time)
            else:
                logger.error(
                    "Monitoring failed after %s attempts for '%s'",
                    MONITOR_MAX_RETRIES,
                    article.get("title", ""),
                )

                fallback = article.copy()
                fallback.update({
                    "relevance_judgment": "uncertain",
                    "relevance_confidence": "low",
                    "relevance_explanation": f"Judge model failed: {e}",
                    "label_judgment": "uncertain",
                    "label_confidence": "low",
                    "label_explanation": f"Judge model failed: {e}",
                    "overall_status": "needs_review",
                    "requires_human_review": 1,
                    "judge_model": JUDGE_MODEL_NAME,
                    "raw_judge_response": "",
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                })
                return fallback


def save_monitoring_results(results: List[Dict[str, Any]]) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for row in results:
        cursor.execute("""
            INSERT OR REPLACE INTO monitoring_results (
                article_id,
                title,
                description,
                clean_text,
                predicted_label,
                source,
                url,
                published_at,
                classified_at,

                relevance_judgment,
                relevance_confidence,
                relevance_explanation,

                label_judgment,
                label_confidence,
                label_explanation,

                overall_status,
                requires_human_review,

                judge_model,
                raw_judge_response,
                evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("article_id"),
            row.get("title"),
            row.get("description"),
            row.get("clean_text"),
            row.get("label"),
            row.get("source"),
            row.get("url"),
            row.get("published_at"),
            row.get("classified_at"),

            row.get("relevance_judgment"),
            row.get("relevance_confidence"),
            row.get("relevance_explanation"),

            row.get("label_judgment"),
            row.get("label_confidence"),
            row.get("label_explanation"),

            row.get("overall_status"),
            row.get("requires_human_review"),

            row.get("judge_model"),
            row.get("raw_judge_response"),
            row.get("evaluated_at"),
        ))

    conn.commit()
    conn.close()


def run_monitoring(limit: int | None = None) -> List[Dict[str, Any]]:
    logger.info("Starting automated monitoring")

    init_monitoring_table()

    classified_articles = fetch_classified_articles(limit=limit)
    logger.info("Loaded %s classified articles", len(classified_articles))

    articles_to_monitor = filter_unmonitored_articles(classified_articles)

    if not articles_to_monitor:
        logger.info("No new classified articles to monitor")
        return []

    monitored_results = []

    for i, article in enumerate(articles_to_monitor, start=1):
        logger.info("Monitoring article %s/%s", i, len(articles_to_monitor))

        result = judge_single_article(article)
        monitored_results.append(result)

        if i < len(articles_to_monitor):
            time.sleep(MONITOR_REQUEST_DELAY_SECONDS)

    save_monitoring_results(monitored_results)

    logger.info("Finished monitoring and saved %s results", len(monitored_results))
    return monitored_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_monitoring()