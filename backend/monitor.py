# backend/monitor.py
from dotenv import load_dotenv

load_dotenv()

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from groq import Groq

from config import (
    DB_PATH,
    ACTION_CATEGORIES,
    MONITOR_MODEL_NAME,
    MONITOR_REQUEST_DELAY_SECONDS,
    MONITOR_MAX_RETRIES,
    MONITOR_RETRY_BACKOFF_SECONDS,
    NOT_RELEVANT_LABEL,
)

logger = logging.getLogger(__name__)

client = Groq()

JUDGE_MODEL_NAME = MONITOR_MODEL_NAME

MONITOR_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "monitoring_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label_judgment": {
                    "type": "string",
                    "enum": ["correct", "incorrect", "uncertain"],
                },
                "label_confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "label_explanation": {
                    "type": "string",
                },
            },
            "required": [
                "label_judgment",
                "label_confidence",
                "label_explanation",
            ],
            "additionalProperties": False,
        },
    },
}


FEW_SHOT_MONITOR_EXAMPLES = [
    {
        "title": "Copenhagen Infrastructure Partners divests ownership of Summerfield Battery to Palisade Investment Partners",
        "description": "Summerfield Battery Energy Storage System is a 240 MW / 960 MWh late-stage construction project in South Australia.",
        "predicted_label": "funding/investment",
        "label_judgment": "correct",
        "label_confidence": "high",
        "label_explanation": "The article is about ownership and investment in a battery energy storage project.",
    },
    {
        "title": "Toyota joins hydrogen truck alliance push",
        "description": "Toyota teams up with Daimler Truck and Volvo Group to scale hydrogen fuel-cell technology for heavy-duty trucks.",
        "predicted_label": "partnership",
        "label_judgment": "correct",
        "label_confidence": "high",
        "label_explanation": "The article describes a partnership related to hydrogen transport technology.",
    },
    {
        "title": "[Lightning Deal] EF ECOFLOW River 3 Plus Portable Power Station at Amazon",
        "description": "A shopping deal for a portable power station and extra battery sold through Amazon.",
        "predicted_label": "market/finance",
        "label_judgment": "incorrect",
        "label_confidence": "high",
        "label_explanation": "Consumer shopping deals are not green energy or climate technology news.",
    },
    {
        "title": "iPhone 18 Pro Max Rumors: Massive Battery, Variable Aperture Camera and 2nm Chip",
        "description": "Apple's next phone is rumored to include a larger battery, improved camera, and new processor.",
        "predicted_label": "other",
        "label_judgment": "incorrect",
        "label_confidence": "high",
        "label_explanation": "Consumer electronics battery rumors are not green energy or climate technology news.",
    },
]


def format_few_shot_examples() -> str:
    """
    Format curated few-shot examples for the monitoring judge prompt.
    """
    examples = []

    for i, example in enumerate(FEW_SHOT_MONITOR_EXAMPLES, start=1):
        examples.append(f"""
Example {i}

Article title: {example["title"]}
Article description: {example["description"]}
Predicted label: {example["predicted_label"]}

Correct judge output:
{{
  "label_judgment": "{example["label_judgment"]}",
  "label_confidence": "{example["label_confidence"]}",
  "label_explanation": "{example["label_explanation"]}"
}}
""".strip())

    return "\n\n".join(examples)


def init_monitoring_table() -> None:
    """
    Create the monitoring_results table if it does not already exist.
    Stores automated evaluation outputs and related metadata.
    """
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


def fetch_classified_articles(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load classified articles from the database for monitoring.
    """
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
    """
    Return article IDs that already have a successful monitoring result.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT article_id FROM monitoring_results")
    rows = cursor.fetchall()
    conn.close()

    return {row[0] for row in rows}


def filter_unmonitored_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only articles that do not yet have monitoring results.
    """
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
    """
    Build the prompt used by the judge model to assess label quality only.
    """
    categories = ", ".join(ACTION_CATEGORIES)

    title = article.get("title", "")
    description = article.get("description", "")
    predicted_label = article.get("label", "")

    few_shot_examples = format_few_shot_examples()

    return f"""
Evaluate this classified news article.

Domain: green energy and climate technology.

Allowed pipeline labels:
{categories}

Use the examples below as guidance for how strict the evaluation should be.

Few-shot examples:
{few_shot_examples}

Now evaluate the new article.

Article title: {title}
Article description: {description}
Predicted label: {predicted_label}

Judge only whether the predicted label is appropriate.

Important rules:
- The project scope is European green energy and climate technology.
- If the article has no clear relevance to Europe, the EU, or a European country, then "{NOT_RELEVANT_LABEL}" is the appropriate label.
- If the predicted label is "{NOT_RELEVANT_LABEL}" and the article has no clear European relevance, then the classifier is correct, even if the article is generally about green energy or climate technology.
- If the article is not primarily about green energy, climate technology, decarbonization, renewable energy infrastructure, clean energy policy, or climate-relevant industrial technology, then labels such as "market/finance", "other", or "new product" are not appropriate.
- Consumer shopping deals, phone battery rumors, power tool battery deals, generic electronics, and unrelated product discounts should usually be judged incorrect unless they are classified as "{NOT_RELEVANT_LABEL}".
- If the predicted label is "{NOT_RELEVANT_LABEL}" and the article is not actually about green energy or climate technology, then the classifier is correct.

Label judgment rules:
- If the predicted label is appropriate, label_judgment should be "correct".
- If the predicted label is inappropriate, label_judgment should be "incorrect".
- If there is not enough information to decide, label_judgment should be "uncertain".

Keep explanations short.
""".strip()

def parse_judge_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse structured JSON returned by the judge model.
    """
    parsed = json.loads(raw_text)

    return {
        "label_judgment": parsed["label_judgment"],
        "label_confidence": parsed["label_confidence"],
        "label_explanation": parsed["label_explanation"],
    }


def derive_overall_status(result: Dict[str, Any], predicted_label: str) -> str:
    """
    Derive a higher-level monitoring status from label judge output.
    """
    label = result["label_judgment"]

    if label == "incorrect":
        return "classification_issue"

    if label == "uncertain":
        return "needs_review"

    return "ok"


def requires_human_review(result: Dict[str, Any], predicted_label: str) -> int:
    """
    Flag label accuracy cases that should be inspected manually.
    """
    if result["label_judgment"] in {"incorrect", "uncertain"}:
        return 1

    if result["label_confidence"] == "low":
        return 1

    return 0


def judge_single_article(article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Evaluate one classified article using the judge model.
    Returns None if all retries fail, so the article can be retried later.
    """
    prompt = build_monitor_prompt(article)

    for attempt in range(1, MONITOR_MAX_RETRIES + 1):
        try:
            logger.info("Using judge model: %s", JUDGE_MODEL_NAME)

            completion = client.chat.completions.create(
                model=JUDGE_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict evaluation assistant. Return concise structured output only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_completion_tokens=700,
                top_p=1,
                stream=False,
                response_format=MONITOR_RESPONSE_SCHEMA,
                stop=None,
            )

            raw_response = completion.choices[0].message.content.strip()
            parsed = parse_judge_response(raw_response)

            predicted_label = article.get("label", "")

            parsed["overall_status"] = derive_overall_status(parsed, predicted_label)
            parsed["requires_human_review"] = requires_human_review(parsed, predicted_label)
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
                return None


def save_monitoring_results(results: List[Dict[str, Any]]) -> None:
    """
    Save successful monitoring results to the database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for row in results:
        cursor.execute("""
            INSERT OR IGNORE INTO monitoring_results (
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

            None,
            None,
            None,

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


def run_monitoring(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run the full monitoring step:
    load classified articles, evaluate unseen ones, and save successful results.
    """
    logger.info("Starting automated monitoring")

    init_monitoring_table()

    classified_articles = fetch_classified_articles(limit=limit)
    logger.info("Loaded %s classified articles", len(classified_articles))

    articles_to_monitor = filter_unmonitored_articles(classified_articles)

    if not articles_to_monitor:
        logger.info("No new classified articles to monitor")
        return []

    monitored_results: List[Dict[str, Any]] = []

    for i, article in enumerate(articles_to_monitor, start=1):
        logger.info("Monitoring article %s/%s", i, len(articles_to_monitor))

        result = judge_single_article(article)

        if result is not None:
            monitored_results.append(result)
        else:
            logger.warning(
                "Skipping article due to monitoring failure: %s",
                article.get("title", ""),
            )

        if i < len(articles_to_monitor):
            time.sleep(MONITOR_REQUEST_DELAY_SECONDS)

    if monitored_results:
        save_monitoring_results(monitored_results)
        logger.info("Finished monitoring and saved %s results", len(monitored_results))
    else:
        logger.info("No successful monitoring results to save")

    return monitored_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_monitoring()