import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

DB_PATH = Path("data/news.db")
OUTPUT_PATH = Path("multilabel_test/multilabel_test_results.csv")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MULTILABEL_CATEGORIES = [
    "policy change",
    "funding/investment",
    "partnership",
    "new invention",
    "infrastructure project",
    "market/finance",
    "new product",
    "other",
    "not relevant to field",
]


def load_test_articles(limit: int = 10) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            article_id,
            title,
            description,
            clean_text,
            label AS original_label,
            source,
            url,
            published_at
        FROM classified_articles
        WHERE label IS NOT NULL
        ORDER BY published_at DESC
        LIMIT ?
    """

    df = pd.read_sql_query(query, conn, params=[limit])
    conn.close()

    return df


def parse_model_output(raw_output: str) -> dict:
    raw_output = raw_output.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "multilabels": ["other"],
            "explanation": "Failed to parse model output.",
            "raw_output": raw_output,
        }

    if isinstance(parsed, list):
        return {
            "multilabels": parsed,
            "explanation": "",
            "raw_output": raw_output,
        }

    if isinstance(parsed, dict):
        labels = parsed.get("multilabels", [])

        if isinstance(labels, str):
            labels = [labels]

        return {
            "multilabels": labels,
            "explanation": parsed.get("explanation", ""),
            "raw_output": raw_output,
        }

    return {
        "multilabels": ["other"],
        "explanation": "Unexpected model output format.",
        "raw_output": raw_output,
    }


def clean_labels(labels: list[str]) -> list[str]:
    allowed = set(MULTILABEL_CATEGORIES)

    cleaned = []
    for label in labels:
        if not isinstance(label, str):
            continue

        normalized = label.strip().lower()

        if normalized in allowed and normalized not in cleaned:
            cleaned.append(normalized)

    if not cleaned:
        return ["other"]

    if "not relevant to field" in cleaned and len(cleaned) > 1:
        return ["not relevant to field"]

    return cleaned


def multilabel_classify_article(row: pd.Series) -> dict:
    prompt = f"""
You are a news event classifier for European green energy and climate technology news.

Classify each article using one or more of these categories:

{", ".join(MULTILABEL_CATEGORIES)}

Rules:
- The article must primarily concern green energy, climate technology, decarbonization, sustainability, renewable energy, clean transport, batteries, hydrogen, electricity infrastructure or climate policy.
- If the article is not clearly relevant to the domain, classify it as "not relevant to field".
- If the article has no explicit connection to Europe, the EU, or a named European country, include "not relevant to field" as one of the labels.
- Articles that are "not relevant to field" may still receive additional event-related labels if appropriate.
- If multiple categories are strongly represented in the article, assign multiple labels.
- Prefer assigning multiple labels when the article clearly combines several event types.
- Order the labels by importance and place the most prominent or dominant category first.
- Only use categories from the predefined list.
- Do not invent new labels.
- Keep the explanation short.

Article title:
{row["title"]}

Article description:
{row["description"]}

Article text:
{row["clean_text"]}

Return valid JSON only in this format:
{{
  "multilabels": ["label1", "label2"],
  "explanation": "short explanation"
}}
""".strip()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a strict news classification assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_completion_tokens=400,
    )

    raw_output = response.choices[0].message.content.strip()
    parsed = parse_model_output(raw_output)
    labels = clean_labels(parsed["multilabels"])

    return {
        "multilabels": ", ".join(labels),
        "multilabel_explanation": parsed["explanation"],
        "raw_multilabel_output": parsed["raw_output"],
    }


def main() -> None:
    df = load_test_articles(limit=10)

    if df.empty:
        print("No classified articles found.")
        return

    results = []

    for _, row in df.iterrows():
        multilabel_result = multilabel_classify_article(row)

        results.append({
            "article_id": row["article_id"],
            "title": row["title"],
            "description": row["description"],
            "source": row["source"],
            "published_at": row["published_at"],
            "original_label": row["original_label"],
            "multilabels": multilabel_result["multilabels"],
            "multilabel_explanation": multilabel_result["multilabel_explanation"],
            "url": row["url"],
        })

    output_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved multilabel test results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()