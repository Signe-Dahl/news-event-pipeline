#api/app.py
from pathlib import Path
import json
import sqlite3
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Green Energy News API", version="1.0.0")

DB_PATH = Path("/app/data/news.db")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "db_exists": DB_PATH.exists(),
        "db_path": str(DB_PATH),
    }


@app.get("/labels")
def get_labels():
    conn = get_connection()
    query = """
        SELECT DISTINCT label
        FROM classified_articles
        WHERE label IS NOT NULL
        ORDER BY label
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df["label"].dropna().tolist()


@app.get("/sources")
def get_sources():
    conn = get_connection()
    query = """
        SELECT DISTINCT source
        FROM classified_articles
        WHERE source IS NOT NULL
        ORDER BY source
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df["source"].dropna().tolist()


@app.get("/summary/today")
def get_today_summary():
    conn = get_connection()

    query = """
        SELECT
            summary_date,
            short_summary,
            key_focus,
            top_stories,
            generated_at
        FROM daily_summaries
        ORDER BY summary_date DESC
        LIMIT 1
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return {}

    row = df.iloc[0].to_dict()
    row["top_stories"] = json.loads(row["top_stories"]) if row["top_stories"] else []

    return row


@app.get("/summary/daily-actions")
def daily_actions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    conn = get_connection()

    query = """
        SELECT
            date(published_at) AS day,
            label,
            COUNT(*) AS count
        FROM classified_articles
        WHERE 1=1
    """
    params = []

    if start_date:
        query += " AND date(published_at) >= date(?)"
        params.append(start_date)

    if end_date:
        query += " AND date(published_at) <= date(?)"
        params.append(end_date)

    query += """
        GROUP BY date(published_at), label
        ORDER BY day ASC, label ASC
    """

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return df.to_dict(orient="records")


@app.get("/articles")
def get_articles(
    label: Optional[str] = None,
    source: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    conn = get_connection()

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
        WHERE 1=1
    """
    params = []

    if label:
        query += " AND label = ?"
        params.append(label)

    if source:
        query += " AND source = ?"
        params.append(source)

    if start_date:
        query += " AND date(published_at) >= date(?)"
        params.append(start_date)

    if end_date:
        query += " AND date(published_at) <= date(?)"
        params.append(end_date)

    if search:
        query += " AND (lower(title) LIKE ? OR lower(description) LIKE ?)"
        pattern = f"%{search.lower()}%"
        params.extend([pattern, pattern])

    query += " ORDER BY published_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return df.to_dict(orient="records")