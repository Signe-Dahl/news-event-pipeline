# api/app.py
from pathlib import Path
import sqlite3
from typing import Optional
import json

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Green Energy News API", version="1.0.0")

import os
from pathlib import Path

# Test - fjernes efter lokal test
DB_PATH = Path(
    os.getenv(
        "DB_PATH",
        Path(__file__).resolve().parent.parent / "data" / "news.db"
    )
)
# DB_PATH = Path("/app/data/news.db")

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
            AND label != 'not relevant to field'
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
            AND label != 'not relevant to field'
        ORDER BY source
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df["source"].dropna().tolist()

@app.get("/summary/daily")
def get_daily_summary():
    conn = get_connection()

    query = """
        SELECT
            summary_date,
            short_summary,
            key_focus,
            summary_json,
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

    result = {
        "summary_date": row.get("summary_date"),
        "generated_at": row.get("generated_at"),
    }

    summary_json = row.get("summary_json")

    if summary_json:
        try:
            parsed_summary = json.loads(summary_json)

            if isinstance(parsed_summary, dict):
                result.update(parsed_summary)

        except Exception:
            pass

    # fallback compatibility
    if "executive_summary" not in result:
        result["executive_summary"] = row.get("short_summary")

    if "recommended_focus" not in result:
        result["recommended_focus"] = row.get("key_focus")

    if "decision_implications" not in result:
        result["decision_implications"] = []

    if "watchlist" not in result:
        result["watchlist"] = []

    if "top_stories" not in result:
        result["top_stories"] = []

    return result
    
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
            AND label != 'not relevant to field'
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
      AND label != 'not relevant to field'
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


# =========================
# Monitoring endpoints
# =========================

@app.get("/monitoring/results")
def get_monitoring_results(
    overall_status: Optional[str] = None,
    requires_human_review: Optional[int] = None,
    label_judgment: Optional[str] = None,
    predicted_label: Optional[str] = None,
    source: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    conn = get_connection()

    query = """
        SELECT
            monitoring_id,
            article_id,
            title,
            description,
            clean_text,
            predicted_label,
            source,
            url,
            published_at,
            classified_at,
            label_judgment,
            label_confidence,
            label_explanation,
            overall_status,
            requires_human_review,
            judge_model,
            raw_judge_response,
            evaluated_at
        FROM monitoring_results
        WHERE 1=1
    """
    params = []

    if overall_status:
        query += " AND overall_status = ?"
        params.append(overall_status)

    if requires_human_review is not None:
        query += " AND requires_human_review = ?"
        params.append(requires_human_review)

    if label_judgment:
        query += " AND label_judgment = ?"
        params.append(label_judgment)

    if predicted_label:
        query += " AND predicted_label = ?"
        params.append(predicted_label)

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

    query += " ORDER BY evaluated_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return df.to_dict(orient="records")


@app.get("/monitoring/summary")
def get_monitoring_summary():
    conn = get_connection()

    total_monitored = int(pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM monitoring_results",
        conn
    )["n"].iloc[0])

    needs_review = int(pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM monitoring_results WHERE requires_human_review = 1",
        conn
    )["n"].iloc[0])

    label_distribution = pd.read_sql_query(
        """
        SELECT label_judgment, COUNT(*) AS count
        FROM monitoring_results
        GROUP BY label_judgment
        ORDER BY count DESC
        """,
        conn
    ).to_dict(orient="records")

    status_distribution = pd.read_sql_query(
        """
        SELECT overall_status, COUNT(*) AS count
        FROM monitoring_results
        GROUP BY overall_status
        ORDER BY count DESC
        """,
        conn
    ).to_dict(orient="records")

    common_problem_labels = pd.read_sql_query(
        """
        SELECT predicted_label, COUNT(*) AS count
        FROM monitoring_results
        WHERE overall_status != 'ok'
        GROUP BY predicted_label
        ORDER BY count DESC
        """,
        conn
    ).to_dict(orient="records")

    daily_issues = pd.read_sql_query(
        """
        SELECT
            date(evaluated_at) AS day,
            overall_status,
            COUNT(*) AS count
        FROM monitoring_results
        GROUP BY date(evaluated_at), overall_status
        ORDER BY day ASC, overall_status ASC
        """,
        conn
    ).to_dict(orient="records")

    conn.close()

    return {
        "total_monitored": total_monitored,
        "needs_review": needs_review,
        "label_distribution": label_distribution,
        "status_distribution": status_distribution,
        "common_problem_labels": common_problem_labels,
        "daily_issues": daily_issues,
    }


@app.get("/monitoring/review-queue")
def get_review_queue(limit: int = Query(100, ge=1, le=500)):
    conn = get_connection()

    query = """
        SELECT
            monitoring_id,
            article_id,
            title,
            description,
            predicted_label,
            source,
            url,
            published_at,
            label_judgment,
            label_confidence,
            label_explanation,
            overall_status,
            requires_human_review,
            evaluated_at
        FROM monitoring_results
        WHERE requires_human_review = 1
        ORDER BY evaluated_at DESC
        LIMIT ?
    """

    df = pd.read_sql_query(query, conn, params=[limit])
    conn.close()

    return df.to_dict(orient="records")