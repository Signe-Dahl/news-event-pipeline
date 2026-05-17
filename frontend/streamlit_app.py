# frontend/streamlit_app.py
import json
import os

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Green Energy News Event Dashboard",
    page_icon="📰",
    layout="wide",
)

API_BASE_URL = "https://Signe22-Article-Data-API.hf.space"

def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for column in columns:
        if column not in df.columns:
            df[column] = None

    return df

@st.cache_data(ttl=300)
def load_classified_articles() -> pd.DataFrame:
    try:
        response = requests.get(
            f"{API_BASE_URL}/articles",
            params={"limit": 500},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data)

        if df.empty:
            return df

        df = ensure_columns(
            df,
            [
                "article_id",
                "title",
                "description",
                "source",
                "label",
                "raw_label",
                "url",
                "published_at",
                "classified_at",
            ],
        )

        df["published_at"] = pd.to_datetime(
            df["published_at"],
            errors="coerce",
            utc=True,
        )
        df["classified_at"] = pd.to_datetime(
            df["classified_at"],
            errors="coerce",
            utc=True,
        )

        df["published_date"] = df["published_at"].dt.date
        df["published_day"] = df["published_at"].dt.strftime("%Y-%m-%d")

        return df

    except Exception as error:
        st.error(f"Failed to load articles from API: {error}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_daily_summary() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/summary/daily", timeout=30)
        response.raise_for_status()
        summary = response.json()

        if not isinstance(summary, dict):
            return {}

        return normalize_summary_payload(summary)

    except Exception as error:
        st.error(f"Failed to load daily summary: {error}")
        return {}


def normalize_summary_payload(summary: dict) -> dict:
    normalized = dict(summary)

    nested_summary = summary.get("summary_json")

    if isinstance(nested_summary, str):
        try:
            parsed = json.loads(nested_summary)
            if isinstance(parsed, dict):
                normalized.update(parsed)
        except Exception:
            pass

    elif isinstance(nested_summary, dict):
        normalized.update(nested_summary)

    normalized["executive_summary"] = (
        normalized.get("executive_summary")
        or normalized.get("short_summary")
        or ""
    )

    normalized["recommended_focus"] = (
        normalized.get("recommended_focus")
        or normalized.get("key_focus")
        or ""
    )

    if not isinstance(normalized.get("decision_implications"), list):
        normalized["decision_implications"] = []

    if not isinstance(normalized.get("watchlist"), list):
        normalized["watchlist"] = []

    if not isinstance(normalized.get("top_stories"), list):
        normalized["top_stories"] = []

    return normalized

def is_valid_url(value: object) -> bool:
    if not isinstance(value, str):
        return False

    return value.startswith(("http://", "https://"))

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    label_options = sorted(df["label"].dropna().unique().tolist())
    source_options = sorted(df["source"].dropna().unique().tolist())

    default_labels = [
        label
        for label in label_options
        if label != "not relevant to field"
    ]

    selected_labels = st.sidebar.multiselect(
        "Action categories",
        options=label_options,
        default=default_labels,
    )

    selected_sources = st.sidebar.multiselect(
        "Sources",
        options=source_options,
        default=[],
    )

    min_date = df["published_date"].min() if not df.empty else None
    max_date = df["published_date"].max() if not df.empty else None

    date_range = None
    if min_date and max_date:
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    search_term = st.sidebar.text_input("Search title or description")

    filtered = df.copy()

    if selected_labels:
        filtered = filtered[filtered["label"].isin(selected_labels)]

    if selected_sources:
        filtered = filtered[filtered["source"].isin(selected_sources)]

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["published_date"] >= start_date)
            & (filtered["published_date"] <= end_date)
        ]

    if search_term:
        search_term = search_term.strip()

        title_matches = filtered["title"].fillna("").str.contains(
            search_term,
            case=False,
            na=False,
            regex=False,
        )

        description_matches = filtered["description"].fillna("").str.contains(
            search_term,
            case=False,
            na=False,
            regex=False,
        )

        filtered = filtered[title_matches | description_matches]

    return filtered


def render_metrics(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Articles", len(df))
    c2.metric("Shown", len(filtered_df))
    c3.metric("Sources", df["source"].nunique())
    c4.metric("Categories", df["label"].nunique())


def render_bullet_list(items: list[str], empty_message: str) -> None:
    if not items:
        st.info(empty_message)
        return

    for item in items:
        st.markdown(f"- {item}")

def get_summary_source_articles(
    df: pd.DataFrame,
    summary: dict,
    fallback_limit: int = 15,
) -> pd.DataFrame:
    stories = summary.get("top_stories", [])

    story_ids = {
        str(story.get("article_id"))
        for story in stories
        if isinstance(story, dict) and story.get("article_id")
    }

    if story_ids:
        matched_df = df[df["article_id"].astype(str).isin(story_ids)]

        if not matched_df.empty:
            return matched_df

    relevant_df = df[df["label"] != "not relevant to field"].copy()

    if "published_at" in relevant_df:
        relevant_df = relevant_df.sort_values("published_at", ascending=False)

    return relevant_df.head(fallback_limit)


def render_daily_summary_source_basis(
    df: pd.DataFrame,
    summary: dict,
) -> pd.DataFrame:
    summary_df = get_summary_source_articles(df, summary)

    generated_at = summary.get("generated_at")
    top_story_count = len(summary.get("top_stories", []))

    formatted_generated_at = None

    if generated_at:
        parsed = pd.to_datetime(generated_at, errors="coerce", utc=True)

        if pd.notnull(parsed):
            formatted_generated_at = parsed.strftime("%Y-%m-%d %H:%M UTC")

    if formatted_generated_at:
        st.caption(
            f"Summary generated {formatted_generated_at} · "
            f"{top_story_count} top stories included"
        )
    else:
        st.caption(
            f"{top_story_count} top stories included in this summary"
        )

    return summary_df


def render_daily_summary(summary: dict) -> None:
    st.subheader("Daily AI Summary")

    if not summary:
        st.info("No daily summary available yet.")
        return

    summary_date = summary.get("summary_date", "Unknown")
    generated_at = summary.get("generated_at")

    if generated_at:
        st.caption(f"Summary date: {summary_date} · Generated at: {generated_at}")
    else:
        st.caption(f"Summary date: {summary_date}")

    st.markdown("### Executive Summary")
    st.write(summary.get("executive_summary") or "No summary available.")

    st.markdown("### Key Signal")
    st.write(summary.get("key_signal") or "No key signal available.")

    st.markdown("### Recommended Focus")
    st.write(summary.get("recommended_focus") or "No focus available.")

    st.markdown("### Decision Implications")
    render_bullet_list(
        summary.get("decision_implications", []),
        "No decision implications available.",
    )

    st.markdown("### Watchlist")
    render_bullet_list(
        summary.get("watchlist", []),
        "No watchlist available.",
    )

    stories = summary.get("top_stories", [])

    if not stories:
        st.info("No top stories available.")
        return

    st.markdown("### Top Stories")

    for story in stories:
        if not isinstance(story, dict):
            continue

        title = story.get("title", "Untitled story")
        label = story.get("label", "Unknown")
        source = story.get("source", "Unknown source")
        published_at = story.get("published_at")
        description = story.get("description", "")
        why_it_matters = story.get("why_it_matters", "")
        decision_relevance = story.get("decision_relevance", "")
        url = story.get("url")
        article_id = story.get("article_id")

        if pd.notnull(published_at):
            published_at = pd.to_datetime(
                published_at,
                errors="coerce",
                utc=True,
            )

            if pd.notnull(published_at):
                published_at = published_at.strftime("%Y-%m-%d %H:%M UTC")
            else:
                published_at = "Unknown date"
        else:
            published_at = "Unknown date"

        with st.expander(title):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Category:** {label}")
            c2.markdown(f"**Source:** {source}")
            c3.markdown(f"**Published:** {published_at}")

            if description:
                st.markdown("**Description**")
                st.write(description)

            if why_it_matters:
                st.markdown("**Why this matters**")
                st.write(why_it_matters)

            if decision_relevance:
                st.markdown("**Decision relevance**")
                st.write(decision_relevance)

            if is_valid_url(url):
                st.link_button("Open article", url)

            if article_id:
                st.caption(f"Article ID: {article_id}")


def render_article_browser(df: pd.DataFrame) -> None:
    st.subheader("Article Browser")

    if df.empty:
        st.info("No articles available for browsing.")
        return

    sort_option = st.selectbox(
        "Sort articles by",
        options=[
            "Newest first",
            "Oldest first",
            "Action category",
            "Source",
        ],
        index=0,
    )

    display_df = df.copy()

    if sort_option == "Newest first":
        display_df = display_df.sort_values("published_at", ascending=False)
    elif sort_option == "Oldest first":
        display_df = display_df.sort_values("published_at", ascending=True)
    elif sort_option == "Action category":
        display_df = display_df.sort_values(
            ["label", "published_at"],
            ascending=[True, False],
        )
    elif sort_option == "Source":
        display_df = display_df.sort_values(
            ["source", "published_at"],
            ascending=[True, False],
        )

    max_rows = st.slider("Number of articles to display", 5, 100, 20)
    display_df = display_df.head(max_rows)

    for _, row in display_df.iterrows():
        title = row.get("title", "Untitled article")

        published_str = (
            row["published_at"].strftime("%Y-%m-%d %H:%M UTC")
            if pd.notnull(row.get("published_at"))
            else "Unknown"
        )

        with st.expander(title):
            meta1, meta2, meta3 = st.columns(3)
            meta1.markdown(f"**Action:** {row.get('label', 'Unknown')}")
            meta2.markdown(f"**Source:** {row.get('source', 'Unknown source')}")
            meta3.markdown(f"**Published:** {published_str}")

            description = row.get("description")
            if pd.notnull(description) and str(description).strip():
                st.markdown("**Description**")
                st.write(description)

            url = row.get("url")
            if is_valid_url(url):
                st.link_button("Open article", url)

            st.markdown("**More details**")

            article_id = row.get("article_id")
            if pd.notnull(article_id):
                st.caption(f"Article ID: {article_id}")

            raw_label = row.get("raw_label")
            if pd.notnull(raw_label) and str(raw_label).strip():
                st.caption(f"Model output: {raw_label}")


def main() -> None:
    st.title("📰 Green Energy News Event Dashboard")
    st.write(
        "This dashboard gives an overview of classified green energy and climate-tech news, "
        "with filters for action categories, dates, sources, and search terms."
    )

    st.error(
        """
        ⚠️ **Important Notice**
        This dashboard uses Large Language Models (LLMs) to classify and summarize news articles.
        The outputs are probabilistic and may contain errors, biases, or misinterpretations.
        The system should be used as a decision-support tool rather than an authoritative source.
        Users are encouraged to critically evaluate the presented summaries, classifications,
        and highlighted stories alongside the underlying article content.
        """
    )

    df = load_classified_articles()
    summary = load_daily_summary()

    if df.empty:
        st.warning(
            "No classified articles found yet. "
            "Check whether the API is live and returning data."
        )
        return

    section = st.segmented_control(
        "View",
        options=["Daily Summary", "Articles"],
        default="Daily Summary",
    )

    if section == "Daily Summary":
        summary_df = render_daily_summary_source_basis(df, summary)
        render_metrics(df, summary_df)
        render_daily_summary(summary)

    elif section == "Articles":
        filtered_df = apply_filters(df)
        render_metrics(df, filtered_df)
        render_article_browser(filtered_df)


if __name__ == "__main__":
    main()