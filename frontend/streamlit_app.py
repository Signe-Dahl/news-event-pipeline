# frontend/streamlit_app.py
import json

import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Green Energy News Event Dashboard",
    page_icon="📰",
    layout="wide",
)

API_BASE_URL = "https://Signe22-Article-Data-API.hf.space"


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

        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["classified_at"] = pd.to_datetime(df["classified_at"], errors="coerce", utc=True)

        df["published_date"] = df["published_at"].dt.date
        df["published_day"] = df["published_at"].dt.strftime("%Y-%m-%d")

        return df

    except Exception as e:
        st.error(f"Failed to load articles from API: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_daily_summary() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/summary/daily", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load daily summary: {e}")
        return {}


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    label_options = sorted(df["label"].dropna().unique().tolist()) if not df.empty else []
    source_options = sorted(df["source"].dropna().unique().tolist()) if not df.empty else []

    selected_labels = st.sidebar.multiselect(
        "Action categories",
        options=label_options,
        default=label_options,
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
        search_term = search_term.lower().strip()
        filtered = filtered[
            filtered["title"].fillna("").str.lower().str.contains(search_term, na=False)
            | filtered["description"].fillna("").str.lower().str.contains(search_term, na=False)
        ]

    return filtered


def find_matching_article(title: str, articles_df: pd.DataFrame):
    """
    Match a top-story title from the daily summary to the classified articles.
    This is used to fill missing fields such as label, source, URL, and published date.
    """
    if articles_df.empty or not title:
        return None

    normalized_title = title.strip().lower()

    matches = articles_df[
        articles_df["title"].fillna("").str.strip().str.lower() == normalized_title
    ]

    if not matches.empty:
        return matches.iloc[0]

    partial_matches = articles_df[
        articles_df["title"].fillna("").str.lower().str.contains(normalized_title, na=False)
    ]

    if not partial_matches.empty:
        return partial_matches.iloc[0]

    return None


def render_daily_summary(summary: dict, articles_df: pd.DataFrame) -> None:
    st.subheader("Daily AI Summary")

    if not summary:
        st.info("No daily summary available yet.")
        return

    st.caption(f"Summary date: {summary.get('summary_date', 'Unknown')}")

    st.markdown("### Executive Summary")
    st.write(summary.get("short_summary", "No summary available."))

    st.markdown("### Recommended Focus")
    st.write(summary.get("key_focus", "No focus available."))

    top_stories = summary.get("top_stories")

    if isinstance(top_stories, str):
        try:
            top_stories = json.loads(top_stories)
        except Exception:
            top_stories = None

    if not isinstance(top_stories, dict):
        return

    stories = top_stories.get("top_stories", [])

    if not stories:
        return

    st.markdown("### Top Stories")

    for story in stories:
        title = story.get("title", "Untitled story")
        matched_article = find_matching_article(title, articles_df)

        label = (
            story.get("label")
            or story.get("category")
            or (matched_article["label"] if matched_article is not None else "Unknown")
        )

        source = (
            story.get("source")
            or (matched_article["source"] if matched_article is not None else "Unknown source")
        )

        published_at = story.get("published_at")

        if not published_at and matched_article is not None:
            published_at = matched_article["published_at"]

        if pd.notnull(published_at):
            published_at = pd.to_datetime(published_at).strftime("%Y-%m-%d %H:%M UTC")
        else:
            published_at = "Unknown date"

        description = (
            story.get("description")
            or (matched_article["description"] if matched_article is not None else "")
        )

        reason = story.get("reason") or story.get("importance") or ""

        url = story.get("url") or (
            matched_article["url"] if matched_article is not None else None
        )

        with st.expander(title):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Category:** {label}")
            c2.markdown(f"**Source:** {source}")
            c3.markdown(f"**Published:** {published_at}")

            if description:
                st.markdown("**Description**")
                st.write(description)

            if reason:
                st.markdown("**Why this matters**")
                st.write(reason)

            if url:
                st.markdown(f"[Open article]({url})")


def render_article_browser(df: pd.DataFrame) -> None:
    st.subheader("Article browser")

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
        display_df = display_df.sort_values(["label", "published_at"], ascending=[True, False])
    elif sort_option == "Source":
        display_df = display_df.sort_values(["source", "published_at"], ascending=[True, False])

    max_rows = st.slider("Number of articles to display", 5, 100, 20)
    display_df = display_df.head(max_rows)

    for _, row in display_df.iterrows():
        published_str = (
            row["published_at"].strftime("%Y-%m-%d %H:%M UTC")
            if pd.notnull(row["published_at"])
            else "Unknown"
        )

        with st.expander(f"{row['title']}"):
            meta1, meta2, meta3 = st.columns(3)
            meta1.markdown(f"**Action:** {row['label']}")
            meta2.markdown(f"**Source:** {row['source']}")
            meta3.markdown(f"**Published:** {published_str}")

            if pd.notnull(row["description"]) and str(row["description"]).strip():
                st.markdown("**Description**")
                st.write(row["description"])

            if pd.notnull(row["url"]) and str(row["url"]).strip():
                st.markdown(f"[Open article]({row['url']})")

            with st.container():
                st.markdown("**More details**")
                st.caption(f"Article ID: {row['article_id']}")
                if pd.notnull(row["raw_label"]) and str(row["raw_label"]).strip():
                    st.caption(f"Model output: {row['raw_label']}")


def main() -> None:
    st.title("📰 Green Energy News Event Dashboard")
    st.write(
        "This dashboard gives an overview of classified green energy and climate-tech news, "
        "with filters for action categories, dates, sources, and search terms."
    )

    df = load_classified_articles()
    summary = load_daily_summary()

    if df.empty:
        st.warning("No classified articles found yet. Check whether the API is live and returning data.")
        return

    filtered_df = apply_filters(df)

    tab1, tab2 = st.tabs(["Daily Summary", "Articles"])

    with tab1:
        render_daily_summary(summary, df)

    with tab2:
        render_article_browser(filtered_df)


if __name__ == "__main__":
    main()