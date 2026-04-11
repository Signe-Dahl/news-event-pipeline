# streamlit_app.py
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Green Energy News Event Dashboard",
    page_icon="📰",
    layout="wide",
)

DB_PATH = Path("/app/data/news.db")

@st.cache_data(ttl=300)
def load_classified_articles() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
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
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["classified_at"] = pd.to_datetime(df["classified_at"], errors="coerce", utc=True)

    df["published_date"] = df["published_at"].dt.date
    df["published_day"] = df["published_at"].dt.strftime("%Y-%m-%d")

    return df


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


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Articles shown", len(df))
    c2.metric("Unique action categories", df["label"].nunique() if not df.empty else 0)
    c3.metric("Unique sources", df["source"].nunique() if not df.empty else 0)

    if df.empty:
        st.info("No classified articles match the current filters.")
        return

    daily_counts = (
        df.groupby(["published_day", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["published_day", "label"])
    )

    st.markdown("#### Actions by day")
    chart_df = (
        daily_counts.pivot(index="published_day", columns="label", values="count")
        .fillna(0)
        .sort_index()
    )
    st.line_chart(chart_df)

    st.markdown("#### Category distribution")
    dist_df = df["label"].value_counts().rename_axis("label").reset_index(name="count")
    st.bar_chart(dist_df.set_index("label"))


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
        published_str = row["published_at"].strftime("%Y-%m-%d %H:%M UTC") if pd.notnull(row["published_at"]) else "Unknown"

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

    if df.empty:
        st.warning("No classified articles found yet. Run the pipeline first.")
        return

    filtered_df = apply_filters(df)

    tab1, tab2 = st.tabs(["Overview", "Articles"])

    with tab1:
        render_overview(filtered_df)

    with tab2:
        render_article_browser(filtered_df)

if __name__ == "__main__":
    main()