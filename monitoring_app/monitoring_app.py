import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="Monitoring Dashboard",
    page_icon="🛠️",
    layout="wide",
)

API_BASE_URL = "https://Signe22-Article-Data-API.hf.space"


@st.cache_data(ttl=300)
def load_monitoring_results() -> pd.DataFrame:
    response = requests.get(
        f"{API_BASE_URL}/monitoring/results",
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
    df["evaluated_at"] = pd.to_datetime(df["evaluated_at"], errors="coerce", utc=True)

    df["published_date"] = df["published_at"].dt.date
    df["evaluated_date"] = df["evaluated_at"].dt.date

    return df


@st.cache_data(ttl=300)
def load_monitoring_summary() -> dict:
    response = requests.get(f"{API_BASE_URL}/monitoring/summary", timeout=30)
    response.raise_for_status()
    return response.json()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Monitoring Filters")

    status_options = sorted(df["overall_status"].dropna().unique().tolist()) if not df.empty else []
    relevance_options = sorted(df["relevance_judgment"].dropna().unique().tolist()) if not df.empty else []
    label_judgment_options = sorted(df["label_judgment"].dropna().unique().tolist()) if not df.empty else []
    predicted_label_options = sorted(df["predicted_label"].dropna().unique().tolist()) if not df.empty else []
    source_options = sorted(df["source"].dropna().unique().tolist()) if not df.empty else []

    selected_status = st.sidebar.multiselect("Overall status", status_options, default=status_options)
    selected_relevance = st.sidebar.multiselect("Relevance judgment", relevance_options, default=relevance_options)
    selected_label_judgment = st.sidebar.multiselect("Label judgment", label_judgment_options, default=label_judgment_options)
    selected_predicted_labels = st.sidebar.multiselect("Predicted labels", predicted_label_options, default=[])
    selected_sources = st.sidebar.multiselect("Sources", source_options, default=[])

    review_only = st.sidebar.checkbox("Only show articles needing review", value=False)

    min_date = df["published_date"].min() if not df.empty else None
    max_date = df["published_date"].max() if not df.empty else None

    date_range = None
    if min_date and max_date:
        date_range = st.sidebar.date_input(
            "Published date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    search_term = st.sidebar.text_input("Search title or description")

    filtered = df.copy()

    if selected_status:
        filtered = filtered[filtered["overall_status"].isin(selected_status)]

    if selected_relevance:
        filtered = filtered[filtered["relevance_judgment"].isin(selected_relevance)]

    if selected_label_judgment:
        filtered = filtered[filtered["label_judgment"].isin(selected_label_judgment)]

    if selected_predicted_labels:
        filtered = filtered[filtered["predicted_label"].isin(selected_predicted_labels)]

    if selected_sources:
        filtered = filtered[filtered["source"].isin(selected_sources)]

    if review_only:
        filtered = filtered[filtered["requires_human_review"] == 1]

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


def render_summary(summary: dict, df: pd.DataFrame) -> None:
    st.subheader("Monitoring Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total monitored", summary.get("total_monitored", 0))
    c2.metric("Needs review", summary.get("needs_review", 0))
    c3.metric("Shown after filters", len(df))
    c4.metric("Problem rate", f"{(len(df[df['overall_status'] != 'ok']) / len(df) * 100):.1f}%" if len(df) else "0.0%")

    if df.empty:
        st.info("No monitoring results match the current filters.")
        return

    st.markdown("#### Monitoring status distribution")
    status_df = df["overall_status"].value_counts().rename_axis("overall_status").reset_index(name="count")
    st.bar_chart(status_df.set_index("overall_status"))

    st.markdown("#### Relevance judgment distribution")
    rel_df = df["relevance_judgment"].value_counts().rename_axis("relevance_judgment").reset_index(name="count")
    st.bar_chart(rel_df.set_index("relevance_judgment"))

    st.markdown("#### Label judgment distribution")
    label_df = df["label_judgment"].value_counts().rename_axis("label_judgment").reset_index(name="count")
    st.bar_chart(label_df.set_index("label_judgment"))

def render_problem_patterns(df: pd.DataFrame) -> None:
    st.subheader("Problem Patterns")

    if df.empty:
        st.info("No data available.")
        return

    issues = df[df["overall_status"] != "ok"]

    if issues.empty:
        st.success("No current problem cases in the filtered selection.")
        return

    st.markdown("#### Most problematic predicted labels")
    bad_labels = (
        issues["predicted_label"]
        .value_counts()
        .rename_axis("predicted_label")
        .reset_index(name="count")
    )
    st.dataframe(bad_labels, use_container_width=True, hide_index=True)

    st.markdown("#### Most problematic sources")
    bad_sources = (
        issues["source"]
        .value_counts()
        .rename_axis("source")
        .reset_index(name="count")
    )
    st.dataframe(bad_sources, use_container_width=True, hide_index=True)


def render_review_queue(df: pd.DataFrame) -> None:
    st.subheader("Review Queue")

    if df.empty:
        st.info("No monitoring results available.")
        return

    queue_df = df[df["requires_human_review"] == 1].copy()

    if queue_df.empty:
        st.success("No articles currently flagged for review in the filtered selection.")
        return

    max_rows = st.slider("Number of review cases to display", 5, 100, 20)
    queue_df = queue_df.sort_values("evaluated_at", ascending=False).head(max_rows)

    for _, row in queue_df.iterrows():
        published_str = row["published_at"].strftime("%Y-%m-%d %H:%M UTC") if pd.notnull(row["published_at"]) else "Unknown"
        evaluated_str = row["evaluated_at"].strftime("%Y-%m-%d %H:%M UTC") if pd.notnull(row["evaluated_at"]) else "Unknown"

        with st.expander(f"{row['title']}"):
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"**Predicted label:** {row['predicted_label']}")
            m2.markdown(f"**Overall status:** {row['overall_status']}")
            m3.markdown(f"**Source:** {row['source']}")
            m4.markdown(f"**Published:** {published_str}")

            st.markdown("**Description**")
            st.write(row["description"] if pd.notnull(row["description"]) else "No description")

            st.markdown("**Judge output**")
            c1, c2 = st.columns(2)
            c1.markdown(f"**Relevance:** {row['relevance_judgment']} ({row['relevance_confidence']})")
            c1.write(row["relevance_explanation"])

            c2.markdown(f"**Label quality:** {row['label_judgment']} ({row['label_confidence']})")
            c2.write(row["label_explanation"])

            st.markdown("**Metadata**")
            st.caption(f"Article ID: {row['article_id']}")
            st.caption(f"Evaluated at: {evaluated_str}")

            if pd.notnull(row["url"]) and str(row["url"]).strip():
                st.markdown(f"[Open article]({row['url']})")


def render_full_table(df: pd.DataFrame) -> None:
    st.subheader("Monitoring Table")

    if df.empty:
        st.info("No rows to display.")
        return

    table_df = df[
        [
            "published_at",
            "source",
            "predicted_label",
            "relevance_judgment",
            "relevance_confidence",
            "label_judgment",
            "label_confidence",
            "overall_status",
            "requires_human_review",
            "title",
        ]
    ].copy()

    table_df["published_at"] = table_df["published_at"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(table_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("🛠️ Monitoring Dashboard")
    st.write(
        "This dashboard helps inspect LLM-as-a-judge monitoring output in order to identify "
        "collection issues, classification issues, and low-confidence cases that may require "
        "pipeline improvements."
    )

    try:
        summary = load_monitoring_summary()
        df = load_monitoring_results()
    except Exception as e:
        st.error(f"Failed to load monitoring data from API: {e}")
        return

    if df.empty:
        st.warning("No monitoring results found yet.")
        return

    filtered_df = apply_filters(df)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Problem Patterns", "Review Queue", "Table"]
    )

    with tab1:
        render_summary(summary, filtered_df)

    with tab2:
        render_problem_patterns(filtered_df)

    with tab3:
        render_review_queue(filtered_df)

    with tab4:
        render_full_table(filtered_df)


if __name__ == "__main__":
    main()