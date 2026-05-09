"""Results view — sentiment classification and category clustering."""

import streamlit as st
from routes.views.upload import run_analysis


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nav_buttons() -> None:
    """Render Back and AI Summary navigation buttons."""
    col_title, col_summary, col_back = st.columns([5, 1, 1])
    with col_title:
        st.title("📊 Classification & Analysis")
    with col_summary:
        if st.button("🤖 AI Summary", key="go_summary_btn"):
            st.session_state.view = "summary"
            st.rerun()
    with col_back:
        if st.button("⬅ Back", key="back_btn"):
            st.session_state.view = "upload"
            st.rerun()


def _render_classification(df_clean) -> None:
    """Render sentiment metrics, charts, and sample reviews."""
    st.subheader("📈 Sentiment Classification Results")

    if "predicted_sentiment" not in df_clean.columns:
        st.info("No sentiment data available.")
        return

    sentiment_counts = df_clean["predicted_sentiment"].value_counts()
    total = len(df_clean)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", total)

    for col, label, emoji in zip(
        [col2, col3, col4],
        ["positive", "neutral", "negative"],
        ["😊", "😐", "😞"],
    ):
        count = sentiment_counts.get(label, 0)
        col.metric(f"{emoji} {label.capitalize()}", count, f"{count / total * 100:.1f}%")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Sentiment Distribution")
        st.bar_chart(sentiment_counts)
    with chart_col2:
        st.subheader("Confidence Scores")
        confidence_data = (
            df_clean.groupby("predicted_sentiment")["confidence_score"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(confidence_data)

    st.subheader("📝 Sample Reviews by Sentiment")
    sentiment_choice = st.selectbox(
        "View samples for:", ["positive", "neutral", "negative"], key="sentiment_select"
    )
    samples = df_clean[df_clean["predicted_sentiment"] == sentiment_choice].head(5)
    emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😞"}

    if samples.empty:
        st.info(f"No {sentiment_choice} reviews found in dataset.")
    else:
        for _, row in samples.iterrows():
            emoji = emoji_map.get(sentiment_choice, "")
            title = str(row.get("reviews.title", "Review"))[:50]
            with st.expander(f"{emoji} {title}... ({row['confidence_score']:.0%} confidence)"):
                st.write(f"**Review:** {str(row['reviews.text'])[:500]}...")
                if "reviews.rating" in row:
                    st.write(
                        f"**Rating:** {row['reviews.rating']}/5 | "
                        f"**Sentiment Score:** {row['confidence_score']:.1%}"
                    )


def _render_clustering(df, df_clean) -> None:
    """Render category clustering metrics and explorer."""
    clustering_result = st.session_state.clustering

    if not clustering_result or "categories" not in df_clean.columns:
        st.divider()
        st.info("ℹ️ No 'categories' column found in dataset. Clustering analysis not available.")
        return

    st.divider()
    st.subheader("🔗 Category Clustering")
    st.metric("Unique Categories Identified", len(clustering_result["categories"]))

    groups = {"E-Readers": [], "Tablets": [], "Smart Devices": [], "Other": []}
    for cat in clustering_result["categories"]:
        cat_lower = str(cat).lower()
        if "kindle" in cat_lower or "e-reader" in cat_lower:
            groups["E-Readers"].append(cat)
        elif "tablet" in cat_lower or "fire" in cat_lower:
            groups["Tablets"].append(cat)
        elif any(x in cat_lower for x in ["alexa", "echo", "tv", "assistant"]):
            groups["Smart Devices"].append(cat)
        else:
            groups["Other"].append(cat)

    col1, col2, col3 = st.columns(3)
    col1.metric("📖 E-Readers", len(groups["E-Readers"]))
    col2.metric("📱 Tablets", len(groups["Tablets"]))
    col3.metric("🤖 Smart Devices", len(groups["Smart Devices"]))

    with st.expander("📂 View All Category Groups"):
        for group_name, categories in groups.items():
            if categories:
                st.write(f"**{group_name}** ({len(categories)} categories)")
                for cat in categories:
                    st.caption(f"• {cat}")

    st.subheader("🔍 Explore by Category")
    selected_cat = st.selectbox(
        "Select a category to explore:", clustering_result["categories"], key="cat_select"
    )

    if selected_cat:
        cat_reviews = df[df.get("categories", "") == selected_cat]
        if cat_reviews.empty:
            st.info("No reviews found for this category.")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Reviews in Category", len(cat_reviews))
        if "reviews.rating" in cat_reviews.columns:
            c2.metric("Avg Rating", f"{cat_reviews['reviews.rating'].mean():.2f}/5")
        if "predicted_sentiment" in cat_reviews.columns:
            pos_pct = (cat_reviews["predicted_sentiment"] == "positive").sum() / len(cat_reviews) * 100
            c3.metric("Positive %", f"{pos_pct:.0f}%")

        with st.expander("📋 Sample Reviews from This Category"):
            for _, row in cat_reviews.head(3).iterrows():
                st.write(f"**{row.get('reviews.title', 'Review')}**")
                st.caption(f"{str(row['reviews.text'])[:300]}...")
                if "reviews.rating" in row:
                    st.caption(f"Rating: {row['reviews.rating']}/5")


# ── View ───────────────────────────────────────────────────────────────────────

def render() -> None:
    """Render the results view."""
    df = st.session_state.df
    df_clean = st.session_state.df_clean

    _nav_buttons()

    if df_clean is None:
        st.warning("⏳ No analysis results yet.")
        if st.button("🚀 Analyse Now", type="primary", key="analyze_now_btn"):
            run_analysis(df)
        return

    _render_classification(df_clean)
    _render_clustering(df, df_clean)