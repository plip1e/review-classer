"""Summary view — AI-powered product summary via Groq."""

from datetime import datetime

import pandas as pd
import streamlit as st

import config as cfg
from utils.api import call_groq_api, create_groq_prompt, prepare_groq_context
from utils.data_processing import classify_sentiments


# ── Helpers ────────────────────────────────────────────────────────────────────

def _filter_by_product(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    """Return rows matching *product_name* across common name columns, or first 100 rows."""
    if not product_name:
        return df.head(100)
    for col in ("name", "product", "product_name"):
        if col in df.columns:
            filtered = df[df[col].astype(str).str.contains(product_name, case=False, na=False)]
            if not filtered.empty:
                return filtered
    return df.head(100)


def _ensure_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add predicted_sentiment column if not already present."""
    if "predicted_sentiment" in df.columns:
        return df
    with st.spinner("Running sentiment analysis..."):
        classifier = cfg.load_sentiment_classifier()
        results = classify_sentiments(df, classifier)
        df = df.copy()
        df["predicted_sentiment"] = results["predicted"]
    return df


# -- View -------------------------------------------------------------------------------------

def render() -> None:
    """Render the AI summary view."""
    col_title, col_back = st.columns([6, 1])
    with col_title:
        st.title("🤖 AI Product Summary (Groq)")
        st.caption("Generate intelligent summaries based on customer reviews and market data.")
    with col_back:
        if st.button("⬅ Back", key="summary_back_btn"):
            st.session_state.view = "results"
            st.rerun()

    groq_client = cfg.get_groq_client()
    if groq_client is None:
        st.error("⚠️ Groq API key not found. Set the GROQ_API_KEY environment variable.")
        st.info("Example: `export GROQ_API_KEY='your-key-here'`")
        return

    if st.session_state.df is None:
        st.warning("⚠️ Please upload or load a dataset first.")
        return

    df = st.session_state.df.copy()

    # -- Configuration --------------------------------------------------------------------
    st.subheader("Configure Summary Parameters")

    col1, col2 = st.columns(2)
    with col1:
        product_name = st.text_input(
            "Product Name (optional):", placeholder="Leave blank for dataset overview"
        )
    with col2:
        summary_type = st.selectbox(
            "Summary Type",
            ["Overview", "Strengths & Weaknesses", "Market Positioning", "Recommendations"],
        )

    st.subheader("Optional Filters")
    f1, f2, f3 = st.columns(3)
    include_price = f1.checkbox("Include Price Information", value=False)
    include_time_range = f2.checkbox("Include Time Period Analysis", value=False)
    include_sentiment = f3.checkbox("Include Sentiment Analysis", value=True)

    date_range = None
    date_col = None
    if include_time_range:
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            try:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                date_col = date_cols[0]
                date_range = st.slider(
                    "Select date range:",
                    value=(df[date_col].min(), df[date_col].max()),
                    min_value=df[date_col].min(),
                    max_value=df[date_col].max(),
                )
            except (ValueError, TypeError):
                st.warning("Could not parse date column.")

    # -- Generate --------------------------------------------------------------------
    if st.button("🚀 Generate Summary", type="primary"):
        with st.spinner("Generating summary with Groq..."):
            try:
                df_filtered = _filter_by_product(df, product_name)
                if df_filtered.empty:
                    df_filtered = df.head(100)

                # Apply date range filter if selected
                if include_time_range and date_range and date_col:
                    df_filtered = df_filtered[
                        (df_filtered[date_col] >= date_range[0]) &
                        (df_filtered[date_col] <= date_range[1])
                    ]

                if include_sentiment:
                    df_filtered = _ensure_sentiment(df_filtered)

                context_string = prepare_groq_context(df_filtered, include_sentiment, include_price)
                prompt = create_groq_prompt(context_string, summary_type, product_name)
                summary_text = call_groq_api(groq_client, prompt)

                st.success("✅ Summary Generated!")
                st.subheader(f"📝 {summary_type} Summary")
                st.write(summary_text)

                st.download_button(
                    label="📥 Download Summary",
                    data=summary_text,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Error generating summary: {exc}")