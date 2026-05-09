"""Upload view — load or upload a dataset and trigger analysis."""

import os
from datetime import datetime

import pandas as pd
import streamlit as st

import config as cfg
from utils.data_processing import clean_data, classify_sentiments, cluster_categories


# -- Helpers -------------------------------------------------------------------------------------

def _get_database_files() -> list[str]:
    """Return sorted list of saved CSV names (without extension)."""
    try:
        data_dir = os.path.join(os.getcwd(), "data")
        return sorted(f.rsplit(".", 1)[0] for f in os.listdir(data_dir) if f.endswith(".csv"))
    except OSError:
        return []


def _load_csv(filepath) -> pd.DataFrame:
    """Read a CSV from a path or file-like object."""
    return pd.read_csv(filepath)


def _reset_analysis() -> None:
    """Clear any previously computed analysis from session state."""
    st.session_state.df_clean = None
    st.session_state.classifications = None
    st.session_state.clustering = None


def run_analysis(df: pd.DataFrame) -> None:
    """
    Run cleaning, classification, and clustering on *df*.
    Saves all results to session state and switches view to 'results'.
    """
    df_clean = None
    class_results = None
    clustering = None

    st.title("📊 Analysing Dataset")
    progress_col, status_col = st.columns([3, 1])
    with progress_col:
        progress = st.progress(0, text="Initialising analysis...")
    with status_col:
        status = st.empty()

    # Clean
    status.info("🧹 Cleaning")
    progress.progress(15, text="Cleaning data...")
    df_clean = clean_data(df)
    if df_clean is None:
        st.error("Data cleaning failed. Ensure 'reviews.text' and 'reviews.rating' columns exist.")
        return

    # Classify
    status.info("🤖 Classifying")
    progress.progress(50, text="Running sentiment classification...")
    try:
        classifier = cfg.load_sentiment_classifier()
        class_results = classify_sentiments(df_clean, classifier)
        df_clean["predicted_sentiment"] = class_results["predicted"]
        df_clean["confidence_score"] = class_results["scores"]
    except Exception as exc:  # pylint: disable=broad-except
        st.warning(f"Classification skipped: {exc}")

    # Cluster
    status.info("🔗 Clustering")
    progress.progress(75, text="Clustering product categories...")
    if "categories" in df_clean.columns:
        try:
            model = cfg.load_sentence_transformer()
            clustering = cluster_categories(df_clean, model)
        except Exception as exc:  # pylint: disable=broad-except
            st.warning(f"Clustering skipped: {exc}")

    # Persist
    st.session_state.df_clean = df_clean
    st.session_state.classifications = class_results
    st.session_state.clustering = clustering
    st.session_state.view = "results"

    status.success("✅ Done!")
    progress.progress(100, text="Analysis complete!")
    st.success("✅ Data preprocessing complete!")
    st.balloons()
    st.rerun()


# -- View -------------------------------------------------------------------------------------

def render() -> None:
    """Render the home / upload view."""
    st.title("🏠 Home")
    st.markdown(
        """
Welcome to **Review Reviewer** — analyse customer reviews with AI-powered insights.

**Features:**
- 📊 Automatic sentiment classification (positive, neutral, negative)
- 🔗 Product category clustering
- 🤖 AI-powered summaries via Groq
        """
    )

    col_upload, col_db = st.columns(2)

    # -- Upload column --------------------------------------------------------------------
    with col_upload:
        st.subheader("📤 Upload New Data")
        data_file = st.file_uploader("Upload CSV file", type="csv")

        if data_file is None:
            st.session_state.data_loaded = False

        if data_file and not st.session_state.get("data_loaded"):
            st.session_state.df = _load_csv(data_file)
            _reset_analysis()
            st.session_state.data_loaded = True

        if data_file and st.session_state.df is not None:
            df = st.session_state.df

            c1, c2 = st.columns(2)
            c1.metric("Rows", len(df))
            c2.metric("Columns", len(df.columns))

            with st.expander("Preview data"):
                st.dataframe(df.head(10), use_container_width=True)

            cs1, cs2 = st.columns([2, 1])
            with cs1:
                custom_name = st.text_input(
                    "Save as:", value=f"data_{datetime.now().strftime('%Y%m%d')}"
                )
            with cs2:
                if st.button("Save to Database", type="primary"):
                    if not st.session_state.is_data_full:
                        dest = os.path.join(os.getcwd(), "data", f"{custom_name}.csv")
                        df.to_csv(dest, index=False)
                        st.success(f"✅ Saved as '{custom_name}'")
                    else:
                        st.error("Database full. Delete a file first.")

            if st.button("🚀 Analyse Data", type="primary", key="analyze_upload_btn"):
                run_analysis(st.session_state.df)

    # -- Database column --------------------------------------------------------------------
    with col_db:
        st.subheader("📁 Load from Database")
        db_files = _get_database_files()

        if not db_files:
            st.info("No saved datasets yet. Upload one to get started!")
        else:
            selected_file = st.selectbox("Select a dataset:", db_files, key="db_select")

            if st.button("📂 Load", type="primary", key="load_btn"):
                path = os.path.join(os.getcwd(), "data", f"{selected_file}.csv")
                st.session_state.df = _load_csv(path)
                _reset_analysis()
                st.success(f"✅ Loaded '{selected_file}' ({len(st.session_state.df):,} rows)")

            if st.session_state.df is not None:
                if st.button("🚀 Analyse Data", type="primary", key="analyze_db_btn"):
                    run_analysis(st.session_state.df)

            with st.expander("🗑️ Manage Database"):
                for file in db_files:
                    cf, cd = st.columns([3, 1])
                    cf.write(file)
                    with cd:
                        if st.button("❌", key=f"del_{file}"):
                            os.remove(os.path.join(os.getcwd(), "data", f"{file}.csv"))
                            st.rerun()