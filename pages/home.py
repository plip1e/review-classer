"""Home page module"""

import streamlit as st
import pandas as pd
import os
import config as cfg


def render_sidebar_database():
    """Render database management in sidebar"""
    if 'rerun_side' in st.session_state:
        if st.session_state.rerun_side:
            st.session_state.rerun_side = False
            st.rerun()

    if "items" not in st.session_state:
        db_f = [file_name.split('.')[0]
                for file_name in os.listdir(os.path.join(os.getcwd(), "data"))]
        st.session_state['db_files'] = db_f

    for i, item in enumerate(st.session_state.db_files):
        col1, col2 = st.columns([3.6, 1])
        with col1:
            st.write(item)
        with col2:
            if st.button("❌", key=f"del_{i}"):
                st.session_state.db_files.pop(i)
                os.remove(os.path.join(os.getcwd(), "data", f"{item}.csv"))
                st.rerun()


def render_upload_section():
    """Render file upload section"""
    
    # Initialize state
    if st.session_state.get("sample_df") is None:
        st.session_state["sample_df"] = None
    
    # CSV file uploader
    data_file = st.file_uploader(label="Upload data file", type="csv", max_upload_size=400)

    if data_file:
        st.session_state.df = pd.read_csv(data_file)
        st.session_state.col_cfg = {col: st.column_config.TextColumn(max_chars=30)
                                    for col in st.session_state.df}
    
    # Buttons row
    col1, col2, col3 = st.columns([2.9, 1, 1])
    st.session_state["err_mode"] = None

    # Sample button
    with col1:
        if st.button(label="Sample Dataframe", key="sample_btn"):
            st.session_state["sample_df"] = 1 if st.session_state.get("sample_df") is None else None
            st.rerun()

    # Custom filename
    with col2:
        custom_file_name = st.text_input("Custom Name",
                                         label_visibility="collapsed",
                                         value="Custom Name")
        if custom_file_name == "Custom Name":
            custom_file_name = cfg.available_loc()

    # Save button
    if "save_state" not in st.session_state:
        st.session_state["save_state"] = False
    
    with col3:
        if st.button(label="Save Dataframe"):
            if st.session_state.df is not None:
                if not st.session_state.is_data_full:
                    st.session_state.df.to_csv(os.path.join(os.getcwd(), "data", custom_file_name))
                    st.session_state.save_state = True
                    st.rerun()
                else:
                    st.session_state.err_mode = 1
            else:
                st.session_state.err_mode = 2
        
        if st.session_state.save_state is True:
            st.success("File Saved")
            st.session_state.save_state = False

    # Show sample
    if st.session_state.get("sample_df") == 1 and st.session_state.df is not None:
        st.dataframe(st.session_state.df.sample(min(6, len(st.session_state.df)), random_state=35),
                     column_config=st.session_state.col_cfg, hide_index=True)

    # Error messages
    if st.session_state.err_mode == 1:
        st.warning("Database is full. Delete a file if you want to save this file ")
    elif st.session_state.err_mode == 2:
        st.warning("Upload CSV file first")


def render():
    """Render home page"""
    st.title("🏠 Home Page")
    st.caption("Customer Review Analysis Platform")
    
    st.write("""
    Welcome to the Review Reviewer - an intelligent platform for analyzing customer reviews and market insights.
    
    **Features:**
    - 📊 **Classification**: Automatically categorize reviews into sentiment categories (positive, neutral, negative)
    - 🔗 **Clustering**: Group product categories into meaningful market segments
    - 🤖 **AI Summaries**: Generate intelligent product summaries using Groq LLM
    
    **Getting Started:**
    1. Upload your review dataset using the file uploader below
    2. Navigate to Classification, Clustering, or AI Summary pages
    3. Analyze insights for your marketing strategy
    """)
    
    st.subheader("📤 Upload Your Data")
    render_upload_section()
    
    if st.session_state.df is not None:
        st.success(f"✅ Dataset loaded: {len(st.session_state.df)} rows")
        with st.expander("Dataset Preview"):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
