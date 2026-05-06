"""Main entry point for the Review Reviewer application"""

import streamlit as st
import os
import config as cfg
from dotenv import load_dotenv
load_dotenv()

# Import pages
from pages import home, classification, clustering, summary

# =======================================
# Page Configuration
# =======================================

st.set_page_config(page_title="Review reviewer", page_icon="💬", layout="wide")

# =======================================
# Session State Initialization
# =======================================

if "df" not in st.session_state:
    st.session_state['df'] = None
    st.session_state["col_cfg"] = None

if "is_data_full" not in st.session_state:
    st.session_state["is_data_full"] = False
    if len([file_name.split('.')[0]
            for file_name in os.listdir(os.path.join(os.getcwd(), "data"))]) > cfg.MAX_SAVED_CSV:
        st.session_state.is_data_full = True

# # =======================================
# # Sidebar Navigation
# # =======================================

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select a page:", 
                    ["Home", "Classification", "Clustering", "AI Summary"])
    
    st.divider()
    
    # Database management
    with st.expander(label="📁 Database"):
        home.render_sidebar_database()

# # =======================================
# # Page Routing
# # =======================================

if page == "Home":
    home.render()
elif page == "Classification":
    classification.render()
elif page == "Clustering":
    clustering.render()
elif page == "AI Summary":
    summary.render()

# =======================================
# Footer
# =======================================

st.divider()
st.markdown("""
---
*Create a website for the marketing department in your company, who needs to gain insights on how well the products are received by customers (from reviews) and what other competitive products exist in the market. For example, users in your webpage can choose between product categories and be shown statistics insights (distribution of ratings, best product ratings, etc), and text summarization for that specific category (which are the best product in this category, etc).*
""")
