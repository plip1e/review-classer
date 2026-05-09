"""Home router — delegates to the correct view based on session state."""

import streamlit as st
from routes.views.upload import render as render_upload
from routes.views.results import render as render_results
from routes.views.summary import render as render_summary


def render() -> None:
    """Switch between upload, results, and summary views."""
    view = st.session_state.get("view", "upload")

    if view == "summary":
        render_summary()
    elif view == "results" and st.session_state.df_clean is not None:
        render_results()
    else:
        render_upload()