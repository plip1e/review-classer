"""Review Reviewer — main entry point"""

import os
import base64
import streamlit as st
from dotenv import load_dotenv

import config as cfg
from routes.home import render as render_home

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Review Reviewer", page_icon="💬")

# ── Session state defaults ─────────────────────────────────────────────────────

_DEFAULTS = {
    "df": None,
    "df_clean": None,
    "classifications": None,
    "clustering": None,
    "data_loaded": False,
    "view": "upload",
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

if "is_data_full" not in st.session_state:
    st.session_state.is_data_full = False
    try:
        data_dir = os.path.join(os.getcwd(), "data")
        csv_count = len([f for f in os.listdir(data_dir) if f.endswith(".csv")])
        st.session_state.is_data_full = csv_count > cfg.MAX_SAVED_CSV
    except OSError:
        pass

# ── Helpers ────────────────────────────────────────────────────────────────────

def _image_link(image_path: str, url: str, width: int = 25) -> None:
    """Render a clickable image in the sidebar."""
    try:
        with open(image_path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode()
        ext = image_path.rsplit(".", 1)[-1]
        st.markdown(
            f'<a href="{url}" target="_blank">'
            f'<img src="data:image/{ext};base64,{data}" width="{width}"></a>',
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.caption(f"[Profile]({url})")

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛈 About")
    st.divider()
    st.markdown(
        """
**Review Reviewer** is an intelligent platform for analysing customer reviews.

### Features
1. **Classification** — sentiment analysis (positive, neutral, negative)
2. **Clustering** — group product categories intelligently
3. **AI Summary** — generate insights via Groq LLM

### How It Works
1. Upload a CSV with review data
2. System automatically analyses sentiment & categories
3. Generate AI summaries for marketing insights
        """
    )
    st.divider()
    st.title("📞 Contact")
    col1, col2 = st.columns([1, 8])
    with col1:
        _image_link("static/github-icon.png", "https://github.com/plip1e")
    with col2:
        _image_link("static/linkedin-icon.png", "https://linkedin.com/in/paul-du-preez-67ab93302")
    st.caption("by Paul Du Preez")

# ── Main ───────────────────────────────────────────────────────────────────────

render_home()

st.divider()
st.caption("Marketing insights platform for analysing customer reviews and market trends.")