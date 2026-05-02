'''App file for web development of the project'''

import streamlit as st
import pandas as pd
import config as cfg
import os

st.set_page_config(page_title="Review reviewer", page_icon="💬", )

## =======================================
## Fragments for uploading files
## =======================================
st.session_state['df'] = None
@st.fragment()
def upload_files_main():
    '''
    Main fragment for upload_files expander
    '''
    # CSV file uploader and reader
    data_file = st.file_uploader(label="Upload data file", type="csv")

    if data_file:
        st.session_state.df = pd.read_csv(data_file)
# Truncate dataframe for display
if st.session_state.df is not None:
    st.session_state["col_cfg"] = {col: st.column_config.TextColumn(max_chars=30)
                                   for col in st.session_state.df}

st.session_state["sample_df"] = None
@st.fragment()
def upload_files_btns(): # pylint: disable=too-many-branches
    '''
    fragment for upload_files functional buttons
    '''
    # Initialize columns for sample dataframe and save button
    col1, col2 = st.columns([3.8, 1])
    st.session_state["err_mode"] = None

    # Column 1: Sample dataframe button and display
    if st.session_state.sample_df is None:
        with col1:
            if st.button(label="Sample Dataframe", key="sample_btn"):
                st.session_state["sample_df"] = 1
                st.rerun(scope="fragment")

    elif st.session_state.sample_df == 1:
        if st.session_state.df is not None:
            with col1:
                if st.button(label="Back", key="back_btn"):
                    st.session_state["sample_df"] = None
                    st.rerun(scope="fragment")
            st.dataframe(st.session_state.df.sample(6, random_state=35),
                         column_config=st.session_state.col_cfg, hide_index=True)
        else:
            with col1:
                if st.button(label="Sample Dataframe", key="sample_btn"):
                    st.session_state["sample_df"] = 1
                    st.rerun(scope="fragment")
            st.session_state.err_mode = 2

    # Column 2: Save dataframe button to save to database
    st.session_state["save_state"] = None
    with col2:
        if st.button(label="Save Dataframe", help=st.session_state.save_state):
            if st.session_state.df is not None:
                if not st.session_state.is_data_full:
                    st.session_state.df.to_csv(os.path.join(os.getcwd(), "data",
                                                            cfg.available_loc("str")))
                    st.session_state.save_state = st.success("File Saved")
                else:
                    st.session_state.err_mode = 1
            else:
                st.session_state.err_mode = 2

    if st.session_state.err_mode == 1:
        st.warning("Database is full. Delete a file if you want to save this file ")
    elif st.session_state.err_mode == 2:
        st.warning(body="Upload CSV file first")
## =======================================







## =======================================
## Main Layout
## =======================================

st.title("Home Page")
st.caption("This is where navigation and shorthand stats should be")

# -> Upload files expander
with st.expander(label="Upload files"):
    upload_files_main()
    upload_files_btns()


## =======================================






st.session_state["is_data_full"] = False
if not any(cfg.available_loc("lst")):
    st.session_state.is_data_full = True
