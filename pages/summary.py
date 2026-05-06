"""AI Summary page module (Groq)"""

import streamlit as st
import pandas as pd
from datetime import datetime
from config import get_groq_client, load_sentiment_classifier
from utils.data_processing import classify_sentiments
from utils.api import prepare_groq_context, create_groq_prompt, call_groq_api


def render():
    """Render AI summary page"""
    st.title("🤖 AI Product Summary (Groq)")
    st.caption("Generate intelligent summaries of products based on customer reviews and market data")
    
    groq_client = get_groq_client()
    if groq_client is None:
        st.error("⚠️ Groq API key not found. Please set the GROQ_API_KEY environment variable.")
        st.info("To use this feature, set your Groq API key: `export GROQ_API_KEY='your-key-here'`")
        return
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first")
        return
    
    df = st.session_state.df.copy()
    
    # Summary configuration
    st.subheader("Configure Summary Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_name = st.text_input("Product Name (leave empty to auto-select)")
    
    with col2:
        summary_type = st.selectbox("Summary Type", ["Overview", "Strengths & Weaknesses", "Market Positioning", "Recommendations"])
    
    # Optional filters
    st.subheader("Optional Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_price = st.checkbox("Include Price Information", value=False)
    
    with col2:
        include_time_range = st.checkbox("Include Time Period Analysis", value=False)
    
    with col3:
        include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)
    
    # Time range filter
    time_range = None
    if include_time_range:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                date_range = st.slider("Select date range:",
                                      value=(df[date_col].min(), df[date_col].max()),
                                      min_value=df[date_col].min(),
                                      max_value=df[date_col].max())
                time_range = date_range
            except:
                st.warning("Could not parse dates")
    
    # Generate summary button
    if st.button("🚀 Generate Summary", type="primary"):
        with st.spinner("Generating summary with Groq..."):
            try:
                # Prepare data for summary
                if product_name:
                    df_filtered = df[df.get('name', '').str.contains(product_name, case=False, na=False) | 
                                     df.get('product', '').str.contains(product_name, case=False, na=False)]
                else:
                    df_filtered = df
                
                if len(df_filtered) == 0:
                    df_filtered = df.head(100)
                
                # Add sentiment if requested
                if include_sentiment and 'predicted_sentiment' not in df_filtered.columns:
                    with st.spinner("Running sentiment analysis..."):
                        classifier = load_sentiment_classifier()
                        class_results = classify_sentiments(df_filtered, classifier)
                        df_filtered['predicted_sentiment'] = class_results['predicted']
                
                # Build context for Groq
                context_string = prepare_groq_context(df_filtered, include_sentiment, include_price)
                
                # Create prompt and call Groq
                prompt = create_groq_prompt(context_string, summary_type, product_name)
                summary_text = call_groq_api(groq_client, prompt)
                
                st.success("Summary Generated!")
                st.subheader(f"📝 {summary_type} Summary")
                st.write(summary_text)
                
                # Download option
                st.download_button(
                    label="📥 Download Summary",
                    data=summary_text,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error generating summary: {e}")
