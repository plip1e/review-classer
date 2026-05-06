"""Classification page module"""

import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report
from config import load_sentiment_classifier
from utils.data_processing import clean_data, classify_sentiments
from utils.visualizations import (
    plot_sentiment_distribution,
    plot_confidence_scores,
    plot_confusion_matrix,
    plot_sentiment_trends
)


def render():
    """Render classification page"""
    st.title("📊 Review Classification")
    st.caption("Automated sentiment classification and analysis of customer reviews")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first")
        return
    
    df = st.session_state.df.copy()
    
    # Clean data
    with st.spinner("Cleaning data..."):
        df_clean = clean_data(df)
        if df_clean is None:
            st.error("Dataset must contain 'reviews.text' and 'reviews.rating' columns")
            return
    
    st.success(f"Loaded {len(df_clean)} reviews")
    
    # Run classification
    st.subheader("Running Classification Model...")
    with st.spinner("Classifying reviews (this may take a moment)..."):
        try:
            classifier = load_sentiment_classifier()
            class_results = classify_sentiments(df_clean, classifier)
            df_clean['predicted_sentiment'] = class_results['predicted']
            df_clean['confidence_score'] = class_results['scores']
        except Exception as e:
            st.error(f"Error during classification: {e}")
            return
    
    # Overall Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(df_clean))
    with col2:
        positive_count = (df_clean['predicted_sentiment'] == 'positive').sum()
        st.metric("Positive", positive_count, f"{positive_count/len(df_clean)*100:.1f}%")
    with col3:
        neutral_count = (df_clean['predicted_sentiment'] == 'neutral').sum()
        st.metric("Neutral", neutral_count, f"{neutral_count/len(df_clean)*100:.1f}%")
    with col4:
        negative_count = (df_clean['predicted_sentiment'] == 'negative').sum()
        st.metric("Negative", negative_count, f"{negative_count/len(df_clean)*100:.1f}%")
    
    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_sentiment_distribution(df_clean)
        st.pyplot(fig)
    
    # Confidence Scores
    with col2:
        fig = plot_confidence_scores(df_clean)
        st.pyplot(fig)
    
    # Confusion Matrix (if we have ground truth)
    if 'reviews_sentiment' in df_clean.columns:
        st.subheader("Model Performance: Confusion Matrix")
        labels = ['positive', 'neutral', 'negative']
        fig = plot_confusion_matrix(df_clean['reviews_sentiment'], df_clean['predicted_sentiment'], labels)
        st.pyplot(fig)
        
        # Classification Report
        st.text(classification_report(df_clean['reviews_sentiment'], df_clean['predicted_sentiment'], labels=labels))
    
    # Sentiment Over Time
    st.subheader("Sentiment Trends Over Time")
    date_cols = [col for col in df_clean.columns if 'date' in col.lower()]
    
    if date_cols:
        date_col = date_cols[0]
        try:
            fig = plot_sentiment_trends(df_clean, date_col)
            if fig:
                st.pyplot(fig)
        except Exception as e:
            st.info(f"Could not generate time series: {e}")
    
    # Sample reviews by sentiment
    st.subheader("Sample Reviews by Sentiment")
    sentiment_choice = st.selectbox("View samples for:", ['positive', 'neutral', 'negative'])
    
    samples = df_clean[df_clean['predicted_sentiment'] == sentiment_choice].head(5)
    for idx, row in samples.iterrows():
        with st.expander(f"{row.get('reviews.title', 'Review')} ({row['confidence_score']:.2%} confidence)"):
            st.write(f"**Review:** {row['reviews.text']}")
            if 'reviews.rating' in row:
                st.write(f"**Rating:** {row['reviews.rating']}/5")
