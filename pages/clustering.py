"""Clustering page module"""

import streamlit as st
import pandas as pd
from config import load_sentence_transformer
from utils.data_processing import cluster_categories
from utils.visualizations import plot_dendrogram


def render():
    """Render clustering page"""
    st.title("🔗 Category Clustering")
    st.caption("Hierarchical clustering of product categories into broader groups")
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first")
        return
    
    df = st.session_state.df.copy()
    
    if 'categories' not in df.columns:
        st.error("Dataset must contain a 'categories' column")
        return
    
    st.subheader("Analyzing categories...")
    with st.spinner("Computing category embeddings and clustering..."):
        try:
            model = load_sentence_transformer()
            clustering_result = cluster_categories(df, model)
            if clustering_result is None:
                st.error("Could not process categories")
                return
        except Exception as e:
            st.error(f"Error during clustering: {e}")
            return
    
    # Display category statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Unique Categories", len(clustering_result['categories']))
    with col2:
        total_products = df['categories'].notna().sum()
        st.metric("Total Reviews with Categories", total_products)
    
    # Dendrogram
    st.subheader("Category Dendrogram (Hierarchical Clustering)")
    fig = plot_dendrogram(clustering_result['linkage'], clustering_result['categories'])
    st.pyplot(fig)
    
    # Category Mapping
    st.subheader("Category Mapping")
    st.caption("Suggested grouping of categories into broader product groups:")
    
    # Create suggested groupings based on similarity
    suggested_groups = {
        'E-Readers': [],
        'Tablets': [],
        'Smart Devices': [],
        'Other': []
    }
    
    for cat in clustering_result['categories']:
        cat_lower = str(cat).lower()
        if 'kindle' in cat_lower or 'e-reader' in cat_lower or 'e-ink' in cat_lower:
            suggested_groups['E-Readers'].append(cat)
        elif 'tablet' in cat_lower or 'fire' in cat_lower:
            suggested_groups['Tablets'].append(cat)
        elif 'alexa' in cat_lower or 'echo' in cat_lower or 'tv' in cat_lower or 'assistant' in cat_lower:
            suggested_groups['Smart Devices'].append(cat)
        else:
            suggested_groups['Other'].append(cat)
    
    for group_name, categories in suggested_groups.items():
        if categories:
            with st.expander(f"{group_name} ({len(categories)} categories)"):
                for cat in categories:
                    st.write(f"• {cat}")
    
    # Interactive category selector
    st.subheader("Category Details")
    selected_cat = st.selectbox("Select a category to explore:", clustering_result['categories'])
    
    if selected_cat:
        cat_reviews = df[df['categories'] == selected_cat]
        st.write(f"**Reviews in this category:** {len(cat_reviews)}")
        if 'reviews.rating' in df.columns:
            avg_rating = cat_reviews['reviews.rating'].mean()
            st.write(f"**Average Rating:** {avg_rating:.2f}/5")
