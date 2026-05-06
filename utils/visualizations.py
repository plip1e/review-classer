"""Utility functions for creating visualizations"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_sentiment_distribution(df: pd.DataFrame):
    """Create sentiment distribution chart"""
    sentiment_counts = df['predicted_sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
    ax.bar(sentiment_counts.index, sentiment_counts.values, 
           color=[colors.get(s, '#95a5a6') for s in sentiment_counts.index])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Review Sentiment Distribution")
    return fig


def plot_confidence_scores(df: pd.DataFrame):
    """Create confidence distribution chart"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for sentiment in ['positive', 'neutral', 'negative']:
        scores = df[df['predicted_sentiment'] == sentiment]['confidence_score']
        ax.hist(scores, alpha=0.6, label=sentiment, bins=20)
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Model Confidence Distribution by Sentiment")
    ax.legend()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels):
    """Create confusion matrix heatmap"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Sentiment Confusion Matrix')
    return fig


def plot_sentiment_trends(df: pd.DataFrame, date_col: str):
    """Create sentiment over time chart"""
    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.sort_values(date_col)
        df_copy['week'] = df_copy[date_col].dt.to_period('W')
        
        sentiment_by_week = df_copy.groupby(['week', 'predicted_sentiment']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        sentiment_by_week.plot(ax=ax, marker='o')
        ax.set_xlabel("Week")
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Trends Over Time")
        ax.legend(title="Sentiment")
        return fig
    except Exception as e:
        print(f"Could not generate time series: {e}")
        return None


def plot_dendrogram(linkage_matrix, labels):
    """Create dendrogram for category clustering"""
    from scipy.cluster.hierarchy import dendrogram
    
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linkage_matrix, 
               labels=labels, 
               orientation='right', 
               leaf_font_size=9,
               ax=ax)
    ax.set_xlabel('Distance')
    return fig
