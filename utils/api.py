"""Utility functions for API interactions"""

import os
import requests


COLAB_API_URL = os.getenv("COLAB_API_URL", "").rstrip("/")


def colab_is_available() -> bool:
    """Return True if the Colab server is reachable."""
    if not COLAB_API_URL:
        return False
    try:
        r = requests.get(f"{COLAB_API_URL}/docs", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def prepare_groq_context(df, include_sentiment=True, include_price=False, reviews_sample_size=10):
    """Prepare context data for Groq summarization"""
    context_parts = []

    if include_sentiment and "predicted_sentiment" in df.columns:
        sentiment_summary = df["predicted_sentiment"].value_counts().to_dict()
        context_parts.append(f"Sentiment Distribution: {sentiment_summary}")

    if "reviews.rating" in df.columns:
        avg_rating = df["reviews.rating"].mean()
        context_parts.append(f"Average Rating: {avg_rating:.2f}/5")
        context_parts.append(f"Total Reviews: {len(df)}")

    if include_price and "price" in df.columns:
        try:
            price_val = df["price"].iloc[0] if len(df) > 0 else "N/A"
            context_parts.append(f"Price: {price_val}")
        except Exception:
            pass

    if "reviews.text" in df.columns:
        reviews_sample = df["reviews.text"].dropna().head(reviews_sample_size).tolist()
        context_parts.append(f"Sample Reviews: {' | '.join(reviews_sample[:3])}")

    return "\n".join(context_parts)


def create_groq_prompt(context_string, summary_type, product_name):
    """Create a prompt for Groq LLM"""
    prompt = f"""You are a market analyst for the marketing department. Generate a concise {summary_type.lower()} summary based on the following customer review data:

{context_string}

Product: {product_name if product_name else 'Analyzed Product'}
Summary Type: {summary_type}

Provide actionable insights that would be valuable for marketing strategies. Keep it focused and professional (2-3 paragraphs)."""
    return prompt


def call_groq_api(groq_client, prompt):
    """Call Groq API and return summary"""
    message = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1000,
    )
    return message.choices[0].message.content
