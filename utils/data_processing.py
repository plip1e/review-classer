"""Utility functions for data processing"""

import os
import pandas as pd
import numpy as np
import requests

# If COLAB_API_URL is set in .env / environment, heavy functions call Colab.
# Otherwise they run locally (original behaviour).
COLAB_API_URL = os.getenv("COLAB_API_URL", "").rstrip("/")


def clean_data(df: pd.DataFrame, drop_cols: list = []) -> pd.DataFrame:
    """Clean and preprocess the dataframe"""
    df_clean = df.drop(columns=drop_cols, errors="ignore")

    required_cols = ["reviews.text", "reviews.rating"]
    if not all(col in df_clean.columns for col in required_cols):
        return None

    df_clean = df_clean.dropna(subset=required_cols)
    df_clean = df_clean.fillna(False)

    def map_sentiment(rating):
        try:
            rating = int(rating)
            if rating >= 4:
                return "positive"
            elif rating == 3:
                return "neutral"
            else:
                return "negative"
        except (ValueError, TypeError):
            return "neutral"

    df_clean["reviews_sentiment"] = df_clean["reviews.rating"].apply(map_sentiment)
    return df_clean


def classify_sentiments(df: pd.DataFrame, classifier=None) -> dict:
    """
    Classify reviews using transformer model.
    Uses Colab remote endpoint when COLAB_API_URL is set, otherwise runs locally.
    `classifier` is ignored when running remotely.
    """
    # Build input texts (same logic as before)
    if "reviews.title" in df.columns:
        review_input = (
            df["reviews.title"].fillna("").astype(str)
            + ". "
            + df["reviews.text"].fillna("").astype(str)
        )
    else:
        review_input = df["reviews.text"].astype(str)

    texts = review_input.str.strip().tolist()
    print(f"Sending {len(texts)} texts to Colab")
    print(f"Sample: {texts[0][:100]}")
    
    if COLAB_API_URL:
        response = requests.post(
            f"{COLAB_API_URL}/classify",
            json={"texts": texts},
            timeout=300,
        )
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        response.raise_for_status()
        
        data = response.json()
        return {
            "predicted": data["predicted"],
            "scores": data["scores"],
            "results": [],  # raw results not returned by remote
        }
    else:
        if classifier is None:
            raise ValueError("classifier must be provided when COLAB_API_URL is not set")
        results = classifier(texts, batch_size=32, truncation=True)
        sentiment_map = {"POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral"}
        predicted = [sentiment_map.get(r["label"], "neutral") for r in results]
        scores = [r["score"] for r in results]
        return {"predicted": predicted, "scores": scores, "results": results}


def cluster_categories(df: pd.DataFrame, model=None) -> dict:
    """
    Cluster product categories into broader groups.
    Uses Colab remote endpoint when COLAB_API_URL is set, otherwise runs locally.
    `model` is ignored when running remotely.
    """
    import re

    if "categories" not in df.columns:
        return None

    raw_categories = df["categories"].dropna().unique().tolist()
    raw_categories = [str(c) for c in raw_categories]

    if COLAB_API_URL:
        # ── Remote path ──────────────────────────────────────────────────────
        response = requests.post(
            f"{COLAB_API_URL}/cluster",
            json={"categories": raw_categories},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        import numpy as np
        return {
            "embeddings": None,  # not returned by remote (not needed locally)
            "linkage": np.array(data["linkage_matrix"]),
            "categories": data["category_names"],
            "corpus": data["corpus"],
            "terms": [],  # not needed downstream
        }
    else:
        # ── Local path (original behaviour) ──────────────────────────────────
        if model is None:
            raise ValueError("model must be provided when COLAB_API_URL is not set")

        rm_punkt = lambda x: re.sub(r"[^\w\s]", "", x)
        cat_stopwords = {
            "electronics", "new", "all", "frys", "e", "computers",
            "all tablets", "tablets", "amazon",
        }

        categories_lst = []
        category_names = []

        for category in raw_categories:
            terms = set()
            for term in category.split(","):
                cleaned = rm_punkt(term.lower().strip())
                if cleaned and cleaned not in cat_stopwords:
                    terms.add(cleaned)
            categories_lst.append(terms)
            category_names.append(category)

        corpus = [" ".join(terms) for terms in categories_lst]
        embeddings = model.encode(corpus)

        from scipy.cluster.hierarchy import linkage
        Z = linkage(embeddings, method="ward")

        return {
            "embeddings": embeddings,
            "linkage": Z,
            "categories": category_names,
            "corpus": corpus,
            "terms": categories_lst,
        }
