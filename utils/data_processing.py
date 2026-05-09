"""Utility functions for data processing"""

import os
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
load_dotenv()

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

    if not COLAB_API_URL:
        raise ValueError(
            'COLAB_API_URL is not set in .env. Please set the remote Colab server URL.'
        )
    
    try:
        response = requests.post(
            f"{COLAB_API_URL}/classify",
            json={"texts": texts},
            timeout=300,
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Validate response structure
        if "predicted" not in data or "scores" not in data:
            raise ValueError(
                f"Invalid response from Colab server. Expected 'predicted' and 'scores' fields. "
                f"Got: {list(data.keys())}"
            )
        
        if len(data["predicted"]) != len(texts):
            raise ValueError(
                f"Response mismatch: sent {len(texts)} texts but got {len(data['predicted'])} predictions"
            )
        
        return {
            "predicted": data["predicted"],
            "scores": data["scores"],
            "results": [],  # raw results not returned by remote
        }
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Failed to connect to Colab server at {COLAB_API_URL}/classify: {str(e)}"
        ) from e
    except (KeyError, ValueError) as e:
        raise ValueError(f"Error processing Colab response: {str(e)}") from e


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
        try:
            response = requests.post(
                f"{COLAB_API_URL}/cluster",
                json={"categories": raw_categories},
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            
            # Validate response structure
            required_fields = ["linkage_matrix", "category_names", "corpus"]
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                raise ValueError(
                    f"Invalid response from Colab server. Missing fields: {missing_fields}. "
                    f"Got: {list(data.keys())}"
                )

            import numpy as np
            return {
                "embeddings": None,  # not returned by remote (not needed locally)
                "linkage": np.array(data["linkage_matrix"]),
                "categories": data["category_names"],
                "corpus": data["corpus"],
                "terms": [],  # not needed downstream
            }
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Colab server at {COLAB_API_URL}/cluster: {str(e)}"
            ) from e
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Error processing Colab cluster response: {str(e)}") from e
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
