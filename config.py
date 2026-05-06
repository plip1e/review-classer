'''Configuration module for project'''

import os
import streamlit as st
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from groq import Groq


# =======================================
# Database Configuration
# =======================================

MAX_SAVED_CSV = 5


def available_loc(type_to_return="str"):
    '''
    Returns list of available locations,
    or string of most applicable loc for saving csv files in the data directory
    args:
        type_to_return: str, either "lst" or "str",
        determines the type of output returned by the function
    '''

    loc_lst = [f'{a}_data.csv' for a in range(1, MAX_SAVED_CSV + 1)
               if f'{a}_data.csv' not in os.listdir(os.path.join(os.getcwd(), "data"))]

    return loc_lst if type_to_return == "lst" else loc_lst[0]


# =======================================
# ML Model Configuration & Caching
# =======================================

@st.cache_resource
def load_sentiment_classifier():
    """Load sentiment classification model with GPU support"""
    classifier = pipeline(
        task="text-classification",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512
    )
    return classifier


@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer for category clustering"""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# =======================================
# API Configuration & Caching
# =======================================

@st.cache_resource
def get_groq_client():
    """Initialize Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


if __name__ == "__main__":
    pass
