# Automated Customer Reviews Classifier

An NLP-powered web application that aggregates and analyses customer reviews using sentiment classification, product category clustering, and review summarisation.

## Project Goal

A comprehensive platform to analyse Amazon product reviews through three core components:
- **Sentiment Classification** - Classify reviews into positive, negative, and neutral categories using transformer models
- **Product Category Clustering** - Organise product categories into broader, meaningful groups using hierarchical clustering
- **Review Summarisation** - Generate article summaries highlighting top products and complaints by category using Groq LLM

## Main Features

### 1. Classification
Classify customer reviews into sentiment categories (positive, negative, neutral) using transformer-based models.

**Implementation:**
- Uses `distilbert/distilbert-base-uncased-finetuned-sst-2-english` pre-trained model
- Combines review title and text for complete context
- GPU-accelerated processing with batch inference
- Outputs classification labels with confidence scores

**Approach:**
- Combines review title and text data
- Runs inference on GPU for performance
- Provides both labels and confidence metrics
- Generates confusion matrix and classification reports

### 2. Product Category Clustering
Simplifies the product dataset by clustering similar categories into broader groups.

**Implementation:**
- Preprocesses categories through:
  - Text normalization (lowercase, punctuation removal)
  - Stop word filtering
  - Duplicate removal
- Uses hierarchical clustering with `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Maps original categories to consolidated groups (e.g., Kindle e-readers, Smart Assistants, Fire e-readers)

**Clustering Steps:**
1. Extract and normalize category terms
2. Generate embeddings using sentence transformers
3. Apply hierarchical clustering with Ward linkage
4. Map original categories to new cluster labels

### 3. Review Summarisation
**Status:** In Development

Generates article summaries and identifies key insights from reviews by category using Groq LLM.

**Features:**
- Extracts top complaints and worst features per product category
- Generates sentiment-based summary reports
- Highlights top-recommended products in each category
- Identifies and reports worst individual products
- Streamlit interface for viewing summaries by category

## Data Preprocessing

The pipeline includes comprehensive data cleaning:
1. **Drops irrelevant columns** (IDs, usernames, dates, etc.)
2. **Handles missing values** in review text and ratings
3. **Feature engineering:**
   - Creates sentiment labels from star ratings (1-2: negative, 3: neutral, 4-5: positive)
   - Normalises numerical features
4. **Text cleaning** ready for model input

## Project Structure

```
.
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── overview.ipynb                  # Complete project documentation and experiments
├── requirements.txt                # Python dependencies
├── data/                           # Datasets directory
│   ├── 1429_1.csv                 # Primary Amazon reviews dataset
│   ├── Customer-revs-of-products.csv
│   └── May19-revs.csv
└── static/                        # Web assets (images, icons)
```

## Installation

1. **Clone or download the project:**
   ```bash
   cd "review classifier project"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Groq API (optional, for summarisation):**
   - Add your Groq API key to `config.py`
   - Summarisation features will function once configured

## Usage

### Running the Streamlit Application

```bash
streamlit run app.py
```

The application provides an interactive interface to:
- Upload or select datasets
- View sentiment classification results with confidence scores
- Explore clustered product categories
- Generate and view review summaries (when summarisation is fully configured)

### Exploring the Development Notebook

```bash
jupyter notebook overview.ipynb
```

The notebook contains the complete experimental pipeline and analysis documentation.

## Technologies & Models

### NLP Models
- **Sentiment Classification:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (Hugging Face Transformers)
- **Category Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Review Summarisation:** Groq LLM

### Libraries & Frameworks
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, transformers, sentence-transformers
- **Visualisation:** matplotlib, seaborn
- **Web Framework:** Streamlit
- **Clustering:** scipy (hierarchical clustering)
- **LLM Integration:** Groq API

## Data Source

- **Primary Dataset:** Kaggle - Datafiniti Consumer Reviews of Amazon Products
- **Format:** CSV files with product names, categories, reviews, ratings, and metadata
- **Scale:** Multiple datasets included for comprehensive testing

## Model Performance

The classification model evaluation includes:
- Precision, Recall, and F1-Score per sentiment category
- Confusion matrix visualisation
- Average confidence scores across the dataset

## To-Do & Future Work

### Completed
- [x] Implement sentiment classification with transformer model
- [x] Build product category clustering pipeline
- [x] Create Streamlit web interface
- [x] Generate sentiment analysis reports and visualisations

### In Progress
- [ ] Complete LLM-based review summarisation integration
- [ ] Optimise Groq API queries for performance
- [ ] Expand summarisation features with product recommendations

### Future Enhancements
- Implement user feedback loop for model improvement
- Deploy clustering-based product recommendation engine
- Add multi-language support for international datasets
- Create automated reporting and export functionality
- Implement model retraining pipeline with new data

## Notes

- The project prioritises **model accuracy vs. efficiency trade-offs**
- GPU acceleration is leveraged for transformer inference when available
- Both GPU-optimised and CPU fallback implementations are included
- Category clustering uses hierarchical methods with dendrograms for visualisation
- All features are integrated into the Streamlit interface for seamless interaction
