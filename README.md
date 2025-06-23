# Public-Sentiment-Tracker

A comprehensive system for ingesting, processing, and analyzing civic data from social media and news sources to identify and classify public grievances.

## Architecture Overview

### Data Collection Layer
- **Twitter API Integration**: Scrapes tweets using keywords and hashtags related to civic issues
- **News Scraping**: Extracts content from local news websites
- **Geo-location Extraction**: Identifies location data from posts when available

### Processing Pipeline
- **Text Preprocessing**: Cleaning, tokenization, and TF-IDF vectorization
- **Keyword Extraction**: Using KeyBERT for identifying key topics
- **Sentiment Analysis**: VADER sentiment scoring
- **Classification**: SVM-based grievance categorization and urgency detection

### Visualization
- **Interactive Maps**: Heatmaps showing issue density by location
- **Trend Analysis**: Sentiment trends over time by category
- **Dashboard**: Streamlit-based interface for data exploration

## Project Structure

```
project_root/
├── data_ingestion/          # Data collection modules
├── preprocessing/           # Text preprocessing utilities
├── nlp_modeling/           # NLP and ML components
├── visualization/          # Visualization and dashboard
├── shared/                 # Common utilities
└── run_pipeline.py        # Main pipeline orchestrator
```

## Setup

1. Create virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate.bat
   ```

2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

Run the complete pipeline:
```bash
python run_pipeline.py
```

Launch the dashboard:
```bash
streamlit run visualization/dashboard_app.py
```
