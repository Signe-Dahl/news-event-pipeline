# Green Energy News Event Pipeline

This project implements an end-to-end data pipeline for collecting, processing, classifying, summarizing and monitoring news articles related to green energy and climate technology. The system combines automated data collection, LLM-based classification, AI-generated daily summarization, and LLM-as-a-judge evaluation to provide insights into both pipeline performance and emerging developments within the green energy domain.

---

## Project Overview
### Pipeline diagram
![Pipeline Diagram](https://raw.githubusercontent.com/Signe-Dahl/news-event-pipeline/monitoring/Pipeline.png) 

### Description of steps
The pipeline performs the following steps:

1. **Data Collection**  
   Retrieves articles from both European Commission RSS feeds and NewsAPI using targeted queries related to green energy topics within the EU.

2. **Preprocessing**  
   Cleans and filters articles to remove noise and prepare them for classification.

3. **Classification**  
   Uses a large language model (via API) to assign an action category to each article.

4. **Daily Summarization**  
   Generates AI-based daily summaries, decision implications and selected top stories from the collected articles.

5. **Monitoring (LLM-as-a-Judge)**  
   A Second LLM evaluates:
   - Whether the predicted label is correct

6. **Storage**  
   All data is stored in a SQLite database:
   - `raw_articles`
   - `classified_articles`
   - `monitoring_results`
   - `daily_summaries`

7. **API + Frontend**  
   Data is exposed through a FastAPI backend and visualized using Streamlit dashboards.
---

## Configuration

Configuration of the pipeline is managed through `backend/config.py` and environment variables.

Required environment variables:

- `NEWSAPI_API_KEY`
- `GROQ_API_KEY`

For local development, these can be stored in a .env file.

Example .env file:
```bash
NEWSAPI_API_KEY=your_newsapi_key
GROQ_API_KEY=your_groq_key
```

---

## How to Run the Pipeline

1. Install dependencies
- `pip install -r backend/requirements.txt`

2. Run pipeline manually
- `python backend/main.py`

This will:
- Collect articles from NewsAPI and EU RSS feed
- Preprocess article data
- Classify articles using an LLM
- Generate daily summaries and top stories
- Store results in `data/news.db`

---

## Automated Pipeline (GitHub Actions)

The pipeline runs automatically every morning at a set time via GitHub Actions.

It performs:
- Full pipeline execution
- Monitoring evaluation
- Updates data/news.db
- Syncs database to Hugging Face API Space

---

## Artifact Tracking

The system tracks multiple types of artefacts:

Data artefacts:
- `raw_articles`
- `classified_articles`
- `monitoring_results`
- `daily_summaries`

Model artefacts:
- LLM configurations and prompts (stored in code)

Code artefacts:
- Version-controlled via GitHub

Metrics artefacts:
- LLM-based evaluation outputs (stored in DB)

Operational artefacts:
- Logs and pipeline statistics (available in GitHub Actions)

--- 

## Frontend (Deployed)

Streamlit dashboard: https://huggingface.co/spaces/Signe22/Green-Energy-News-Dashboard 

API (FastAPI): https://huggingface.co/spaces/Signe22/Article_Data_API

---

## Frontend Docker Setup

The Streamlit frontend is containerized using Docker and started with Docker Compose.

### Run the frontend with Docker Compose

This method requires that the repository has been cloned locally.

From the frontend/ directory:
```bash
docker compose up --build
```

This will build the image and start the Streamlit app.

Then open: http://localhost:8501

Stop the frontend:
```bash
docker compose down
```

### Run the frontend via Docker
First, log in to Docker Hub:
```bash
docker login
```

Then pull the pre-built image:
```bash
docker pull signe1504/news-dashboard:2.0
```

Run the container locally:
```bash
docker run -p 8501:8501 signe1504/news-dashboard:2.0
```

Then open: http://localhost:8501

---
## Monitoring Dashboard

The monitoring dashboard provides insights into:
- Classification quality
- Relevance filtering performance
- Low-confidence predictions
- Problematic sources and categories
- Emerging problem patterns

Deployed here: https://huggingface.co/spaces/Signe22/Green-Energy-News-Monitoring

--- 

## Reproducibility

To reproduce the project:
1. Clone the repository
2. Create a local `.env` file or put the required environment variables into GitHub secrets
3. Run the pipeline locally or via GitHub Actions
4. Launch the Streamlit app using Docker Compose
