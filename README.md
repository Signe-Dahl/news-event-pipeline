# Green Energy News Event Pipeline

This project implements an end-to-end data pipeline for collecting, processing, classifying, and monitoring news articles related to green energy and climate technology. The system combines automated data collection, LLM-based classification, and LLM-as-a-judge evaluation to provide insights into both pipeline performance and data quality.

---

## Project Overview
### Pipeline diagram
<img width="1020" height="894" alt="image" src="https://github.com/user-attachments/assets/c3a5f7db-f50d-4764-9bfb-625cb25417f5" />


### Description of steps
The pipeline performs the following steps:

1. **Data Collection**  
   Retrieves articles from a news API using a targeted query for green energy topics.

2. **Preprocessing**  
   Cleans and filters articles to remove noise and prepare them for classification.

3. **Classification**  
   Uses a large language model (via API) to assign action categories to each article.

4. **Monitoring (LLM-as-a-Judge)**  
   A Second LLM evaluates:
   - Whether the predicted label is correct

5. **Storage**  
   All data is stored in a SQLite database:
   - `raw_articles`
   - `classified_articles`
   - `monitoring_results`

6. **API + Frontend**  
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
- pip install -r backend/requirements.txt
2. Run pipeline manually
- python backend/main.py

This will:
- Collect articles
- Preprocess data
- Classify articles
- Run monitoring evaluation
- Store results in data/news.db

---

## Automated Pipeline (GitHub Actions)

The pipeline runs automatically daily at 07:00 Danish time via GitHub Actions.

It performs:
- Full pipeline execution
- Monitoring evaluation
- Updates data/news.db
- Syncs database to Hugging Face API Space

---

## Artifact Tracking

The system tracks multiple types of artefacts:

Data artefacts:
- raw_articles
- classified_articles
- monitoring_results

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
docker pull signe1504/news-dashboard:1.0
```

Run the container locally:
```bash
docker run -p 8501:8501 signe1504/news-dashboard:1.0
```

Then open: http://localhost:8501

---
## Monitoring Dashboard

The monitoring dashboard provides insights into:
- Classification quality
- Relevance filtering performance
- Low-confidence predictions
- Problematic sources and categories

Deployed here: https://huggingface.co/spaces/Signe22/Green-Energy-News-Monitoring

--- 

## Reproducibility

To reproduce the project:
1. Clone the repository
2. Create a local .env file or put the required environment variables into GitHub secrets
3. Run the pipeline locally or via GitHub Actions
4. Launch the Streamlit app using Docker Compose
