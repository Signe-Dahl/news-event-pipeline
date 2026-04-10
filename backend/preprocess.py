# preprocess.py
import logging
from typing import Any, Dict, List

from config import (
    EXCLUDED_TITLE_TERMS,
    EXCLUDED_DESCRIPTION_TERMS,
    MIN_DESCRIPTION_LENGTH,
)

logger = logging.getLogger(__name__)


# Text normalizer
def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


# Relevance filtering
def is_relevant_article(article: Dict[str, Any]) -> bool:
    title = (article.get("title") or "").lower()
    description = (article.get("description") or "").lower()

    if not title:
        return False

    if not description:
        return False

    if len(description.strip()) < MIN_DESCRIPTION_LENGTH:
        return False

    # Filter out irrelevant articles like stock watchlists etc.
    for term in EXCLUDED_TITLE_TERMS:
        if term in title:
            return False

    for term in EXCLUDED_DESCRIPTION_TERMS:
        if term in description:
            return False

    return True

# Build classification input
def build_clean_text(article: Dict[str, Any]) -> str:
    title = normalize_whitespace(article.get("title", ""))
    description = normalize_whitespace(article.get("description", ""))

    if title and description:
        return f"Title: {title}\nDescription: {description}"
    elif title:
        return f"Title: {title}"
    else:
        return description


# Process single article
def preprocess_article(article: Dict[str, Any]) -> Dict[str, Any]:
    processed = article.copy()

    processed["title"] = normalize_whitespace(processed.get("title", ""))
    processed["description"] = normalize_whitespace(processed.get("description", ""))
    processed["source"] = normalize_whitespace(processed.get("source", ""))
    processed["url"] = normalize_whitespace(processed.get("url", ""))

    processed["clean_text"] = build_clean_text(processed)

    return processed


# Main preprocessing steps
def preprocess_articles(raw_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("Starting preprocessing")

    # Step 1: filter irrelevant
    filtered_articles = [
        article for article in raw_articles if is_relevant_article(article)
    ]
    logger.info("After filtering: %s articles", len(filtered_articles))

    # Step 2: clean + normalize
    processed_articles = [
        preprocess_article(article) for article in filtered_articles
    ]
    logger.info("After normalization: %s articles", len(processed_articles))

    return processed_articles

# Test
if __name__ == "__main__":
    sample_articles = [
        {
            "article_id": "1",
            "title": "Green Energy Stocks To Consider – March 31st",
            "description": "Nuvve, NWTN, and others are stocks to watch...",
            "source": "example",
            "url": "https://example.com/1",
        },
        {
            "article_id": "2",
            "title": "Google’s new clean energy deal includes massive battery",
            "description": "Form Energy's iron-air batteries will store wind and solar power.",
            "source": "techcrunch",
            "url": "https://example.com/2",
        },
    ]

    processed = preprocess_articles(sample_articles)

    print("\nProcessed articles:")
    for article in processed:
        print("-" * 50)
        print("Title:", article["title"])
        print("Clean text:", article["clean_text"])