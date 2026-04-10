# main.py
import logging
from typing import Any, Dict, List

from collector import collect_articles
from preprocess import preprocess_articles
from classifier import classify_articles


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def run_pipeline() -> Dict[str, Any]:
    """
    Runs the full pipeline end to end:
    1. Collect raw articles
    2. Preprocess / filter
    3. Classify processed articles
    """
    logger.info("Starting pipeline run")

    raw_articles: List[Dict[str, Any]] = collect_articles()
    logger.info("Collected %s raw articles", len(raw_articles))

    processed_articles: List[Dict[str, Any]] = preprocess_articles(raw_articles)
    logger.info("Processed %s articles", len(processed_articles))

    classified_articles: List[Dict[str, Any]] = classify_articles(processed_articles)
    logger.info("Classified %s articles", len(classified_articles))

    logger.info("Pipeline run completed")

    return {
        "raw_count": len(raw_articles),
        "processed_count": len(processed_articles),
        "classified_count": len(classified_articles),
        "classified_articles": classified_articles,
    }


if __name__ == "__main__":
    results = run_pipeline()

    print("\nPipeline summary")
    print("-" * 50)
    print(f"Raw articles:        {results['raw_count']}")
    print(f"Processed articles:  {results['processed_count']}")
    print(f"Classified articles: {results['classified_count']}")

    preview = results["classified_articles"][:5]
    if preview:
        print("\nSample classified articles:")
        for i, article in enumerate(preview, start=1):
            print("-" * 50)
            print(f"{i}. {article.get('title', 'No title')}")
            print(f"Label: {article.get('label', 'No label')}")
            print(f"Source: {article.get('source', 'Unknown')}")
            print(f"Published: {article.get('published_at', 'Unknown')}")