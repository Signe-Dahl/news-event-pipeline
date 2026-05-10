# backend/rss_collector.py
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List

import requests

from config import COLLECTION_PROFILE, REQUEST_TIMEOUT


RSS_FEEDS = [
    {
        "name": "ec_energy_news",
        "url": "https://energy.ec.europa.eu/node/2/rss_en",
        "source": "European Commission - DG Energy",
        "language": "en",
    }
]


def make_article_id(
    url: str,
    title: str,
    source: str = "",
    published_at: str = "",
) -> str:
    if url:
        base = url.strip().lower()
    else:
        base = f"{title.strip().lower()}|{source.strip().lower()}|{published_at.strip()}"

    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def parse_rss_date(value: str) -> str:
    if not value:
        return ""

    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
    except Exception:
        return value


def normalize_rss_article(
    item: ET.Element,
    feed: Dict[str, str],
) -> Dict[str, Any]:
    title = (item.findtext("title") or "").strip()
    description = (item.findtext("description") or "").strip()
    url = (item.findtext("link") or "").strip()
    published_at = parse_rss_date(item.findtext("pubDate") or "")

    source = feed["source"]

    article_id = make_article_id(
        url=url,
        title=title,
        source=source,
        published_at=published_at,
    )

    raw = {
        "feed_name": feed["name"],
        "feed_url": feed["url"],
        "title": title,
        "description": description,
        "url": url,
        "source": source,
        "publishedAt": published_at,
    }

    return {
        "article_id": article_id,
        "profile": COLLECTION_PROFILE["name"],
        "matched_keyword": f"rss:{feed['name']}",
        "title": title,
        "description": description,
        "url": url,
        "source": source,
        "category": None,
        "language": feed.get("language"),
        "country": None,
        "published_at": published_at,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "raw": raw,
    }


def fetch_rss_feed(feed: Dict[str, str]) -> List[Dict[str, Any]]:
    response = requests.get(feed["url"], timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    articles = []
    for item in root.findall(".//item"):
        articles.append(normalize_rss_article(item, feed))

    return articles


def fetch_rss_articles() -> List[Dict[str, Any]]:
    articles = []

    for feed in RSS_FEEDS:
        articles.extend(fetch_rss_feed(feed))

    return articles


if __name__ == "__main__":
    articles = fetch_rss_articles()

    print(f"Fetched {len(articles)} RSS articles")

    for article in articles[:5]:
        print(f"- {article['published_at']} | {article['title']}")
        print(f"  {article['url']}")