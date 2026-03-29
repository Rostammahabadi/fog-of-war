#!/usr/bin/env python3
"""
Step 1b: Collect article URLs via Google News RSS feeds (what the paper actually used).
Then scrape full text with trafilatura.
"""

import requests
import json
import time
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urlparse

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target sources
SOURCES = [
    "middleeasteye.net",
    "aljazeera.com",
    "thenationalnews.com",
    "al-monitor.com",
    "reuters.com",
    "bloomberg.com",
    "theguardian.com",
    "ft.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "foxnews.com",
]

SOURCE_NAMES = {
    "middleeasteye.net": "Middle East Eye",
    "aljazeera.com": "Al Jazeera",
    "thenationalnews.com": "The National (UAE)",
    "al-monitor.com": "Al-Monitor",
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "theguardian.com": "The Guardian",
    "ft.com": "Financial Times",
    "apnews.com": "AP News",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "foxnews.com": "Fox News",
}

# Search queries to cover the conflict comprehensively
QUERIES = [
    "Iran Israel war",
    "Iran Israel strikes",
    "Operation Epic Fury",
    "Iran retaliation",
    "Middle East conflict",
    "Middle East war 2026",
    "Strait of Hormuz",
    "Natanz nuclear",
    "Qatar LNG",
    "Iran Supreme Leader death",
    "Mojtaba Khamenei",
    "Cyprus missiles Iran",
    "Israel ground invasion Lebanon",
    "Iran oil tanker attack",
    "US Iran military",
    "Israel Iran nuclear",
    "Middle East escalation",
    "Iran war casualties",
    "Geneva talks Iran",
    "NATO Middle East",
    "UK Iran strikes",
    "Iran sanctions",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def resolve_google_news_url(google_url):
    """Try to resolve Google News redirect URL to the actual article URL."""
    try:
        resp = requests.get(google_url, headers=HEADERS, allow_redirects=True, timeout=15)
        return resp.url
    except:
        return google_url


def search_google_news_rss(query, source_domain=None, after="2026-02-01", before="2026-03-08"):
    """Search Google News RSS for articles."""
    if source_domain:
        full_query = f"{query} site:{source_domain} after:{after} before:{before}"
    else:
        full_query = f"{query} after:{after} before:{before}"
    
    url = f"https://news.google.com/rss/search?q={quote_plus(full_query)}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        
        root = ET.fromstring(resp.content)
        articles = []
        
        for item in root.findall(".//item"):
            title = item.find("title")
            link = item.find("link")
            pub_date = item.find("pubDate")
            source = item.find("source")
            
            article = {
                "title": title.text if title is not None else "",
                "google_url": link.text if link is not None else "",
                "pub_date": pub_date.text if pub_date is not None else "",
                "source_name": source.text if source is not None else "",
                "source_url": source.get("url", "") if source is not None else "",
            }
            articles.append(article)
        
        return articles
    except Exception as e:
        print(f"    Error: {e}")
        return []


def get_source_domain(url):
    """Extract and match source domain from URL."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        hostname = hostname.replace("www.", "")
        for domain in SOURCES:
            if domain in hostname:
                return domain
    except:
        pass
    return None


def main():
    all_articles = {}  # url -> article data
    
    print("=" * 60)
    print("Google News RSS Collection")
    print(f"Date range: Feb 1 - Mar 7, 2026")
    print("=" * 60)
    
    # Strategy 1: Search per source + per query
    total_searches = len(QUERIES) * len(SOURCES)
    search_num = 0
    
    for query in QUERIES:
        for source_domain in SOURCES:
            if source_domain == "bbc.co.uk":
                continue
            search_num += 1
            
            source_name = SOURCE_NAMES.get(source_domain, source_domain)
            print(f"\r[{search_num}/{total_searches}] {source_name}: '{query}'", end="", flush=True)
            
            articles = search_google_news_rss(query, source_domain)
            
            new = 0
            for article in articles:
                # Use title as dedup key (since Google URLs are redirects)
                title_key = article["title"].strip().lower()
                if title_key and title_key not in all_articles:
                    article["_source_domain"] = source_domain
                    article["_source_name"] = source_name
                    article["_query"] = query
                    all_articles[title_key] = article
                    new += 1
            
            if new > 0:
                print(f" -> +{new} new")
            
            # Rate limit: ~1 req/sec
            time.sleep(1.5)
    
    # Strategy 2: Broad queries without site filter (catch articles we missed)
    print("\n\nBroad searches (no site filter)...")
    broad_queries = [
        "Iran Israel war 2026",
        "Middle East conflict February March 2026",
        "Operation Epic Fury Iran",
        "Iran nuclear strikes 2026",
        "Strait of Hormuz closure 2026",
        "Qatar energy halt war",
    ]
    
    for query in broad_queries:
        print(f"  Searching: '{query}'")
        articles = search_google_news_rss(query)
        
        new = 0
        for article in articles:
            title_key = article["title"].strip().lower()
            if title_key and title_key not in all_articles:
                # Check if it's from a target source by source name
                source_text = (article.get("source_name", "") + " " + article.get("source_url", "")).lower()
                matched_domain = None
                for domain, name in SOURCE_NAMES.items():
                    if domain in source_text or name.lower() in source_text:
                        matched_domain = domain
                        break
                
                article["_source_domain"] = matched_domain
                article["_source_name"] = SOURCE_NAMES.get(matched_domain, article.get("source_name", "Other"))
                article["_query"] = query
                all_articles[title_key] = article
                new += 1
        
        print(f"    Found {len(articles)}, {new} new")
        time.sleep(2)
    
    # Summary
    target = {k: v for k, v in all_articles.items() if v.get("_source_domain")}
    
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total unique articles (by title): {len(all_articles)}")
    print(f"From target sources: {len(target)}")
    
    source_counts = {}
    for a in target.values():
        name = a.get("_source_name", "Unknown")
        source_counts[name] = source_counts.get(name, 0) + 1
    
    print("\nBreakdown:")
    for name, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    
    # Save
    output_file = os.path.join(OUTPUT_DIR, "google_news_articles.json")
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "collected_at": datetime.now().isoformat(),
                "total": len(all_articles),
                "target_source_count": len(target),
            },
            "articles": list(all_articles.values()),
        }, f, indent=2, default=str)
    
    print(f"\nSaved to {output_file}")
    print(f"\nNext step: Resolve Google redirect URLs and scrape full text with trafilatura")


if __name__ == "__main__":
    main()
