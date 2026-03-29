#!/usr/bin/env python3
"""
Step 1: Collect article URLs from GDELT's DOC API for the 12 sources used in the paper.
Date range: Feb 1 - Mar 7, 2026
"""

import requests
import json
import time
import os
from datetime import datetime, timedelta

# The 12 outlets from the paper
SOURCES = {
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

# GDELT DOC API - free, no key needed
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Keywords to search for Middle East conflict coverage
KEYWORDS = [
    "Iran Israel war",
    "Iran strikes",
    "Israel strikes Iran",
    "Middle East conflict 2026",
    "Operation Epic Fury",
    "Strait of Hormuz",
    "Natanz nuclear",
    "Qatar LNG halt",
    "Iran Supreme Leader",
    "Khamenei",
    "Cyprus missiles",
    "Iran retaliation",
    "Middle East escalation",
    "Iran war",
    "Israel Iran military",
]

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def query_gdelt(keyword, start_date="20260201000000", end_date="20260308000000", max_records=250):
    """Query GDELT DOC API for articles matching keyword in date range."""
    params = {
        "query": keyword,
        "mode": "ArtList",
        "maxrecords": max_records,
        "format": "json",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "sort": "DateDesc",
    }
    
    try:
        resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
        return articles
    except Exception as e:
        print(f"  Error querying '{keyword}': {e}")
        return []


def is_target_source(url):
    """Check if URL belongs to one of our 12 target outlets."""
    url_lower = url.lower()
    for domain in SOURCES:
        if domain in url_lower:
            return domain
    return None


def main():
    all_articles = {}  # url -> article data (dedup by URL)
    
    print("=" * 60)
    print("GDELT Article Collection")
    print(f"Date range: Feb 1 - Mar 7, 2026")
    print(f"Target sources: {len(SOURCES)}")
    print("=" * 60)
    
    for i, keyword in enumerate(KEYWORDS):
        print(f"\n[{i+1}/{len(KEYWORDS)}] Searching: '{keyword}'")
        articles = query_gdelt(keyword)
        
        new_count = 0
        target_count = 0
        for article in articles:
            url = article.get("url", "")
            source_domain = is_target_source(url)
            
            if url not in all_articles:
                article["_source_domain"] = source_domain
                article["_source_name"] = SOURCES.get(source_domain, "Other")
                all_articles[url] = article
                new_count += 1
                if source_domain:
                    target_count += 1
        
        total_target = sum(1 for a in all_articles.values() if a.get("_source_domain"))
        print(f"  Found {len(articles)} articles, {new_count} new, {target_count} from target sources")
        print(f"  Running total: {len(all_articles)} unique articles, {total_target} from target sources")
        
        # Be nice to the API
        time.sleep(2)
    
    # Also search by source domain directly
    print("\n" + "=" * 60)
    print("Searching by source domain...")
    for domain, name in SOURCES.items():
        if domain == "bbc.co.uk":
            continue  # Already covered by bbc.com search
        
        keyword = f"sourcedomain:{domain} (Iran OR Israel OR war OR conflict OR strikes)"
        print(f"\n  Searching {name} ({domain})...")
        articles = query_gdelt(keyword)
        
        new_count = 0
        for article in articles:
            url = article.get("url", "")
            if url not in all_articles:
                source_domain = is_target_source(url) or domain
                article["_source_domain"] = source_domain
                article["_source_name"] = name
                all_articles[url] = article
                new_count += 1
        
        print(f"  Found {len(articles)} articles, {new_count} new")
        time.sleep(2)
    
    # Filter to target sources only
    target_articles = {url: a for url, a in all_articles.items() if a.get("_source_domain")}
    other_articles = {url: a for url, a in all_articles.items() if not a.get("_source_domain")}
    
    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total unique articles: {len(all_articles)}")
    print(f"From target sources: {len(target_articles)}")
    print(f"From other sources: {len(other_articles)}")
    
    print("\nBreakdown by target source:")
    source_counts = {}
    for a in target_articles.values():
        name = a.get("_source_name", "Unknown")
        source_counts[name] = source_counts.get(name, 0) + 1
    for name, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    
    # Save results
    output_file = os.path.join(OUTPUT_DIR, "gdelt_articles.json")
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "collected_at": datetime.now().isoformat(),
                "date_range": "2026-02-01 to 2026-03-07",
                "total_articles": len(all_articles),
                "target_source_articles": len(target_articles),
                "keywords_used": KEYWORDS,
            },
            "target_articles": list(target_articles.values()),
            "other_articles": list(other_articles.values()),
        }, f, indent=2, default=str)
    
    print(f"\nSaved to {output_file}")
    
    # Save just URLs for scraping
    urls_file = os.path.join(OUTPUT_DIR, "target_urls.txt")
    with open(urls_file, "w") as f:
        for url in sorted(target_articles.keys()):
            f.write(url + "\n")
    print(f"Target URLs saved to {urls_file}")


if __name__ == "__main__":
    main()
