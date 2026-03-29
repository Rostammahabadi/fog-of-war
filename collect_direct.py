#!/usr/bin/env python3
"""
Bypass Google News entirely. Collect article URLs directly from news outlet
RSS feeds and sitemaps, then match to our DB records by title.
"""

import psycopg2
import requests
import xml.etree.ElementTree as ET
import re
import time
import json
import trafilatura
from datetime import datetime
from difflib import SequenceMatcher

DB = "dbname=fog_of_war"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15',
    'Accept': '*/*',
}

# Direct RSS feeds and sitemaps for each outlet
OUTLET_FEEDS = {
    'aljazeera.com': {
        'rss': [
            'https://www.aljazeera.com/xml/rss/all.xml',
        ],
        'sitemap': 'https://www.aljazeera.com/sitemap.xml',
    },
    'reuters.com': {
        'rss': [],
        'sitemap': 'https://www.reuters.com/arc/outboundfeeds/sitemap-index/?outputType=xml',
    },
    'bbc.com': {
        'rss': [
            'https://feeds.bbci.co.uk/news/world/middle_east/rss.xml',
            'https://feeds.bbci.co.uk/news/world/rss.xml',
        ],
        'sitemap': 'https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml',
    },
    'theguardian.com': {
        'rss': [
            'https://www.theguardian.com/world/middleeast/rss',
            'https://www.theguardian.com/world/iran/rss',
        ],
        'sitemap': 'https://www.theguardian.com/sitemaps/news.xml',
    },
    'apnews.com': {
        'rss': [],
        'sitemap': 'https://apnews.com/sitemap.xml',
    },
    'foxnews.com': {
        'rss': [
            'https://moxie.foxnews.com/google-publisher/world.xml',
        ],
        'sitemap': 'https://www.foxnews.com/sitemap.xml',
    },
    'middleeasteye.net': {
        'rss': [
            'https://www.middleeasteye.net/rss',
        ],
        'sitemap': 'https://www.middleeasteye.net/sitemap.xml',
    },
    'al-monitor.com': {
        'rss': [
            'https://www.al-monitor.com/rss',
        ],
        'sitemap': 'https://www.al-monitor.com/sitemap.xml',
    },
    'thenationalnews.com': {
        'rss': [],
        'sitemap': 'https://www.thenationalnews.com/sitemap.xml',
    },
    'ft.com': {
        'rss': [],
        'sitemap': 'https://www.ft.com/sitemaps/news.xml', 
    },
    'bloomberg.com': {
        'rss': [],
        'sitemap': 'https://www.bloomberg.com/feeds/sitemap_news.xml',
    },
}

def fetch_xml(url):
    """Fetch and parse XML."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return ET.fromstring(r.content)
    except Exception as e:
        print(f"    Error fetching {url}: {e}")
    return None

def extract_urls_from_sitemap(xml_root):
    """Extract URLs from a sitemap XML."""
    urls = []
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9',
          'news': 'http://www.google.com/schemas/sitemap-news/0.9'}
    
    # Regular sitemap
    for url_elem in xml_root.findall('.//sm:url', ns):
        loc = url_elem.find('sm:loc', ns)
        lastmod = url_elem.find('sm:lastmod', ns)
        title_elem = url_elem.find('.//news:title', ns)
        
        if loc is not None:
            entry = {
                'url': loc.text,
                'lastmod': lastmod.text if lastmod is not None else None,
                'title': title_elem.text if title_elem is not None else None,
            }
            urls.append(entry)
    
    # Sitemap index (contains links to sub-sitemaps)
    for sitemap_elem in xml_root.findall('.//sm:sitemap', ns):
        loc = sitemap_elem.find('sm:loc', ns)
        if loc is not None:
            urls.append({'type': 'sitemap_index', 'url': loc.text})
    
    return urls

def extract_urls_from_rss(xml_root):
    """Extract URLs from an RSS feed."""
    urls = []
    for item in xml_root.findall('.//item'):
        link = item.find('link')
        title = item.find('title')
        pub_date = item.find('pubDate')
        if link is not None:
            urls.append({
                'url': link.text,
                'title': title.text if title is not None else None,
                'pub_date': pub_date.text if pub_date is not None else None,
            })
    return urls

def normalize_title(title):
    """Normalize a title for fuzzy matching."""
    if not title:
        return ""
    # Remove source suffix, lowercase, strip punctuation
    title = re.sub(r'\s*[-–|]\s*(Reuters|Bloomberg|BBC|Fox News|AP News|The Guardian|Financial Times|Al Jazeera|Al-Monitor|Middle East Eye|The National).*$', '', title, flags=re.IGNORECASE)
    title = title.lower().strip()
    title = re.sub(r'[^\w\s]', '', title)
    return title

def title_similarity(t1, t2):
    """Compare two titles."""
    n1, n2 = normalize_title(t1), normalize_title(t2)
    if not n1 or not n2:
        return 0
    return SequenceMatcher(None, n1, n2).ratio()


def main():
    conn = psycopg2.connect(DB)
    cur = conn.cursor()
    
    # Get all articles that need URLs
    cur.execute("""
        SELECT id, title, title_normalized, source_domain
        FROM articles 
        WHERE (url IS NULL OR url LIKE '%%news.google.com%%')
        AND scrape_status != 'success'
    """)
    needs_url = cur.fetchall()
    
    print(f"{'=' * 60}")
    print(f"Direct URL Collection from News Outlets")
    print(f"Articles needing real URLs: {len(needs_url)}")
    print(f"{'=' * 60}")
    
    # Group by source
    by_source = {}
    for aid, title, title_norm, domain in needs_url:
        by_source.setdefault(domain, []).append((aid, title, title_norm))
    
    for domain, articles in sorted(by_source.items(), key=lambda x: -len(x[1])):
        print(f"\n  {domain}: {len(articles)} articles need URLs")
    
    total_matched = 0
    
    for domain, config in OUTLET_FEEDS.items():
        articles_for_source = by_source.get(domain, [])
        if not articles_for_source:
            continue
        
        print(f"\n{'='*50}")
        print(f"🔍 {domain}: {len(articles_for_source)} articles to match")
        
        found_urls = []
        
        # Try RSS feeds
        for rss_url in config.get('rss', []):
            print(f"  📡 RSS: {rss_url}")
            root = fetch_xml(rss_url)
            if root:
                entries = extract_urls_from_rss(root)
                found_urls.extend(entries)
                print(f"    Found {len(entries)} entries")
            time.sleep(1)
        
        # Try sitemap
        sitemap_url = config.get('sitemap')
        if sitemap_url:
            print(f"  🗺️  Sitemap: {sitemap_url}")
            root = fetch_xml(sitemap_url)
            if root:
                entries = extract_urls_from_sitemap(root)
                # Check for sitemap index
                sub_sitemaps = [e for e in entries if e.get('type') == 'sitemap_index']
                regular_urls = [e for e in entries if not e.get('type')]
                found_urls.extend(regular_urls)
                print(f"    Found {len(regular_urls)} URLs, {len(sub_sitemaps)} sub-sitemaps")
                
                # Follow sub-sitemaps that look like they contain 2026 content
                for sub in sub_sitemaps[:5]:  # Limit to avoid too many requests
                    sub_url = sub['url']
                    if any(x in sub_url for x in ['2026', 'news', 'latest', 'articles']):
                        print(f"    📄 Sub-sitemap: {sub_url}")
                        sub_root = fetch_xml(sub_url)
                        if sub_root:
                            sub_entries = extract_urls_from_sitemap(sub_root)
                            sub_regular = [e for e in sub_entries if not e.get('type')]
                            found_urls.extend(sub_regular)
                            print(f"      Found {len(sub_regular)} URLs")
                        time.sleep(1)
            time.sleep(1)
        
        if not found_urls:
            print(f"  ⚠️  No URLs found from feeds/sitemaps")
            continue
        
        print(f"  Total discovered URLs: {len(found_urls)}")
        
        # Match by title similarity
        matched = 0
        for aid, title, title_norm in articles_for_source:
            best_match = None
            best_score = 0
            
            for entry in found_urls:
                entry_title = entry.get('title', '')
                if entry_title:
                    score = title_similarity(title, entry_title)
                    if score > best_score:
                        best_score = score
                        best_match = entry
            
            if best_match and best_score > 0.6:
                matched += 1
                cur.execute("""
                    UPDATE articles SET url = %s, scrape_status = 'pending', scrape_error = NULL
                    WHERE id = %s
                """, (best_match['url'], aid))
        
        conn.commit()
        total_matched += matched
        print(f"  ✅ Matched {matched}/{len(articles_for_source)} articles by title")
    
    print(f"\n{'='*60}")
    print(f"Total matched: {total_matched}")
    print(f"Now run scrape_articles.py to scrape the matched URLs")
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
