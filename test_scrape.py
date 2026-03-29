#!/usr/bin/env python3
"""Test scraping one article from each source to diagnose issues."""

import psycopg2
import trafilatura
from googlenewsdecoder import new_decoderv1

DB = "dbname=fog_of_war"

def test_source(cur, source_name):
    cur.execute("""
        SELECT id, title, google_url FROM articles 
        WHERE source_name = %s AND scrape_status = 'pending'
        AND published_at >= '2026-02-01' AND published_at < '2026-03-08'
        LIMIT 1
    """, (source_name,))
    row = cur.fetchone()
    if not row:
        print(f"  ⚠️  No pending articles found")
        return
    
    article_id, title, google_url = row
    print(f"  Title: {title[:70]}...")
    
    # Step 1: Decode URL
    try:
        result = new_decoderv1(google_url, interval=0.3)
        if result and result.get("status"):
            url = result["decoded_url"]
            print(f"  ✅ URL decoded: {url[:80]}...")
        else:
            print(f"  ❌ URL decode failed: {result}")
            return
    except Exception as e:
        print(f"  ❌ URL decode error: {e}")
        return
    
    # Step 2: Fetch
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            print(f"  ❌ fetch_url returned None (blocked/paywall?)")
            # Try with requests directly
            import requests
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            resp = requests.get(url, headers=headers, timeout=15)
            print(f"  📡 Direct request: status={resp.status_code}, length={len(resp.text)}")
            if resp.status_code == 200 and len(resp.text) > 1000:
                downloaded = resp.text
                print(f"  🔄 Trying trafilatura.extract on direct fetch...")
            else:
                return
        else:
            print(f"  ✅ Fetched: {len(downloaded)} chars")
    except Exception as e:
        print(f"  ❌ Fetch error: {e}")
        return
    
    # Step 3: Extract
    try:
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if text and len(text.strip()) > 100:
            print(f"  ✅ Extracted: {len(text)} chars, {len(text.split())} words")
            print(f"  📝 Preview: {text[:150]}...")
        elif text:
            print(f"  ⚠️  Too short: {len(text)} chars")
            print(f"  📝 Preview: {text[:150]}...")
        else:
            print(f"  ❌ Extract returned None")
            # Try alternative extraction
            from trafilatura import bare_extraction
            result = bare_extraction(downloaded)
            if result:
                print(f"  🔄 bare_extraction keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    except Exception as e:
        print(f"  ❌ Extract error: {e}")


def main():
    conn = psycopg2.connect(DB)
    cur = conn.cursor()
    
    sources = [
        "Reuters", "Bloomberg", "Fox News", "Al Jazeera", "AP News",
        "BBC", "The National (UAE)", "Financial Times", "The Guardian",
        "Al-Monitor", "Middle East Eye"
    ]
    
    for source in sources:
        print(f"\n{'='*50}")
        print(f"🔍 Testing: {source}")
        print(f"{'='*50}")
        test_source(cur, source)
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
