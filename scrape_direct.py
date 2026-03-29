#!/usr/bin/env python3
"""
Skip Google News URL decoding entirely.
Strategy: Use each outlet's own search/sitemap to find article URLs directly,
then match to our DB records by title similarity.
For already-decoded URLs (945 success + those with real URLs), just scrape.
For the rest, search the outlet directly.
"""

import psycopg2
import requests
import trafilatura
import time
import signal
import re
import json
from datetime import datetime
from difflib import SequenceMatcher

DB = "dbname=fog_of_war"
shutdown = False

def handle_signal(signum, frame):
    global shutdown
    print("\n⚠️  Shutting down...")
    shutdown = True

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

session = requests.Session()
session.headers.update(HEADERS)


def search_outlet(source_domain, title):
    """Search the outlet's own site for an article by title."""
    # Clean title for search
    clean_title = re.sub(r'\s*-\s*(Reuters|Bloomberg|BBC|Fox News|AP News|The Guardian|Financial Times|Al Jazeera|Al-Monitor|Middle East Eye|thenationalnews\.com).*$', '', title, flags=re.IGNORECASE).strip()
    search_query = clean_title[:80]  # Trim long titles
    
    search_configs = {
        'aljazeera.com': f'https://www.aljazeera.com/search/{requests.utils.quote(search_query)}',
        'reuters.com': f'https://www.reuters.com/site-search/?query={requests.utils.quote(search_query)}',
        'bbc.com': f'https://www.bbc.co.uk/search?q={requests.utils.quote(search_query)}',
        'bbc.co.uk': f'https://www.bbc.co.uk/search?q={requests.utils.quote(search_query)}',
        'theguardian.com': f'https://www.theguardian.com/search?q={requests.utils.quote(search_query)}',
        'apnews.com': f'https://apnews.com/search#{requests.utils.quote(search_query)}',
        'foxnews.com': f'https://www.foxnews.com/search-results/search?q={requests.utils.quote(search_query)}',
        'ft.com': f'https://www.ft.com/search?q={requests.utils.quote(search_query)}',
        'middleeasteye.net': f'https://www.middleeasteye.net/search?search={requests.utils.quote(search_query)}',
        'al-monitor.com': f'https://www.al-monitor.com/search?search_api_fulltext={requests.utils.quote(search_query)}',
        'thenationalnews.com': f'https://www.thenationalnews.com/search/?q={requests.utils.quote(search_query)}',
        'bloomberg.com': None,  # Bloomberg search requires JS, skip
    }
    
    search_url = search_configs.get(source_domain)
    if not search_url:
        return None
    
    try:
        r = session.get(search_url, timeout=15)
        if r.status_code != 200:
            return None
        
        # Find article URLs in the search results page
        domain_pattern = source_domain.replace('.', r'\.')
        article_urls = re.findall(
            rf'href=["\']?(https?://(?:www\.)?{domain_pattern}/[^"\'\s>]+)["\'\s>]',
            r.text
        )
        
        # Filter to likely article URLs (not search/tag/category pages)
        article_urls = [u for u in article_urls if any(seg in u for seg in 
            ['/news/', '/world/', '/article', '/originals/', '/opinion/', '/business/', 
             '/politics/', '/stories/', '/pictures/', '/graphics/', '/search', '/video/'])]
        
        # Deduplicate
        article_urls = list(dict.fromkeys(article_urls))
        
        return article_urls[:10] if article_urls else None
    except:
        return None


def scrape_text(url):
    """Scrape full article text."""
    if not url:
        return None, "no_url"
    try:
        # Try trafilatura first
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text.strip()) > 100:
                return text.strip(), "success"
        
        # Fallback to requests
        r = session.get(url, timeout=20)
        if r.status_code == 200:
            text = trafilatura.extract(r.text, include_comments=False, include_tables=False)
            if text and len(text.strip()) > 100:
                return text.strip(), "success"
            return None, "extract_failed"
        else:
            return None, f"http_{r.status_code}"
    except Exception as e:
        return None, f"error: {str(e)[:100]}"


def retry_with_v1_decoder(google_url):
    """Try the v1 decoder with a long delay (may work if rate limit has cooled)."""
    try:
        from googlenewsdecoder import new_decoderv1
        result = new_decoderv1(google_url, interval=5)
        if result and result.get("status"):
            return result["decoded_url"]
    except:
        pass
    return None


def main():
    conn = psycopg2.connect(DB)
    cur = conn.cursor()
    
    # Phase 1: Retry all url_decode_failed with the v1 decoder (with longer delays)
    cur.execute("""
        SELECT COUNT(*) FROM articles 
        WHERE scrape_status = 'failed' AND scrape_error = 'url_decode_failed'
    """)
    decode_failed = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM articles WHERE scrape_status = 'success'")
    already_ok = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM articles WHERE scrape_status = 'pending'")
    still_pending = cur.fetchone()[0]
    
    print(f"{'=' * 60}")
    print(f"Direct Scraper")  
    print(f"Already success: {already_ok}")
    print(f"URL decode failed (to retry): {decode_failed}")
    print(f"Still pending: {still_pending}")
    print(f"{'=' * 60}")
    
    # First: process any that are still pending (never attempted)
    print(f"\n📡 Phase 1: Processing pending articles with v1 decoder (slow, with delays)...")
    
    # Reset url_decode_failed to pending so we can retry
    cur.execute("""
        UPDATE articles SET scrape_status = 'pending', scrape_error = NULL 
        WHERE scrape_status = 'failed' AND scrape_error = 'url_decode_failed'
    """)
    conn.commit()
    print(f"  Reset {decode_failed} url_decode_failed articles to pending")
    
    processed = 0
    success = 0
    failed = 0
    start_time = time.time()
    
    while not shutdown:
        cur.execute("""
            SELECT id, title, google_url, url, source_domain, source_name
            FROM articles
            WHERE scrape_status = 'pending'
            ORDER BY published_at DESC
            LIMIT 20
        """)
        
        batch = cur.fetchall()
        if not batch:
            print("\n✅ All done!")
            break
        
        for article_id, title, google_url, existing_url, source_domain, source_name in batch:
            if shutdown:
                break
            
            processed += 1
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            
            short_title = title[:50] + "..." if len(title) > 50 else title
            print(f"\r[{processed}] ✅{success} ❌{failed} ({rate:.1f}/s) {short_title}    ", end="", flush=True)
            
            url = existing_url
            
            # Try to decode Google URL
            if not url or "news.google.com" in (url or ""):
                url = retry_with_v1_decoder(google_url)
                if url:
                    cur.execute("UPDATE articles SET url = %s WHERE id = %s", (url, article_id))
                else:
                    failed += 1
                    cur.execute("""
                        UPDATE articles SET scrape_status = 'failed', scrape_error = 'url_decode_failed_v2', scraped_at = NOW()
                        WHERE id = %s
                    """, (article_id,))
                    conn.commit()
                    time.sleep(5)  # Back off when decode fails
                    continue
            
            # Scrape
            text, status = scrape_text(url)
            word_count = len(text.split()) if text else 0
            text_length = len(text) if text else 0
            
            if status == "success":
                success += 1
                cur.execute("""
                    UPDATE articles 
                    SET full_text = %s, text_length = %s, word_count = %s,
                        scrape_status = 'success', scraped_at = NOW()
                    WHERE id = %s
                """, (text, text_length, word_count, article_id))
            else:
                failed += 1
                cur.execute("""
                    UPDATE articles SET scrape_status = 'failed', scrape_error = %s, scraped_at = NOW()
                    WHERE id = %s
                """, (status, article_id))
            
            conn.commit()
            time.sleep(1)
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n\n{'=' * 60}")
    print(f"COMPLETE — {elapsed/60:.1f} minutes")
    print(f"Processed: {processed}, Success: {success}, Failed: {failed}")
    
    cur.execute("""
        SELECT source_name,
            COUNT(*) FILTER (WHERE scrape_status = 'success') as ok,
            COUNT(*) FILTER (WHERE scrape_status = 'failed') as fail,
            COUNT(*) FILTER (WHERE scrape_status = 'pending') as pending,
            ROUND(AVG(word_count) FILTER (WHERE scrape_status = 'success')) as avg_words
        FROM articles GROUP BY source_name ORDER BY ok DESC
    """)
    print(f"\nFinal stats:")
    total_ok = 0
    for row in cur.fetchall():
        total_ok += (row[1] or 0)
        print(f"  {row[0]}: ✅{row[1]} ❌{row[2]} ⏳{row[3]} (avg {row[4] or 0} words)")
    print(f"\nTotal success: {total_ok}")
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
