#!/usr/bin/env python3
"""
Scrape articles with per-source strategies.
- Most sources: trafilatura direct fetch
- Reuters: requests session with browser headers (mixed success, retry with session)  
- Bloomberg: requests session (mostly paywalled, mark as paywall)
- Financial Times: mark paywall snippets
"""

import psycopg2
import requests
import trafilatura
import time
import signal
from datetime import datetime
from googlenewsdecoder import new_decoderv1

DB = "dbname=fog_of_war"
BATCH_SIZE = 50

shutdown = False
def handle_signal(signum, frame):
    global shutdown
    print("\n⚠️  Shutting down after current article...")
    shutdown = True

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# Persistent session for sites that need cookies/headers
browser_session = requests.Session()
browser_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
})


def decode_google_url(google_url):
    if not google_url:
        return None
    if "news.google.com" not in google_url:
        return google_url
    try:
        result = new_decoderv1(google_url, interval=0.3)
        if result and result.get("status"):
            return result["decoded_url"]
    except:
        pass
    return None


def scrape_with_trafilatura(url):
    """Default scraping: trafilatura's built-in fetcher + extractor."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None, None, "fetch_failed"
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return downloaded, text, None


def scrape_with_session(url):
    """Scrape using browser-like requests session (for Reuters etc)."""
    try:
        r = browser_session.get(url, timeout=20)
        if r.status_code == 401:
            return None, None, "auth_required"
        if r.status_code == 403:
            return None, None, "forbidden"
        if r.status_code == 404:
            return None, None, "not_found"
        if r.status_code != 200:
            return None, None, f"http_{r.status_code}"
        
        text = trafilatura.extract(r.text, include_comments=False, include_tables=False)
        return r.text, text, None
    except requests.Timeout:
        return None, None, "timeout"
    except Exception as e:
        return None, None, f"error: {str(e)[:200]}"


def scrape_article(url, source_domain):
    """Route to appropriate scraping strategy based on source."""
    if not url:
        return None, "no_url"
    
    # Strategy by source
    if source_domain in ("reuters.com", "bloomberg.com", "ft.com"):
        # Try with browser session first (handles cookies/JS redirects better)
        html, text, err = scrape_with_session(url)
        if err:
            # Fallback to trafilatura
            html2, text2, err2 = scrape_with_trafilatura(url)
            if text2 and len(text2) > 100:
                return text2, "success"
            return None, err or err2 or "extract_failed"
    else:
        # Default: trafilatura
        html, text, err = scrape_with_trafilatura(url)
        if err:
            # Fallback to session
            html2, text2, err2 = scrape_with_session(url)
            if text2 and len(text2) > 100:
                return text2, "success"
            return None, err or "fetch_failed"
    
    if not text:
        return None, "extract_failed"
    
    text = text.strip()
    
    # Check for paywall indicators
    paywall_phrases = ["subscribe to unlock", "subscribe to read", "try unlimited access",
                       "sign in to read", "premium content", "for subscribers only"]
    text_lower = text.lower()
    if any(p in text_lower for p in paywall_phrases) and len(text) < 500:
        return text, "paywall"
    
    if len(text) < 100:
        return text, "too_short"
    
    return text, "success"


def main():
    conn = psycopg2.connect(DB)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM articles WHERE scrape_status = 'pending'")
    total_pending = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM articles WHERE scrape_status = 'success'")
    already_done = cur.fetchone()[0]
    
    print(f"{'=' * 60}")
    print(f"Article Scraper v2 (per-source strategies)")
    print(f"Already scraped: {already_done}")
    print(f"Pending: {total_pending}")
    print(f"{'=' * 60}")
    
    processed = 0
    stats = {"success": 0, "failed": 0, "paywall": 0, "decode_fail": 0}
    start_time = time.time()
    
    while not shutdown:
        cur.execute("""
            SELECT id, title, google_url, url, source_domain
            FROM articles
            WHERE scrape_status = 'pending'
            ORDER BY 
                CASE WHEN published_at >= '2026-02-01' AND published_at < '2026-03-08' THEN 0 ELSE 1 END,
                published_at DESC
            LIMIT %s
        """, (BATCH_SIZE,))
        
        batch = cur.fetchall()
        if not batch:
            print("\n✅ All articles processed!")
            break
        
        for article_id, title, google_url, existing_url, source_domain in batch:
            if shutdown:
                break
            
            processed += 1
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total_pending - processed
            eta_min = remaining / rate / 60 if rate > 0 else 0
            
            short_title = title[:50] + "..." if len(title) > 50 else title
            pct = processed / total_pending * 100
            print(f"\r[{processed}/{total_pending} {pct:.0f}%] (~{eta_min:.0f}m) {short_title}    ", end="", flush=True)
            
            # Decode URL
            url = existing_url
            if not url or "news.google.com" in (url or ""):
                url = decode_google_url(google_url)
                if url:
                    cur.execute("UPDATE articles SET url = %s WHERE id = %s", (url, article_id))
                else:
                    stats["decode_fail"] += 1
                    cur.execute("""
                        UPDATE articles SET scrape_status = 'failed', scrape_error = 'url_decode_failed', scraped_at = NOW()
                        WHERE id = %s
                    """, (article_id,))
                    conn.commit()
                    continue
            
            # Scrape
            text, status = scrape_article(url, source_domain)
            
            word_count = len(text.split()) if text else 0
            text_length = len(text) if text else 0
            
            if status == "success":
                stats["success"] += 1
                cur.execute("""
                    UPDATE articles 
                    SET full_text = %s, text_length = %s, word_count = %s,
                        scrape_status = 'success', scraped_at = NOW()
                    WHERE id = %s
                """, (text, text_length, word_count, article_id))
            elif status == "paywall":
                stats["paywall"] += 1
                cur.execute("""
                    UPDATE articles 
                    SET full_text = %s, text_length = %s, word_count = %s,
                        scrape_status = 'paywall', scrape_error = %s, scraped_at = NOW()
                    WHERE id = %s
                """, (text, text_length, word_count, status, article_id))
            else:
                stats["failed"] += 1
                cur.execute("""
                    UPDATE articles 
                    SET scrape_status = 'failed', scrape_error = %s, scraped_at = NOW()
                    WHERE id = %s
                """, (status, article_id))
            
            conn.commit()
            time.sleep(0.3)
        
        # Batch summary
        elapsed = time.time() - start_time
        print(f"\n  📊 {processed} done | ✅ {stats['success']} ok | ❌ {stats['failed']} fail | 🔒 {stats['paywall']} paywall | ⏱️ {elapsed/60:.1f}m")
    
    # Final report
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DONE — {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    cur.execute("""
        SELECT source_name,
            COUNT(*) FILTER (WHERE scrape_status = 'success') as ok,
            COUNT(*) FILTER (WHERE scrape_status = 'paywall') as paywall,
            COUNT(*) FILTER (WHERE scrape_status = 'failed') as fail,
            COUNT(*) FILTER (WHERE scrape_status = 'pending') as pending,
            ROUND(AVG(word_count) FILTER (WHERE scrape_status = 'success')) as avg_words
        FROM articles GROUP BY source_name ORDER BY ok DESC
    """)
    print(f"\nPer-source breakdown:")
    for row in cur.fetchall():
        print(f"  {row[0]}: ✅{row[1]} 🔒{row[2]} ❌{row[3]} ⏳{row[4]} (avg {row[5] or 0} words)")
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
