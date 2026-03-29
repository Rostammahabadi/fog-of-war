#!/usr/bin/env python3
"""
Scrape Financial Times articles using Playwright with the user's Chrome session.

Usage:
    # First run: launches visible browser so you can verify you're logged in
    python scrape_ft_playwright.py

    # After confirming login works:
    python scrape_ft_playwright.py --headless
"""

import argparse
import psycopg2
import time
import signal
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

DB = "dbname=fog_of_war"
CHROME_USER_DATA = str(Path.home() / "Library/Application Support/Google/Chrome")
RATE_LIMIT_SECONDS = 2  # be polite to FT

shutdown = False
def handle_signal(signum, frame):
    global shutdown
    print("\nShutting down after current article...")
    shutdown = True

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def extract_article_text(page) -> str | None:
    """Extract article body text from an FT article page."""
    selectors = [
        "article [data-trackable='article-body']",
        "article .article-body",
        "article .body-content",
        "#article-body",
        "article",
    ]
    for selector in selectors:
        el = page.query_selector(selector)
        if el:
            text = el.inner_text().strip()
            if len(text) > 200:
                return text
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode (use after confirming login works)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max articles to scrape (0 = all)")
    args = parser.parse_args()

    conn = psycopg2.connect(DB)
    cur = conn.cursor()

    # Count FT articles that need scraping
    cur.execute("""
        SELECT COUNT(*) FROM articles
        WHERE source_domain = 'ft.com'
          AND (scrape_status IN ('pending', 'failed', 'paywall'))
          AND published_at >= '2026-02-01' AND published_at < '2026-03-08'
    """)
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM articles
        WHERE source_domain = 'ft.com' AND scrape_status = 'success'
          AND word_count > 200
    """)
    already_done = cur.fetchone()[0]

    print(f"{'=' * 60}")
    print(f"FT Playwright Scraper")
    print(f"Already scraped (>200 words): {already_done}")
    print(f"To scrape: {total}")
    print(f"Mode: {'headless' if args.headless else 'visible (verify login)'}")
    print(f"{'=' * 60}")

    if total == 0:
        print("Nothing to scrape.")
        return

    with sync_playwright() as p:
        # Launch Chromium with persistent context (uses your Chrome cookies/session)
        context = p.chromium.launch_persistent_context(
            user_data_dir=CHROME_USER_DATA + "/Default",
            headless=args.headless,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.new_page()

        # Quick login check
        print("Checking FT login status...")
        page.goto("https://www.ft.com/", wait_until="domcontentloaded", timeout=15000)
        time.sleep(2)

        # Look for signs of being logged in
        logged_in = page.query_selector("[data-trackable='Sign Out']") or \
                    page.query_selector("a[href*='logout']") or \
                    page.query_selector("[data-trackable='myft']")

        if not logged_in and not args.headless:
            print("\nNot logged in. Please log in to FT in the browser window.")
            print("Press Enter here once you're logged in...")
            input()
        elif not logged_in:
            print("ERROR: Not logged in and running headless. Run without --headless first.")
            context.close()
            return

        print("Scraping FT articles...\n")

        stats = {"success": 0, "failed": 0, "skipped": 0}
        processed = 0
        start_time = time.time()
        limit = args.limit if args.limit > 0 else total

        while not shutdown and processed < limit:
            cur.execute("""
                SELECT id, title, url FROM articles
                WHERE source_domain = 'ft.com'
                  AND (scrape_status IN ('pending', 'failed', 'paywall'))
                  AND published_at >= '2026-02-01' AND published_at < '2026-03-08'
                ORDER BY
                  CASE WHEN url LIKE '%%ft.com%%' THEN 0 ELSE 1 END,
                  published_at DESC
                LIMIT 20
            """)
            batch = cur.fetchall()
            if not batch:
                break

            for article_id, title, url in batch:
                if shutdown or processed >= limit:
                    break

                processed += 1
                short_title = title[:55] + "..." if len(title) > 55 else title
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = min(limit, total) - processed
                eta_min = remaining / rate / 60 if rate > 0 else 0

                print(f"\r[{processed}/{min(limit, total)}] (~{eta_min:.0f}m left) {short_title}    ",
                      end="", flush=True)

                try:
                    # If no direct ft.com URL, search FT by title
                    if not url or "ft.com" not in url:
                        search_q = title.replace(" - Financial Times", "").strip()[:80]
                        page.goto(f"https://www.ft.com/search?q={search_q}&sort=date",
                                  wait_until="domcontentloaded", timeout=20000)
                        time.sleep(3)
                        link = page.query_selector("a[data-trackable='heading-link']") or \
                               page.query_selector("a[href*='/content/']")
                        if link:
                            href = link.get_attribute("href")
                            if href:
                                url = href if href.startswith("http") else "https://www.ft.com" + href
                                cur.execute("UPDATE articles SET url = %s WHERE id = %s",
                                            (url, article_id))
                            else:
                                raise Exception("no href on search result")
                        else:
                            raise Exception("no search results")

                    page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    time.sleep(3)  # let JS render

                    text = extract_article_text(page)

                    if text and len(text) > 200:
                        word_count = len(text.split())
                        cur.execute("""
                            UPDATE articles
                            SET full_text = %s, text_length = %s, word_count = %s,
                                scrape_status = 'success', scrape_error = NULL, scraped_at = NOW()
                            WHERE id = %s
                        """, (text, len(text), word_count, article_id))
                        stats["success"] += 1
                    else:
                        cur.execute("""
                            UPDATE articles
                            SET scrape_status = 'failed',
                                scrape_error = 'playwright_extract_failed',
                                scraped_at = NOW()
                            WHERE id = %s
                        """, (article_id,))
                        stats["failed"] += 1

                except Exception as e:
                    err_msg = str(e)[:200]
                    cur.execute("""
                        UPDATE articles
                        SET scrape_status = 'failed',
                            scrape_error = %s,
                            scraped_at = NOW()
                        WHERE id = %s
                    """, (f"playwright_error: {err_msg}", article_id))
                    stats["failed"] += 1

                conn.commit()
                time.sleep(RATE_LIMIT_SECONDS)

            # Batch summary
            print(f"\n  OK: {stats['success']} | Failed: {stats['failed']} | "
                  f"{(time.time() - start_time)/60:.1f}m elapsed")

        context.close()

    # Final report
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"  Scraped: {stats['success']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"{'=' * 60}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
