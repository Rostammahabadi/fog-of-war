#!/usr/bin/env python3
"""Load collected Google News articles into PostgreSQL."""

import json
import psycopg2
from datetime import datetime
from email.utils import parsedate_to_datetime

DB = "dbname=fog_of_war"

def parse_pub_date(date_str):
    """Parse RFC 2822 date from Google News RSS."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except:
        try:
            # Try ISO format
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except:
            return None

def main():
    with open("data/google_news_articles.json") as f:
        data = json.load(f)
    
    articles = data["articles"]
    print(f"Loading {len(articles)} articles into PostgreSQL...")
    
    conn = psycopg2.connect(DB)
    cur = conn.cursor()
    
    inserted = 0
    skipped = 0
    errors = 0
    
    for a in articles:
        title = a.get("title", "").strip()
        if not title:
            skipped += 1
            continue
        
        title_normalized = title.lower().strip()
        source_domain = a.get("_source_domain", "")
        source_name = a.get("_source_name", "Other")
        
        if not source_domain:
            skipped += 1
            continue
        
        published_at = parse_pub_date(a.get("pub_date"))
        google_url = a.get("google_url", "")
        query_source = a.get("_query", "")
        
        try:
            cur.execute("""
                INSERT INTO articles (title, title_normalized, google_url, source_domain, source_name, 
                                      published_at, query_source, raw_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (title_normalized, source_domain) DO NOTHING
            """, (
                title, title_normalized, google_url, source_domain, source_name,
                published_at, query_source, json.dumps(a)
            ))
            if cur.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            conn.rollback()
            continue
    
    conn.commit()
    
    # Stats
    cur.execute("SELECT COUNT(*) FROM articles")
    total = cur.fetchone()[0]
    
    cur.execute("SELECT * FROM corpus_stats")
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    
    print(f"\nInserted: {inserted}, Skipped (dupes): {skipped}, Errors: {errors}")
    print(f"Total in DB: {total}")
    print(f"\nCorpus stats:")
    for row in rows:
        stats = dict(zip(cols, row))
        print(f"  {stats['source_name']}: {stats['total_articles']} articles, "
              f"earliest: {stats['earliest']}, latest: {stats['latest']}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
