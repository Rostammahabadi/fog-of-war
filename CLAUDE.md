# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fog of War is an LLM evaluation pipeline for geopolitical forecasting, based on the paper "When AI Navigates the Fog of War". It evaluates how well LLMs can predict escalating events given temporally-gated intelligence briefings across 11 temporal nodes (T0-T10) representing a simulated Middle East conflict scenario (Feb 27 - Mar 7, 2026).

## Architecture

There are two layers:

1. **News corpus layer** (root-level scripts) - Collects and scrapes news articles into a PostgreSQL database (`fog_of_war`). Scripts connect via `psycopg2` with `DB = "dbname=fog_of_war"`.
   - `collect_google_news.py` / `collect_gdelt.py` / `collect_direct.py` - Article URL collection from different sources (Google News, GDELT API, direct RSS/sitemaps)
   - `scrape_articles.py` - Scrapes article full text using trafilatura, with per-source strategies (paywall handling for Bloomberg/FT/Reuters)
   - `load_articles.py` - Loads scraped articles into the database
   - `schema.sql` - PostgreSQL schema (articles, temporal_nodes, questions, llm_responses, context_packages)

2. **Evaluation pipeline** (`pipeline/`) - Self-contained Python pipeline that fetches multi-source data, builds temporally-gated intelligence briefings, runs LLM inference via OpenRouter/OpenAI, and evaluates predictions. All pipeline modules run from the `pipeline/` directory.

## Commands

### Pipeline (run from `pipeline/` directory)
```bash
cd pipeline
python main.py fetch                              # Fetch data sources into cache
python main.py build --node T3                     # Build briefing for a node
python main.py run --nodes all                     # Run inference on all nodes
python main.py run --model anthropic/claude-3.5-sonnet --nodes T1,T2,T3
python main.py evaluate                            # Evaluate most recent results
python main.py full --evaluate                     # End-to-end pipeline
```

### Environment setup
```bash
cd pipeline
pip install -r requirements.txt
export OPENROUTER_API_KEY="..."
export OPENAI_API_KEY="..."    # optional
```

### Scraping (run from repo root, requires PostgreSQL `fog_of_war` database)
```bash
python scrape_articles.py      # Scrape pending articles
python test_scrape.py          # Test one article per source
```

## Key Design Constraints

- **Temporal gating is the core invariant**: `context_builder.py` filters all data to `<= target_date`. Never include data timestamped after the target node's date. This is the "fog of war" constraint.
- **Pipeline modules use relative imports** from `pipeline/` - they must be run with `pipeline/` as the working directory.
- The pipeline uses a JSON file cache (`pipeline/output/pipeline_data_cache.json`) rather than the PostgreSQL database. The DB is only used by the root-level collection/scraping scripts.
- News article data lives in `data/` as JSON files (google_news_articles.json, gdelt_articles.json) and is loaded by the pipeline's data_fetcher.
- Configuration is centralized in `pipeline/config.py` (date ranges, temporal nodes, models, API endpoints, rate limits).
