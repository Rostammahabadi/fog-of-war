# Fog of War

LLM evaluation pipeline for geopolitical forecasting, based on the paper [*When AI Navigates the Fog of War*](https://arxiv.org/abs/2603.16642).

How well can frontier AI models predict escalating events when they can only see what a real intelligence analyst would see at that moment? This pipeline answers that question by feeding temporally-gated intelligence briefings to LLMs and scoring their probabilistic forecasts against ground truth.

## Results

| Model | 1-MAE | Accuracy | Brier Score |
|-------|-------|----------|-------------|
| Claude Opus 4.6 | **0.850** | 93.2% | 0.077 |
| Gemini 3 Flash | 0.837 | 90.9% | 0.046 |
| Qwen 3.5-35B | 0.821 | 95.5% | 0.079 |
| Kimi K2.5 | 0.782 | 93.2% | 0.092 |
| Claude Sonnet 4.6 | 0.757 | 84.1% | 0.126 |
| GPT-5.4 | 0.701 | 86.4% | 0.116 |

Open `pipeline/output/results.html` in a browser for the full interactive dashboard.

## The Scenario

A simulated Middle East conflict escalating over 9 days (Feb 27 - Mar 7, 2026), divided into 11 temporal nodes:

| Node | Date | Event |
|------|------|-------|
| T0 | Feb 27 | Operation Epic Fury |
| T1 | Feb 28 | Israeli-US strikes on Iranian facilities |
| T2 | Feb 28 12:00 | Iranian retaliatory strikes |
| T3 | Mar 1 | Missiles toward British bases |
| T4 | Mar 1 12:00 | Oil refinery/tanker attacked |
| T5 | Mar 2 | Qatar halts energy production |
| T6 | Mar 2 12:00 | Natanz nuclear facility damaged |
| T7 | Mar 3 | US suggests citizen evacuation |
| T8 | Mar 3 12:00 | Nine countries involved; Israeli ground invasion |
| T9 | Mar 3 18:00 | Mojtaba Khamenei becomes leader |
| T10 | Mar 6 | Iranian apology to neighboring countries |

At each node, models receive only intelligence published **before** that timestamp. 42 binary questions test whether they can forecast what happens next.

## Setup

### Requirements

- Python 3.10+
- PostgreSQL (for news corpus layer only)
- [OpenRouter](https://openrouter.ai/) API key

### Install

```bash
python -m venv .venv
source .venv/bin/activate
cd pipeline
pip install -r requirements.txt
```

### Environment

```bash
export OPENROUTER_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"  # optional, for direct OpenAI access
```

## Usage

All pipeline commands run from the `pipeline/` directory:

```bash
cd pipeline
```

### Full pipeline (recommended)

Fetches data, builds briefings, runs inference across all models and nodes, then evaluates:

```bash
python main.py full --evaluate
```

### Step by step

```bash
# 1. Fetch data sources (news, financial, GDELT) into local cache
python main.py fetch

# 2. Build temporally-gated briefings for specific nodes
python main.py build --node T3

# 3. Run inference on all or specific nodes/models
python main.py run --nodes all
python main.py run --model anthropic/claude-sonnet-4-6 --nodes T1,T2,T3

# 4. Evaluate predictions against ground truth
python main.py evaluate
```

### Results

After evaluation, results are saved to `pipeline/output/`:
- `evaluation_<timestamp>.json` - Raw evaluation data
- `evaluation_<timestamp>.md` - Markdown report
- `results.html` - Interactive dashboard

## Models

The pipeline evaluates these models via OpenRouter:

- `anthropic/claude-opus-4-6`
- `anthropic/claude-sonnet-4-6`
- `openai/gpt-5.4`
- `google/gemini-3.1-flash-lite-preview`
- `qwen/qwen3.5-35b-a3b`
- `moonshotai/kimi-k2.5`

To add a model, edit the `MODELS` list in `pipeline/config.py`.

## News Corpus

The pipeline needs news article data to build intelligence briefings. You have two options:

### Option A: Pipeline fetch (simpler)

The pipeline's `fetch` command pulls from public APIs (GDELT, yfinance) directly:

```bash
cd pipeline
python main.py fetch
```

This gives you GDELT articles and financial data, which is enough to run the evaluation.

### Option B: Full scraping pipeline (richer corpus)

For a more complete corpus from 12+ news outlets, use the root-level scripts. This requires PostgreSQL.

```bash
# Set up the database
createdb fog_of_war
psql fog_of_war < schema.sql

# Collect article URLs from multiple sources
python collect_google_news.py
python collect_gdelt.py
python collect_direct.py

# Scrape full text
python scrape_articles.py
```

Note: Some sources (Bloomberg, FT) have paywalls and may return limited content.

## Architecture

```
fog-of-war/
├── pipeline/                  # Evaluation pipeline (self-contained)
│   ├── main.py                # CLI entrypoint
│   ├── config.py              # Models, nodes, API config
│   ├── data_fetcher.py        # Multi-source data collection
│   ├── context_builder.py     # Temporally-gated briefing assembly
│   ├── run_inference.py       # LLM inference via OpenRouter
│   ├── probability_extractor.py  # LLM-based probability extraction
│   ├── evaluator.py           # Scoring (Brier, 1-MAE, accuracy)
│   ├── ground_truth.json      # 42 binary questions with labels
│   └── output/                # Generated results (gitignored except results.html)
├── collect_*.py               # News URL collection scripts
├── scrape_*.py                # Article scraping scripts
├── schema.sql                 # PostgreSQL schema
└── data/                      # Scraped article data (gitignored)
```

## Evaluation Methodology

- **1-MAE** (paper's primary metric): 1 minus mean absolute error between predicted probability and binary outcome. Higher is better. Scale 0-1.
- **Brier Score**: Mean squared error between predicted probability and outcome. Lower is better. Scale 0-1.
- **Accuracy**: Binary prediction correctness (predicted probability > 0.5 = event occurs).
- **Probability extraction**: LLM-based extraction using Claude Haiku 4.5. Explicit percentages are extracted directly; qualitative language ("very likely", "unlikely") is mapped to calibrated probability values.

## Citation

```bibtex
@article{fogofwar2026,
  title={When AI Navigates the Fog of War},
  year={2026},
  url={https://arxiv.org/abs/2603.16642}
}
```

## License

MIT License. See [LICENSE](LICENSE).
