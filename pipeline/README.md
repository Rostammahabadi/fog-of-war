# Fog of War LLM Evaluation Pipeline

A complete Python-based LLM evaluation pipeline for geopolitical forecasting, inspired by the paper ["When AI Navigates the Fog of War"](https://arxiv.org/html/2603.16642v1).

## 🎯 Key Features

- **Strict Temporal Gating**: Ensures "fog of war" compliance - never includes data timestamped after target date T
- **Multi-Source Data Integration**: Economic signals, tactical OSINT, sentiment analysis, ground truth events, and news articles
- **Multi-Model LLM Support**: Works with OpenRouter and OpenAI APIs for Claude, GPT, Gemini, and other models
- **Comprehensive Evaluation**: Accuracy, calibration, Brier scores, and thematic analysis
- **Production Ready**: Full error handling, rate limiting, retry logic, and logging

## 🏗️ Architecture

```
├── data_fetcher.py      # Multi-source data ingestion with temporal gating
├── context_builder.py   # Intelligence briefing generation  
├── prompt_builder.py    # LLM prompt construction
├── run_inference.py     # Multi-model LLM inference with rate limiting
├── evaluator.py         # Prediction evaluation and metrics
├── main.py             # CLI orchestrator
├── config.py           # Central configuration
└── requirements.txt    # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd ~/fog-of-war/pipeline
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENAI_API_KEY="your-openai-key"  # Optional
```

### 3. Run Full Pipeline

```bash
# Complete end-to-end pipeline
python main.py full --evaluate

# Or run step by step:
python main.py fetch                    # Fetch data sources
python main.py run --nodes all          # Run inference on all temporal nodes  
python main.py evaluate                 # Evaluate against ground truth
```

## 📊 Temporal Nodes

The pipeline evaluates 11 temporal nodes from the paper:

| Node | Date | Event |
|------|------|--------|
| T0 | Feb 27 | Operation Epic Fury |
| T1 | Feb 28 00:00 | Israeli-US Strikes |
| T2 | Feb 28 12:00 | Iranian Strikes |
| T3 | Mar 1 00:00 | Missiles toward British Bases |
| T4 | Mar 1 12:00 | Oil Refinery/Tanker Attacked |
| T5 | Mar 2 00:00 | Qatar Halts Energy Production |
| T6 | Mar 2 12:00 | Natanz Nuclear Facility Damaged |
| T7 | Mar 3 00:00 | US Suggests Citizen Evacuation |
| T8 | Mar 3 12:00 | Nine Countries; Ground Invasion |
| T9 | Mar 3 18:00 | Mojtaba Khamenei Becomes Leader |
| T10 | Mar 7 00:00 | Late Escalation Node |

## 📚 Usage Examples

### Fetch Data
```bash
# Fetch all data sources for date range
python main.py fetch --start-date 2026-02-01 --end-date 2026-03-31

# Use custom output file
python main.py fetch --output my_data_cache.json
```

### Build Intelligence Briefing
```bash
# Build briefing for specific node
python main.py build --node T3

# Output as markdown
python main.py build --node T3 --format markdown

# Verbose output with summary
python main.py build --node T3 --verbose
```

### Run LLM Inference
```bash
# Single model on specific nodes
python main.py run --model anthropic/claude-3.5-sonnet --nodes T1,T2,T3

# Multiple models on all nodes
python main.py run --model anthropic/claude-3.5-sonnet --model openai/gpt-4o --nodes all

# Custom temperature
python main.py run --temperature 0.3 --nodes all
```

### Evaluate Results
```bash
# Evaluate most recent results
python main.py evaluate

# Evaluate specific results file
python main.py evaluate --inference-results results.json

# Use custom ground truth
python main.py evaluate --ground-truth my_ground_truth.json

# Verbose evaluation summary
python main.py evaluate --verbose
```

### Full Pipeline Options
```bash
# Complete pipeline with fresh data
python main.py full --fetch-fresh --evaluate

# Specific models with custom temperature
python main.py full --model anthropic/claude-3.5-sonnet --temperature 0.5 --evaluate
```

## 📁 Data Sources

### Economic Signals
- **Brent Crude Oil (BZ=F)**: Daily close prices via yfinance
- **Tel Aviv 35 Index (TA35.TA)**: Daily close prices via yfinance

### Tactical Intelligence  
- **Military Aircraft Counts**: ADS-B Exchange API (with realistic stub for testing)
- **Regional Coverage**: Middle East theater (25°N-42°N, 25°E-65°E)

### Sentiment Analysis
- **GDELT Project API**: Daily tone and event counts for regional conflicts
- **Search Terms**: Iran, Israel, Middle East conflict coverage

### Ground Truth Events
- **UCDP GED**: Uppsala Conflict Data Program Georeferenced Event Dataset
- **ACLED**: Armed Conflict Location & Event Data (with synthetic data for testing)

### News Articles
- **Google News**: 8,853 articles from existing collection
- **GDELT Articles**: 1,041 articles from existing collection

## 🔧 Configuration

Edit `config.py` to customize:

- **Date ranges**: `START_DATE`, `END_DATE`
- **Temporal nodes**: `TEMPORAL_NODES` dictionary
- **Models**: `MODELS` list  
- **API endpoints**: `OPENROUTER_BASE_URL`, `GDELT_BASE_URL`
- **Rate limiting**: `RATE_LIMIT_DELAY`, `MAX_RETRIES`
- **Regional bounds**: `MIDDLE_EAST_COUNTRIES`, `ADSB_REGION_BOUNDS`

## 📊 Output Files

The pipeline generates structured outputs in `~/fog-of-war/pipeline/output/`:

```
output/
├── pipeline_data_cache.json           # Cached data from all sources
├── briefing_T3_20260301_0000.json     # Intelligence briefing for node T3
├── briefing_T3_20260301_0000.md       # Markdown version of briefing
├── node_T3_results.json               # LLM inference results for T3
├── sequence_temporal_sequence_*.json  # Complete temporal sequence results
├── evaluation_*.json                  # Evaluation results with metrics
├── evaluation_*.md                    # Human-readable evaluation report
└── pipeline.log                       # Execution logs
```

## 🧪 Evaluation Metrics

### Accuracy Metrics
- **Binary Accuracy**: Correct binary predictions (event occurs/doesn't occur)
- **Brier Score**: Mean squared error between predicted probabilities and outcomes (lower = better)
- **Overall Success Rate**: Percentage of successful model inferences

### Calibration Metrics  
- **Expected Calibration Error (ECE)**: Average absolute difference between predicted probabilities and actual frequencies
- **Reliability Curve**: Calibration analysis across probability bins
- **Confidence vs. Accuracy**: Relationship between prediction confidence and actual accuracy

### Qualitative Assessment
- **Evidence Usage**: How well responses reference specific data points
- **Logical Structure**: Presence of reasoning connectors and structured analysis  
- **Specificity**: Use of concrete predictions vs. vague statements
- **Uncertainty Handling**: Appropriate acknowledgment of limitations and confidence levels

### Thematic Analysis
Performance breakdown by evaluation themes:
- **Initial Outbreak**: Early conflict detection
- **Threshold Crossings**: Major escalation events
- **Economic Shockwaves**: Market and energy disruptions  
- **Political Signaling**: Diplomatic and leadership changes

## ⚠️ Critical Implementation Notes

### Fog of War Compliance
The **most critical** feature is strict temporal gating in `context_builder.py`:

```python
def _apply_temporal_gating(self, target_date: datetime) -> Dict[str, Any]:
    """CRITICAL: Only include data timestamped <= target_date"""
    mask = pd.to_datetime(df['timestamp']) <= target_date
    filtered_df = df[mask].copy()
```

This ensures the LLM never sees "future" data, maintaining the fog-of-war constraint.

### Error Handling
- **API Rate Limits**: Automatic retry with exponential backoff
- **Network Failures**: Graceful degradation with cached data fallbacks  
- **Missing Data**: Continue pipeline execution with warnings
- **Model Failures**: Record errors but continue with other models

### Production Considerations
- **API Keys**: Store securely in environment variables
- **Data Sources**: Replace synthetic data with real APIs for production
- **Monitoring**: Enable comprehensive logging for debugging
- **Scaling**: Consider async processing for multiple models

## 🤝 Contributing

This pipeline is designed for research and can be extended:

1. **Add Data Sources**: Implement new fetchers in `data_fetcher.py`
2. **Custom Evaluations**: Extend `evaluator.py` with domain-specific metrics
3. **New Models**: Add model configurations to `config.py` and update `run_inference.py`
4. **Advanced Prompting**: Enhance prompt engineering in `prompt_builder.py`

## 📜 License & Citation

Inspired by "When AI Navigates the Fog of War" - please cite the original paper when using this pipeline for research.

---

**Built with ❤️ for advancing AI safety and geopolitical forecasting research.**