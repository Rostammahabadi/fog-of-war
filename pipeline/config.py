"""
Configuration module for Fog of War LLM evaluation pipeline.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Any
from pathlib import Path

# Paths
PIPELINE_ROOT = Path(__file__).parent
OUTPUT_DIR = PIPELINE_ROOT / "output"
DATA_DIR = PIPELINE_ROOT.parent / "data"

# Date ranges
START_DATE = datetime(2026, 2, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 3, 31, tzinfo=timezone.utc)

# Temporal nodes from the paper
TEMPORAL_NODES = {
    "T0": datetime(2026, 2, 27, 0, 0, tzinfo=timezone.utc),   # Operation Epic Fury
    "T1": datetime(2026, 2, 28, 0, 0, tzinfo=timezone.utc),   # Israeli-US Strikes
    "T2": datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc),  # Iranian Strikes
    "T3": datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc),    # Missiles toward British Bases
    "T4": datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),   # Oil Refinery/Tanker Attacked
    "T5": datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc),    # Qatar Halts Energy Production
    "T6": datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc),   # Natanz Nuclear Facility Damaged
    "T7": datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc),    # US Suggests Citizen Evacuation
    "T8": datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc),   # Nine Countries; Ground Invasion
    "T9": datetime(2026, 3, 3, 18, 0, tzinfo=timezone.utc),   # Mojtaba Khamenei Becomes Leader
    "T10": datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc),   # Late Escalation Node
}

# Node descriptions for ground truth events
NODE_DESCRIPTIONS = {
    "T0": "Operation Epic Fury - Initial military operation",
    "T1": "Israeli-US coordinated strikes on Iranian facilities",
    "T2": "Iranian retaliatory strikes on Israeli and US positions",
    "T3": "Missile attacks directed toward British military bases",
    "T4": "Major oil refinery or tanker facility attacked",
    "T5": "Qatar announces halt to energy production/exports",
    "T6": "Natanz nuclear facility sustains significant damage",
    "T7": "US State Department suggests citizen evacuation from region",
    "T8": "Nine-country coalition announces ground invasion preparations",
    "T9": "Mojtaba Khamenei assumes leadership role in Iran",
    "T10": "Late escalation - additional major military action",
}

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Supported models
MODELS = [
    # Paper baseline models
    "anthropic/claude-sonnet-4-6",
    "google/gemini-3-flash-preview",
    "openai/gpt-5.4",
    "qwen/qwen3.5-35b-a3b",
    "moonshotai/kimi-k2.5",
    # Our addition
    "anthropic/claude-opus-4-6",
]

# Extraction model (precise/literal for probability extraction from responses)
EXTRACTION_MODEL = "anthropic/claude-haiku-4-5"

# Ground truth
GROUND_TRUTH_FILE = PIPELINE_ROOT / "ground_truth.json"

# Financial instruments
FINANCIAL_SYMBOLS = {
    "brent_crude": "BZ=F",
    "tel_aviv_35": "TA35.TA"
}

# Geographic regions for analysis
MIDDLE_EAST_COUNTRIES = [
    "Israel", "Iran", "Iraq", "Syria", "Lebanon", "Jordan", "Saudi Arabia",
    "UAE", "Qatar", "Kuwait", "Bahrain", "Oman", "Yemen", "Turkey"
]

# GDELT configuration
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_PARAMS = {
    "query": "Iran OR Israel OR Middle East conflict",
    "mode": "ArtList",
    "format": "json",
    "maxrecords": 250,
    "sort": "DateDesc"
}

# ADS-B Exchange configuration (placeholder for real API)
ADSB_BASE_URL = "https://adsbexchange.com/api/aircraft/v2"  # Placeholder
ADSB_REGION_BOUNDS = {
    "lat_min": 25.0,
    "lat_max": 42.0, 
    "lon_min": 25.0,
    "lon_max": 65.0
}

# UCDP/ACLED configuration
UCDP_BASE_URL = "https://ucdpapi.pcr.uu.se/api"
ACLED_BASE_URL = "https://api.acleddata.com/acled/read"

# Evaluation themes from the paper
EVALUATION_THEMES = [
    "Initial Outbreak",
    "Threshold Crossings", 
    "Economic Shockwaves",
    "Political Signaling"
]

# Questions from the paper (42 verifiable + 5 exploratory)
VERIFIABLE_QUESTIONS = [
    "Will there be a military strike within 24 hours?",
    "Will oil prices rise by more than 5% tomorrow?",
    "Will any country issue evacuation advisories?",
    "Will there be diplomatic communications at the UN level?",
    "Will any nuclear facilities be targeted?",
    # ... (expand with all 42 questions from paper)
]

EXPLORATORY_QUESTIONS = [
    "What is the most likely escalation scenario in the next 72 hours?",
    "Which regional actor is most likely to intervene next?",
    "What economic sectors will be most affected?",
    "How will alliance structures shift?",
    "What are the long-term geopolitical implications?"
]

# Rate limiting
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
MAX_RETRIES = 3
TIMEOUT = 120  # LLM responses can be long; 30s was too short

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"