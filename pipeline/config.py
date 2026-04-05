"""
Configuration module for Fog of War LLM evaluation pipeline.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

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
    "T8": datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc),   # Nine Countries Involved and Israeli Ground Invasion
    "T9": datetime(2026, 3, 3, 18, 0, tzinfo=timezone.utc),   # Mojtaba Khamenei Becomes Leader
    "T10": datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc),   # Iranian Apology to Neighboring Countries
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
    "T8": "Nine countries involved and Israeli ground invasion",
    "T9": "Mojtaba Khamenei assumes leadership role in Iran",
    "T10": "Iranian apology to neighboring countries",
}

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Supported models
MODELS = [
    # Paper baseline models
    "anthropic/claude-sonnet-4-6",
    "google/gemini-3.1-flash-lite-preview",
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

# Paper's 42 verifiable questions, keyed by temporal node
VERIFIABLE_QUESTIONS_BY_NODE = {
    "T0": [
        "Is the United States likely to conduct direct military strikes on Iranian territory?",
        "Could the United States prioritize economic sanctions or diplomatic pressure over military action?",
        "Does the announcement of Operation Epic Fury signal the beginning of a broader military conflict?",
    ],
    "T1": [
        "Are the Israeli-U.S. strikes likely to escalate into a broader military campaign?",
        "Is Iran likely to retaliate militarily in response to the strikes?",
        "If Iran retaliates, is it more likely to indiscriminately bomb Israeli-U.S. territory than to target U.S. military bases?",
        "Could Gulf states become direct targets of Iranian retaliation?",
        "Is regional airspace in the Middle East likely to face widespread closures due to security risks?",
    ],
    "T2": [
        "Are the Iranian strikes likely to remain symbolic retaliation without sustained military operations?",
        "Could Iran attempt to disrupt shipping through the Strait of Hormuz?",
        "Are other countries in the region likely to become directly involved in the conflict?",
        "Is large-scale closure of Middle Eastern airspace likely following these strikes?",
        "Could the conflict trigger internal rebellion within Iran?",
    ],
    "T3": [
        "Is the United Kingdom likely to become directly involved in the conflict?",
        "Is NATO likely to become involved, expanding the conflict into the Mediterranean theater?",
        "Could the conflict disrupt commercial shipping or maritime security in the Mediterranean?",
    ],
    "T4": [
        "Is Iran likely to continue targeting oil tankers in an attempt to disrupt traffic through the Strait of Hormuz?",
        "Could international naval forces establish escort missions to protect commercial shipping?",
        "Could American naval forces establish escort missions to protect commercial shipping?",
        "Are energy facilities such as refineries, desalination plants, and oil terminals likely to become primary targets?",
        "Could these attacks lead to significant volatility in global oil prices?",
    ],
    "T5": [
        "Could Qatar's decision lead to natural gas shortages in Europe or Asia?",
        "Are global natural gas prices likely to increase significantly as a result?",
        "Could other LNG facilities or energy infrastructure in the Gulf region become targets?",
        "Are major energy-importing countries likely to seek alternative supply sources?",
    ],
    "T6": [
        "Are the United States and Israel likely to continue targeting Iranian nuclear facilities?",
        "Will Israel's nuclear-related infrastructure be damaged?",
        "Is Iran likely to withdraw from nuclear non-proliferation commitments?",
        "Could nuclear weapons be used as part of the conflict?",
    ],
    "T7": [
        "Are other countries likely to begin evacuating their citizens from the region as well?",
        "Could the United States deploy ground forces if the conflict escalates further?",
        "Are foreign governments likely to close or reduce operations at diplomatic missions in the region?",
    ],
    "T8": [
        "Is the conflict likely to expand further, involving additional countries?",
        "Could multiple countries initiate ground operations as the war escalates?",
        "Is Iran likely to increase military or logistical support for Hezbollah?",
    ],
    "T9": [
        "Is the new leadership more likely to escalate military retaliation rather than pursue negotiations?",
        "Are the United States and Israel likely to target the new leadership structure in further strikes?",
        "Could the leadership transition trigger domestic unrest or protests in Iran?",
    ],
    "T10": [
        "Is Iran likely to reduce or halt attacks on neighboring Gulf states?",
        "Could Iran begin pursuing ceasefire negotiations or diplomatic talks?",
        "Are international actors such as the EU or the United Nations likely to push for negotiations following this signal?",
        "Could the overall intensity of the conflict begin to decrease?",
    ],
}

# Flat list for backward compatibility
VERIFIABLE_QUESTIONS = [q for qs in VERIFIABLE_QUESTIONS_BY_NODE.values() for q in qs]

# Paper's 5 general exploratory questions (asked at every node)
EXPLORATORY_QUESTIONS = [
    "What are the potential future actions by the United States and Israel?",
    "What are the potential future actions by Iran?",
    "What are the potential involvement or reactions from other major countries?",
    "Will the conflict escalate into a global war?",
    "What is the most probable pathway to de-escalation or resolution of the Iran-US conflict, and what is a realistic timeline?",
]

# Inference parameters (paper: temperature 0.3, max tokens 2048, no system prompt)
DEFAULT_TEMPERATURE = 0.3
MAX_RESPONSE_TOKENS = 2048

# Rate limiting / concurrency
MAX_CONCURRENT_REQUESTS = 10  # parallel API calls per node
RATE_LIMIT_DELAY = 0.2  # seconds between API calls (lower since we're parallel now)
MAX_RETRIES = 3
TIMEOUT = 120  # LLM responses can be long; 30s was too short

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"