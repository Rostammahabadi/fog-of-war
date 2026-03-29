"""
Fog of War LLM Evaluation Pipeline

A complete Python-based LLM evaluation pipeline for geopolitical forecasting,
inspired by the paper "When AI Navigates the Fog of War".

Key Features:
- Strict temporal gating (fog-of-war compliance)
- Multi-source data integration (economic, tactical, sentiment, ground truth)
- Multi-model LLM inference with rate limiting
- Comprehensive evaluation metrics (accuracy, calibration, Brier scores)
- Modular, production-ready architecture

Usage:
    python main.py fetch                 # Fetch data
    python main.py run --nodes all       # Run inference 
    python main.py evaluate             # Evaluate results
    python main.py full --evaluate      # Complete pipeline
"""

__version__ = "1.0.0"
__author__ = "OpenClaw AI Research"

from .config import TEMPORAL_NODES, MODELS
from .data_fetcher import DataFetcher
from .context_builder import ContextBuilder
from .prompt_builder import PromptBuilder
from .run_inference import LLMRunner
from .evaluator import Evaluator

__all__ = [
    "DataFetcher",
    "ContextBuilder", 
    "PromptBuilder",
    "LLMRunner",
    "Evaluator",
    "TEMPORAL_NODES",
    "MODELS"
]