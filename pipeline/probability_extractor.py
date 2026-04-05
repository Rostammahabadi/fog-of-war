"""
LLM-based probability extraction from model responses.
Replaces rigid regex patterns with a cheap LLM call to extract
the final probability estimate from each response.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict

from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EXTRACTION_MODEL, OUTPUT_DIR

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are extracting a probability estimate from an intelligence analyst's response.

The analyst was asked this specific question:
"{question}"

Their full response is below. Extract the probability they assigned to this specific question.

Rules:
- Look for explicit percentages first (e.g. "70%", "0.70")
- If they gave a range (e.g. "60-70%"), return the midpoint (65)
- If they gave multiple probabilities for different scenarios, return the one most directly answering the question above
- If they used qualitative language instead of a number, map it:
  - "almost certain" / "extremely likely" / "near certain" → 95
  - "very likely" / "highly probable" → 85
  - "likely" / "probable" → 70
  - "more likely than not" / "lean towards" → 60
  - "roughly even" / "coin flip" / "toss-up" → 50
  - "unlikely" / "improbable" → 30
  - "very unlikely" / "highly improbable" → 15
  - "almost impossible" / "near zero" → 5
- If they said the event already occurred or is confirmed, return 97
- If they said the event is impossible or ruled out, return 3
- If no probability AT ALL can be inferred from the response, return -1

Return ONLY valid JSON on a single line, nothing else:
{{"probability": <number 0-100 or -1>, "raw_quote": "<the exact short phrase where they state this probability>"}}

RESPONSE:
{response}"""


class LLMProbabilityExtractor:

    def __init__(self, cache_path: Path = OUTPUT_DIR / "extraction_cache.json"):
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.client = None
        if OPENROUTER_API_KEY:
            self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

    def extract_probability(self, response_text: str, question: str,
                            model_name: str, node_id: str) -> Dict:
        if not response_text:
            return {"probability": None, "raw_quote": None, "source": "none"}

        key = self._cache_key(model_name, node_id, response_text)
        if key in self.cache:
            cached = self.cache[key]
            # Skip stale regex-extracted cache entries — re-extract with LLM
            if cached.get("source") != "regex":
                return cached

        result = self._call_extraction_model(response_text, question)

        # Retry once on failure with a longer truncation window
        if result["source"] == "none":
            result = self._call_extraction_model(response_text, question, truncate_len=10000)

        self.cache[key] = result
        self._save_cache()
        return result

    def _call_extraction_model(self, response_text: str, question: str,
                               truncate_len: int = 6000) -> Dict:
        if not self.client:
            return {"probability": None, "raw_quote": None, "source": "none"}

        truncated = response_text[:truncate_len]

        try:
            resp = self.client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[{
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(question=question, response=truncated)
                }],
                max_tokens=200,
                temperature=0,
                timeout=30
            )
            raw = resp.choices[0].message.content.strip()

            # Parse JSON from response (handle markdown code blocks)
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            prob = parsed.get("probability")
            quote = parsed.get("raw_quote", "")

            if prob is not None and prob >= 0:
                return {
                    "probability": prob / 100.0,
                    "raw_quote": quote,
                    "source": "llm"
                }
            else:
                return {"probability": None, "raw_quote": None, "source": "none"}

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return {"probability": None, "raw_quote": None, "source": "none"}

    def _cache_key(self, model_name: str, node_id: str, response_text: str) -> str:
        content = f"{model_name}|{node_id}|{response_text[:500]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_cache(self) -> Dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_cache(self):
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=2)
