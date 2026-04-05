"""
LLM inference runner for Fog of War pipeline.
Handles API calls to multiple LLM providers with rate limiting and retry logic.
"""

import json
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import openai
from openai import OpenAI
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    OPENROUTER_API_KEY, OPENAI_API_KEY, OPENROUTER_BASE_URL,
    MODELS, TEMPORAL_NODES, OUTPUT_DIR, RATE_LIMIT_DELAY,
    MAX_RETRIES, TIMEOUT, DEFAULT_TEMPERATURE, MAX_RESPONSE_TOKENS,
    VERIFIABLE_QUESTIONS_BY_NODE, EXPLORATORY_QUESTIONS,
    MAX_CONCURRENT_REQUESTS
)
from data_fetcher import DataFetcher
from context_builder import ContextBuilder
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class LLMRunner:
    """
    Handles LLM inference across multiple providers and models.
    Implements rate limiting, retry logic, and result persistence.
    """
    
    def __init__(self):
        self.openrouter_client = None
        self.openai_client = None
        self.prompt_builder = PromptBuilder()
        
        # Initialize clients
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize API clients."""
        if OPENROUTER_API_KEY:
            self.openrouter_client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY
            )
            logger.info("OpenRouter client initialized")
        
        if OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized")
        
        if not self.openrouter_client and not self.openai_client:
            logger.warning("No API clients initialized - check API keys")
    
    @retry(stop=stop_after_attempt(MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    def _make_llm_request(self, client: OpenAI, model: str,
                         user_prompt: str,
                         temperature: float = DEFAULT_TEMPERATURE) -> Dict[str, Any]:
        """
        Make LLM API request with retry logic.
        Paper protocol: no system prompt, single user message only.

        Args:
            client: OpenAI client instance
            model: Model identifier
            user_prompt: User prompt (context + question)
            temperature: Sampling temperature (paper: 0.3)

        Returns:
            API response dictionary
        """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=MAX_RESPONSE_TOKENS,
                timeout=TIMEOUT
            )

            return {
                "success": True,
                "response": response.choices[0].message.content,
                "model": model,
                "usage": response.usage.model_dump() if response.usage else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"LLM request failed for {model}: {e}")
            raise
    
    def run_single_inference(self, model: str, user_prompt: str,
                           temperature: float = DEFAULT_TEMPERATURE) -> Dict[str, Any]:
        """
        Run inference on a single model (paper protocol: no system prompt).

        Args:
            model: Model identifier
            user_prompt: User prompt (context + question)
            temperature: Sampling temperature (paper: 0.3)

        Returns:
            Inference result dictionary
        """
        logger.info(f"Running inference on {model}")

        # Route through OpenRouter for all provider-prefixed models (e.g. openai/gpt-5.4)
        # Only use direct OpenAI client for bare model names (e.g. gpt-5.4)
        client = None
        if "/" not in model and (model.startswith("gpt-") or model.startswith("o1-")):
            client = self.openai_client
        else:
            client = self.openrouter_client

        if not client:
            logger.error(f"No client available for model {model}")
            return {
                "success": False,
                "error": "No API client available",
                "model": model
            }

        try:
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

            result = self._make_llm_request(client, model, user_prompt, temperature)

            logger.info(f"Inference completed for {model}")
            return result

        except Exception as e:
            logger.error(f"Inference failed for {model}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _load_checkpoint(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Load existing node results as a checkpoint."""
        checkpoint_file = OUTPUT_DIR / f"node_{node_id}_results.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint for {node_id}: {e}")
        return None

    def run_node_analysis(self, node_id: str, intelligence_briefing: Dict[str, Any],
                         models: List[str], temperature: float = DEFAULT_TEMPERATURE,
                         save_results: bool = True,
                         skip_exploratory: bool = False,
                         no_cache: bool = False) -> Dict[str, Any]:
        """
        Run LLM analysis for a specific temporal node.
        Paper protocol: each question is posed independently (separate API call).
        Supports checkpointing: loads prior results and only runs missing calls.

        Args:
            node_id: Temporal node identifier (T0, T1, etc.)
            intelligence_briefing: Intelligence briefing from ContextBuilder
            models: List of models to run
            temperature: Sampling temperature (paper: 0.3)
            save_results: Whether to save results to disk
            skip_exploratory: Skip unscored exploratory questions to save cost

        Returns:
            Combined results from all models
        """
        # Get node-specific verifiable questions + exploratory questions
        node_questions = VERIFIABLE_QUESTIONS_BY_NODE.get(node_id, [])
        all_questions = node_questions + ([] if skip_exploratory else EXPLORATORY_QUESTIONS)
        num_verifiable = len(node_questions)

        # Load checkpoint to find already-completed work
        checkpoint = None if no_cache else self._load_checkpoint(node_id)
        cached_responses = {}  # (model, question_index) -> question_result
        if checkpoint:
            for model, model_data in checkpoint.get('model_results', {}).items():
                for i, qr in enumerate(model_data.get('question_results', [])):
                    if qr.get('success'):
                        cached_responses[(model, i)] = qr

        # Build context text once (shared across all questions for this node)
        context_text = self.prompt_builder.build_context_text(intelligence_briefing)

        # Determine which jobs still need to run
        jobs = []
        for model in models:
            for i, question in enumerate(all_questions):
                if (model, i) not in cached_responses:
                    user_prompt = self.prompt_builder.build_paper_prompt(context_text, question)
                    jobs.append((model, i, question, user_prompt))

        skipped = len(models) * len(all_questions) - len(jobs)
        if skipped > 0:
            logger.info(f"Checkpoint for {node_id}: {skipped} calls cached, {len(jobs)} remaining")

        if len(jobs) == 0:
            logger.info(f"Node {node_id} fully cached, skipping inference")
            return checkpoint

        logger.info(f"Running node analysis for {node_id}: {len(jobs)} API calls "
                    f"(max {MAX_CONCURRENT_REQUESTS} concurrent)")

        results = {
            'node_id': node_id,
            'target_date': intelligence_briefing.get('target_date'),
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'model_results': {},
            'metadata': {
                'total_models': len(models),
                'successful_models': 0,
                'failed_models': 0,
                'questions_per_model': len(all_questions),
                'verifiable_questions': num_verifiable,
                'exploratory_questions': len(EXPLORATORY_QUESTIONS),
            }
        }

        # Run remaining jobs in parallel with a thread pool
        job_results = {}
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool:
            future_to_job = {
                pool.submit(self.run_single_inference, model, prompt, temperature): (model, idx, question)
                for model, idx, question, prompt in jobs
            }
            for future in as_completed(future_to_job):
                model, idx, question = future_to_job[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Parallel call failed for {model} q{idx}: {e}")
                    result = {"success": False, "error": str(e), "model": model}
                job_results[(model, idx)] = (question, result)

        # Reassemble results by model, merging cached + fresh
        for model in models:
            model_question_results = []
            model_success = True

            for i in range(len(all_questions)):
                # Use cached result if available, otherwise use fresh result
                if (model, i) in cached_responses:
                    qr = cached_responses[(model, i)]
                elif (model, i) in job_results:
                    question, result = job_results[(model, i)]
                    is_verifiable = i < num_verifiable
                    qr = {
                        'question': question,
                        'question_type': 'verifiable' if is_verifiable else 'exploratory',
                        'success': result.get('success', False),
                        'response': result.get('response', ''),
                        'usage': result.get('usage'),
                        'timestamp': result.get('timestamp'),
                    }
                else:
                    # Should not happen, but handle gracefully
                    qr = {
                        'question': all_questions[i],
                        'question_type': 'verifiable' if i < num_verifiable else 'exploratory',
                        'success': False,
                        'response': '',
                        'usage': None,
                        'timestamp': None,
                    }

                model_question_results.append(qr)
                if not qr.get('success'):
                    model_success = False

            if model_success:
                results['metadata']['successful_models'] += 1
            else:
                results['metadata']['failed_models'] += 1

            results['model_results'][model] = {
                'success': model_success,
                'model': model,
                'question_results': model_question_results,
                # Concatenated response for backward compatibility with evaluator
                'response': '\n\n---\n\n'.join(
                    f"Q: {qr['question']}\n\n{qr['response']}"
                    for qr in model_question_results if qr['success']
                ),
            }

        # Save results (acts as checkpoint for future runs)
        if save_results:
            self._save_node_results(node_id, results)

        logger.info(f"Node analysis complete for {node_id}: "
                   f"{results['metadata']['successful_models']}/{len(models)} models succeeded")

        return results
    
    def run_temporal_sequence(self, nodes: List[str], data: Dict[str, Any],
                            models: List[str], temperature: float = DEFAULT_TEMPERATURE,
                            save_individual: bool = True,
                            skip_exploratory: bool = False,
                            no_cache: bool = False) -> Dict[str, Any]:
        """
        Run LLM inference across temporal sequence.

        Args:
            nodes: List of temporal node IDs to analyze
            data: Full dataset from DataFetcher
            models: List of models to run
            temperature: Sampling temperature
            save_individual: Whether to save individual node results
            skip_exploratory: Skip unscored exploratory questions to save cost

        Returns:
            Combined results across all nodes
        """
        logger.info(f"Running temporal sequence for {len(nodes)} nodes")
        
        context_builder = ContextBuilder(data)
        sequence_results = {
            'sequence_id': f"temporal_sequence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            'execution_timestamp': datetime.now(timezone.utc).isoformat(),
            'nodes': nodes,
            'models': models,
            'node_results': {},
            'summary': {
                'total_nodes': len(nodes),
                'completed_nodes': 0,
                'total_inferences': 0,
                'successful_inferences': 0
            }
        }
        
        for node_id in nodes:
            if node_id not in TEMPORAL_NODES:
                logger.error(f"Unknown node ID: {node_id}")
                continue
            
            target_date = TEMPORAL_NODES[node_id]
            logger.info(f"Processing node {node_id} at {target_date}")
            
            # Build context with strict temporal gating
            intelligence_briefing = context_builder.build_context(target_date)
            
            # Run analysis for this node
            node_results = self.run_node_analysis(
                node_id, intelligence_briefing, models,
                temperature, save_individual, skip_exploratory, no_cache
            )
            
            sequence_results['node_results'][node_id] = node_results
            sequence_results['summary']['completed_nodes'] += 1
            
            # Update inference counts
            sequence_results['summary']['total_inferences'] += node_results['metadata']['total_models']
            sequence_results['summary']['successful_inferences'] += node_results['metadata']['successful_models']
        
        # Save sequence results
        self._save_sequence_results(sequence_results)
        
        logger.info(f"Temporal sequence complete: "
                   f"{sequence_results['summary']['successful_inferences']}"
                   f"/{sequence_results['summary']['total_inferences']} inferences succeeded")
        
        return sequence_results
    
    def run_full_pipeline(self, models: Optional[List[str]] = None,
                         temperature: float = DEFAULT_TEMPERATURE,
                         fetch_fresh_data: bool = False,
                         skip_exploratory: bool = False,
                         no_cache: bool = False) -> Dict[str, Any]:
        """
        Run complete end-to-end pipeline.

        Args:
            models: List of models to use (defaults to all available)
            temperature: Sampling temperature
            fetch_fresh_data: Whether to fetch fresh data or use cache
            skip_exploratory: Skip unscored exploratory questions to save cost

        Returns:
            Complete pipeline results
        """
        logger.info("Starting full pipeline execution")
        
        if not models:
            models = MODELS
        
        pipeline_results = {
            'pipeline_id': f"full_pipeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            'execution_timestamp': datetime.now(timezone.utc).isoformat(),
            'configuration': {
                'models': models,
                'temperature': temperature,
                'fetch_fresh_data': fetch_fresh_data
            },
            'stages': {}
        }
        
        try:
            # Stage 1: Data Fetching
            logger.info("Pipeline Stage 1: Data Fetching")
            data_fetcher = DataFetcher()
            
            cache_file = OUTPUT_DIR / "pipeline_data_cache.json"
            
            if fetch_fresh_data or not cache_file.exists():
                from config import START_DATE, END_DATE
                data = data_fetcher.fetch_all_data(START_DATE, END_DATE)
                data_fetcher.save_data_cache(data, cache_file)
                logger.info("Fresh data fetched and cached")
            else:
                data = data_fetcher.load_data_cache(cache_file)
                logger.info("Data loaded from cache")
            
            pipeline_results['stages']['data_fetching'] = {
                'status': 'completed',
                'fresh_data': fetch_fresh_data,
                'cache_file': str(cache_file)
            }
            
            # Stage 2: Temporal Sequence Analysis
            logger.info("Pipeline Stage 2: Temporal Sequence Analysis")
            all_nodes = list(TEMPORAL_NODES.keys())
            
            sequence_results = self.run_temporal_sequence(
                all_nodes, data, models, temperature,
                skip_exploratory=skip_exploratory, no_cache=no_cache
            )
            
            pipeline_results['stages']['temporal_analysis'] = sequence_results
            
            # Stage 3: Results Summary
            logger.info("Pipeline Stage 3: Results Summary")
            summary = self._generate_pipeline_summary(sequence_results)
            pipeline_results['summary'] = summary
            
            # Save complete pipeline results
            self._save_pipeline_results(pipeline_results)
            
            logger.info("Full pipeline execution completed successfully")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            pipeline_results['error'] = str(e)
            pipeline_results['status'] = 'failed'
            return pipeline_results
    
    def _save_node_results(self, node_id: str, results: Dict[str, Any]):
        """Save individual node results to disk."""
        output_file = OUTPUT_DIR / f"node_{node_id}_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.debug(f"Node {node_id} results saved to {output_file}")
    
    def _save_sequence_results(self, results: Dict[str, Any]):
        """Save temporal sequence results to disk."""
        output_file = OUTPUT_DIR / f"sequence_{results['sequence_id']}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.debug(f"Sequence results saved to {output_file}")
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save complete pipeline results to disk."""
        output_file = OUTPUT_DIR / f"pipeline_{results['pipeline_id']}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved to {output_file}")
    
    def _generate_pipeline_summary(self, sequence_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level summary of pipeline results."""
        summary = {
            'execution_overview': {
                'total_nodes': sequence_results['summary']['total_nodes'],
                'completed_nodes': sequence_results['summary']['completed_nodes'],
                'total_inferences': sequence_results['summary']['total_inferences'],
                'successful_inferences': sequence_results['summary']['successful_inferences'],
                'success_rate': sequence_results['summary']['successful_inferences'] / 
                              max(sequence_results['summary']['total_inferences'], 1)
            },
            'model_performance': {},
            'temporal_coverage': {}
        }
        
        # Analyze model performance across all nodes
        model_stats = {}
        for node_id, node_results in sequence_results['node_results'].items():
            for model, model_result in node_results['model_results'].items():
                if model not in model_stats:
                    model_stats[model] = {'total': 0, 'successful': 0}
                
                model_stats[model]['total'] += 1
                if model_result.get('success'):
                    model_stats[model]['successful'] += 1
        
        for model, stats in model_stats.items():
            success_rate = stats['successful'] / max(stats['total'], 1)
            summary['model_performance'][model] = {
                'total_runs': stats['total'],
                'successful_runs': stats['successful'],
                'success_rate': success_rate
            }
        
        # Temporal coverage
        for node_id in sequence_results['node_results']:
            target_date = TEMPORAL_NODES.get(node_id)
            if target_date:
                summary['temporal_coverage'][node_id] = {
                    'target_date': target_date.isoformat(),
                    'analysis_completed': node_id in sequence_results['node_results']
                }
        
        return summary
    
    def load_results(self, result_type: str, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Load saved results from disk.
        
        Args:
            result_type: Type of results ('node', 'sequence', 'pipeline')
            identifier: Result identifier
            
        Returns:
            Loaded results or None if not found
        """
        file_patterns = {
            'node': f"node_{identifier}_results.json",
            'sequence': f"sequence_{identifier}.json", 
            'pipeline': f"pipeline_{identifier}.json"
        }
        
        pattern = file_patterns.get(result_type)
        if not pattern:
            logger.error(f"Unknown result type: {result_type}")
            return None
        
        result_file = OUTPUT_DIR / pattern
        
        if not result_file.exists():
            logger.error(f"Results file not found: {result_file}")
            return None
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load results from {result_file}: {e}")
            return None


if __name__ == "__main__":
    # Test the LLM runner
    import logging
    from datetime import datetime, timezone, timedelta
    
    logging.basicConfig(level=logging.INFO)
    
    runner = LLMRunner()
    
    # Test single model with minimal data
    test_briefing = {
        'target_date': '2026-03-01T00:00:00+00:00',
        'executive_summary': {
            'threat_level': 'MODERATE',
            'confidence': 'MEDIUM',
            'situation_overview': 'Test briefing for pipeline validation'
        }
    }
    
    # Test with available models (fallback to fewer if API keys not available)
    available_models = []
    if runner.openrouter_client:
        available_models.append("anthropic/claude-3.5-sonnet")
    if runner.openai_client:
        available_models.append("gpt-4o")
    
    if available_models:
        print(f"Testing with models: {available_models}")

        # Test single inference (paper protocol: no system prompt)
        user_prompt = runner.prompt_builder.build_paper_prompt(
            str(test_briefing),
            "What is the probability of military escalation within 24 hours?"
        )

        result = runner.run_single_inference(
            available_models[0], user_prompt
        )
        
        print(f"Test inference result: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"Response length: {len(result.get('response', ''))}")
    else:
        print("No API clients available - check API keys")
        print("Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variables")