#!/usr/bin/env python3
"""
Main orchestrator for Fog of War LLM evaluation pipeline.
Provides command-line interface for all pipeline operations.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import (
    TEMPORAL_NODES, NODE_DESCRIPTIONS, MODELS, OUTPUT_DIR, START_DATE, END_DATE,
    LOG_FORMAT, LOG_LEVEL
)
from data_fetcher import DataFetcher
from context_builder import ContextBuilder
from prompt_builder import PromptBuilder
from run_inference import LLMRunner
from evaluator import Evaluator

# Setup logging
def setup_logging(level: str = LOG_LEVEL):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / "pipeline.log")
        ]
    )

logger = logging.getLogger(__name__)


def cmd_fetch(args):
    """Fetch all data sources."""
    logger.info("Starting data fetch operation")
    
    fetcher = DataFetcher()
    
    # Use custom date range if provided
    start_date = START_DATE
    end_date = END_DATE
    
    if args.start_date:
        start_date = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
    
    logger.info(f"Fetching data from {start_date} to {end_date}")
    
    # Fetch data
    data = fetcher.fetch_all_data(start_date, end_date)
    
    # Save to cache
    cache_file = OUTPUT_DIR / args.output
    fetcher.save_data_cache(data, cache_file)
    
    logger.info(f"Data fetch completed. Cache saved to {cache_file}")
    
    # Print summary
    print(f"\nData Fetch Summary:")
    print(f"Cache file: {cache_file}")
    
    for source, df in data.items():
        if hasattr(df, '__len__'):
            if hasattr(df, 'columns'):  # DataFrame
                print(f"{source}: {len(df)} records")
            else:  # Dictionary or list
                if isinstance(df, dict):
                    total = sum(len(v) if hasattr(v, '__len__') else 1 for v in df.values())
                    print(f"{source}: {total} items")
                else:
                    print(f"{source}: {len(df)} items")


def cmd_build(args):
    """Build intelligence briefing for specific node."""
    logger.info(f"Building context for node {args.node}")
    
    if args.node not in TEMPORAL_NODES:
        logger.error(f"Unknown node: {args.node}")
        print(f"Error: Unknown node '{args.node}'")
        print(f"Available nodes: {', '.join(TEMPORAL_NODES.keys())}")
        return 1
    
    # Load data
    cache_file = OUTPUT_DIR / args.data_cache
    if not cache_file.exists():
        logger.error(f"Data cache not found: {cache_file}")
        print(f"Error: Data cache not found: {cache_file}")
        print("Run 'python main.py fetch' first to create data cache")
        return 1
    
    fetcher = DataFetcher()
    data = fetcher.load_data_cache(cache_file)
    
    # Build context
    builder = ContextBuilder(data)
    target_date = TEMPORAL_NODES[args.node]
    
    briefing = builder.build_context(target_date)
    
    # Save briefing
    output_file = OUTPUT_DIR / f"briefing_{args.node}_{target_date.strftime('%Y%m%d_%H%M')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(briefing, f, indent=2, default=str)
    
    # Also save as markdown if requested
    if args.format == 'markdown' or args.markdown:
        markdown = builder.format_as_markdown(briefing)
        md_file = output_file.with_suffix('.md')
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        logger.info(f"Markdown briefing saved to {md_file}")
        
        if args.format == 'markdown':
            print(markdown)
        else:
            print(f"Briefing saved to: {output_file}")
            print(f"Markdown version: {md_file}")
    else:
        print(f"Briefing saved to: {output_file}")
        
        if args.verbose:
            print(f"\nBriefing Summary for {args.node}:")
            print(f"Target Date: {briefing['target_date']}")
            print(f"Threat Level: {briefing['executive_summary']['threat_level']}")
            print(f"Key Developments: {len(briefing['executive_summary']['key_developments'])}")


def cmd_run(args):
    """Run LLM inference."""
    logger.info("Starting LLM inference")
    
    # Load data
    cache_file = OUTPUT_DIR / args.data_cache
    if not cache_file.exists():
        logger.error(f"Data cache not found: {cache_file}")
        print(f"Error: Data cache not found: {cache_file}")
        print("Run 'python main.py fetch' first to create data cache")
        return 1
    
    fetcher = DataFetcher()
    data = fetcher.load_data_cache(cache_file)
    
    # Setup models
    models = args.model if args.model else MODELS
    if isinstance(models, str):
        models = [models]
    
    # Setup nodes
    if args.nodes == 'all':
        nodes = list(TEMPORAL_NODES.keys())
    else:
        nodes = args.nodes.split(',')
        # Validate nodes
        for node in nodes:
            if node not in TEMPORAL_NODES:
                logger.error(f"Unknown node: {node}")
                print(f"Error: Unknown node '{node}'")
                return 1
    
    logger.info(f"Running inference on models: {models}")
    logger.info(f"Running inference on nodes: {nodes}")
    
    # Run inference
    runner = LLMRunner()
    
    if len(nodes) == 1:
        # Single node analysis
        node_id = nodes[0]
        target_date = TEMPORAL_NODES[node_id]
        
        builder = ContextBuilder(data)
        briefing = builder.build_context(target_date)
        
        results = runner.run_node_analysis(node_id, briefing, models, args.temperature)
        
        print(f"\nInference completed for node {node_id}")
        print(f"Models: {models}")
        print(f"Success rate: {results['metadata']['successful_models']}/{results['metadata']['total_models']}")
        
    else:
        # Temporal sequence analysis
        results = runner.run_temporal_sequence(nodes, data, models, args.temperature)
        
        print(f"\nTemporal sequence completed")
        print(f"Nodes: {len(nodes)}")
        print(f"Models: {models}")
        print(f"Success rate: {results['summary']['successful_inferences']}/{results['summary']['total_inferences']}")
    
    if args.verbose:
        print(f"\nResults saved to: {OUTPUT_DIR}")


def cmd_evaluate(args):
    """Run evaluation against ground truth."""
    logger.info("Starting evaluation")
    
    # Load inference results
    if args.inference_results:
        results_file = Path(args.inference_results)
    else:
        # Find most recent sequence results
        sequence_files = list(OUTPUT_DIR.glob("sequence_temporal_sequence_*.json"))
        if not sequence_files:
            logger.error("No sequence results found")
            print("Error: No inference results found")
            print("Run 'python main.py run --nodes all' first")
            return 1
        
        results_file = max(sequence_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using most recent results: {results_file}")
    
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        print(f"Error: Results file not found: {results_file}")
        return 1
    
    # Load results
    with open(results_file, 'r', encoding='utf-8') as f:
        sequence_results = json.load(f)
    
    # Load ground truth (real questions from ground_truth.json, or synthetic fallback)
    if args.ground_truth:
        gt_file = Path(args.ground_truth)
        with open(gt_file, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
    else:
        # Use real ground truth — evaluator loads it internally from ground_truth.json
        # Still need per-node structure for evaluate_sequence interface
        ground_truth_data = {}
        for node_id in TEMPORAL_NODES:
            ground_truth_data[node_id] = {
                'events': [{'description': NODE_DESCRIPTIONS.get(node_id, ''), 'date': TEMPORAL_NODES[node_id].isoformat()}]
            }
    
    # Run evaluation
    evaluator = Evaluator()
    evaluation_results = evaluator.evaluate_sequence(sequence_results, ground_truth_data)
    
    # Save evaluation results
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"evaluation_{timestamp}.json"
    
    evaluator.save_evaluation_results(evaluation_results, output_file)
    
    # Print summary
    print(f"\nEvaluation completed")
    print(f"Results saved to: {output_file}")
    print(f"Report saved to: {output_file.with_suffix('.md')}")
    
    if args.verbose:
        sequence_metrics = evaluation_results['sequence_metrics']
        print(f"\nEvaluation Summary:")
        print(f"Overall Accuracy: {sequence_metrics.get('overall_sequence_accuracy', 0):.2%}")
        print(f"Nodes Evaluated: {sequence_metrics.get('total_nodes_evaluated', 0)}")
        
        model_comparison = evaluation_results['model_comparison']
        if model_comparison:
            print(f"\nTop Model by Accuracy:")
            best_model = max(model_comparison.items(), key=lambda x: x[1].get('mean_accuracy', 0))
            print(f"{best_model[0]}: {best_model[1].get('mean_accuracy', 0):.2%}")


def cmd_full(args):
    """Run complete end-to-end pipeline."""
    logger.info("Starting full pipeline execution")
    
    # Setup models
    models = args.model if args.model else MODELS
    if isinstance(models, str):
        models = [models]
    
    # Run full pipeline
    runner = LLMRunner()
    results = runner.run_full_pipeline(
        models=models,
        temperature=args.temperature,
        fetch_fresh_data=args.fetch_fresh
    )
    
    if results.get('error'):
        logger.error(f"Pipeline failed: {results['error']}")
        print(f"Pipeline execution failed: {results['error']}")
        return 1
    
    print(f"\nFull pipeline completed successfully")
    print(f"Pipeline ID: {results['pipeline_id']}")
    
    # Print summary
    if 'summary' in results:
        summary = results['summary']
        exec_overview = summary['execution_overview']
        
        print(f"\nExecution Summary:")
        print(f"Nodes: {exec_overview['completed_nodes']}/{exec_overview['total_nodes']}")
        print(f"Inferences: {exec_overview['successful_inferences']}/{exec_overview['total_inferences']}")
        print(f"Success Rate: {exec_overview['success_rate']:.2%}")
        
        # Model performance
        if 'model_performance' in summary:
            print(f"\nModel Performance:")
            for model, perf in summary['model_performance'].items():
                print(f"{model}: {perf['success_rate']:.2%} success rate")
    
    # Run evaluation if requested
    if args.evaluate:
        logger.info("Running evaluation as part of full pipeline")
        
        # Get sequence results
        temporal_analysis = results['stages']['temporal_analysis']
        
        # Build ground truth structure (evaluator loads real questions internally)
        ground_truth_data = {}
        for node_id in TEMPORAL_NODES:
            ground_truth_data[node_id] = {
                'events': [{'description': NODE_DESCRIPTIONS.get(node_id, ''), 'date': TEMPORAL_NODES[node_id].isoformat()}]
            }
        
        # Evaluate
        evaluator = Evaluator()
        evaluation_results = evaluator.evaluate_sequence(temporal_analysis, ground_truth_data)
        
        # Save evaluation
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        eval_output = OUTPUT_DIR / f"full_pipeline_evaluation_{timestamp}.json"
        evaluator.save_evaluation_results(evaluation_results, eval_output)
        
        print(f"\nEvaluation completed and saved to: {eval_output}")


def generate_synthetic_ground_truth():
    """Generate synthetic ground truth data for evaluation."""
    from config import NODE_DESCRIPTIONS
    
    ground_truth = {}
    
    for node_id, description in NODE_DESCRIPTIONS.items():
        target_date = TEMPORAL_NODES[node_id]
        
        # Create synthetic event based on node description
        ground_truth[node_id] = {
            'events': [{
                'description': description,
                'date': target_date.strftime('%Y-%m-%d'),
                'timestamp': target_date.isoformat(),
                'fatalities': hash(node_id) % 50 + 5,  # Synthetic casualty count
                'severity': 'high' if 'nuclear' in description.lower() else 'medium'
            }],
            'economic_impact': {
                'oil_price_change': (hash(node_id) % 20 - 10) / 100.0,  # -10% to +10%
                'market_volatility': 'high' if node_id in ['T2', 'T6', 'T8'] else 'medium'
            }
        }
    
    return ground_truth


def main():
    """Main entry point."""
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    parser = argparse.ArgumentParser(
        description="Fog of War LLM Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py fetch                          # Fetch all data sources
  python main.py build --node T3               # Build briefing for node T3
  python main.py run --model claude-3.5-sonnet --nodes T1,T2  # Run inference on specific nodes
  python main.py run --nodes all               # Run inference on all temporal nodes
  python main.py evaluate                      # Evaluate most recent results
  python main.py full --evaluate               # Complete pipeline with evaluation

Temporal Nodes:
  T0: Feb 27 - Operation Epic Fury
  T1: Feb 28 00:00 - Israeli-US Strikes  
  T2: Feb 28 12:00 - Iranian Strikes
  T3: Mar 1 00:00 - Missiles toward British Bases
  T4: Mar 1 12:00 - Oil Refinery/Tanker Attacked
  T5: Mar 2 00:00 - Qatar Halts Energy Production
  T6: Mar 2 12:00 - Natanz Nuclear Facility Damaged
  T7: Mar 3 00:00 - US Suggests Citizen Evacuation
  T8: Mar 3 12:00 - Nine Countries; Ground Invasion
  T9: Mar 3 18:00 - Mojtaba Khamenei Becomes Leader
  T10: Mar 7 00:00 - Late Escalation Node
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--log-level', default=LOG_LEVEL,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch all data sources')
    fetch_parser.add_argument('--start-date', help='Start date (ISO format)')
    fetch_parser.add_argument('--end-date', help='End date (ISO format)')
    fetch_parser.add_argument('--output', default='pipeline_data_cache.json',
                             help='Output cache file name')
    fetch_parser.set_defaults(func=cmd_fetch)
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build intelligence briefing')
    build_parser.add_argument('--node', required=True, 
                             choices=list(TEMPORAL_NODES.keys()),
                             help='Temporal node to build context for')
    build_parser.add_argument('--data-cache', default='pipeline_data_cache.json',
                             help='Data cache file to use')
    build_parser.add_argument('--format', choices=['json', 'markdown'], 
                             default='json', help='Output format')
    build_parser.add_argument('--markdown', action='store_true',
                             help='Also save markdown version')
    build_parser.set_defaults(func=cmd_build)
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run LLM inference')
    run_parser.add_argument('--model', action='append',
                           help='Model to use (can be specified multiple times)')
    run_parser.add_argument('--nodes', default='all',
                           help='Comma-separated node IDs or "all"')
    run_parser.add_argument('--temperature', type=float, default=0.7,
                           help='Sampling temperature')
    run_parser.add_argument('--data-cache', default='pipeline_data_cache.json',
                           help='Data cache file to use')
    run_parser.set_defaults(func=cmd_run)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate predictions')
    eval_parser.add_argument('--inference-results',
                            help='Inference results file (default: most recent)')
    eval_parser.add_argument('--ground-truth',
                            help='Ground truth file (default: synthetic)')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    full_parser.add_argument('--model', action='append',
                            help='Model to use (can be specified multiple times)')
    full_parser.add_argument('--temperature', type=float, default=0.7,
                            help='Sampling temperature')
    full_parser.add_argument('--fetch-fresh', action='store_true',
                            help='Fetch fresh data instead of using cache')
    full_parser.add_argument('--evaluate', action='store_true',
                            help='Run evaluation after inference')
    full_parser.set_defaults(func=cmd_full)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Execute command
        return args.func(args) or 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\nPipeline interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        print(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())