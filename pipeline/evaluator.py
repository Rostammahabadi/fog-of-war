"""
Evaluation module for Fog of War pipeline.
Compares LLM predictions against ground truth events and generates evaluation metrics.
"""

import json
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    TEMPORAL_NODES, NODE_DESCRIPTIONS, EVALUATION_THEMES,
    OUTPUT_DIR, VERIFIABLE_QUESTIONS, EXPLORATORY_QUESTIONS,
    GROUND_TRUTH_FILE
)
from probability_extractor import LLMProbabilityExtractor

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates LLM predictions against ground truth events.
    Calculates accuracy, calibration, Brier scores, and thematic analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.extractor = LLMProbabilityExtractor()
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> Dict[str, List[Dict]]:
        """Load ground truth questions grouped by node_id."""
        if not GROUND_TRUTH_FILE.exists():
            self.logger.warning("ground_truth.json not found, using empty ground truth")
            return {}
        with open(GROUND_TRUTH_FILE, 'r') as f:
            questions = json.load(f)
        by_node = {}
        for q in questions:
            node = q["node_id"]
            by_node.setdefault(node, []).append(q)
        return by_node
    
    def evaluate_node_predictions(self, predictions: Dict[str, Any], 
                                 ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate predictions for a single temporal node.
        
        Args:
            predictions: LLM predictions for the node
            ground_truth: Actual events that occurred
            
        Returns:
            Evaluation results dictionary
        """
        node_id = predictions.get('node_id', 'unknown')
        self.logger.info(f"Evaluating predictions for node {node_id}")
        
        evaluation = {
            'node_id': node_id,
            'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
            'target_date': predictions.get('target_date'),
            'model_evaluations': {},
            'aggregate_metrics': {},
            'ground_truth_summary': self._summarize_ground_truth(ground_truth)
        }
        
        # Evaluate each model's predictions (including partial results)
        for model, model_result in predictions.get('model_results', {}).items():
            has_any_response = any(
                qr.get('success') for qr in model_result.get('question_results', [])
            ) if 'question_results' in model_result else model_result.get('success')
            if has_any_response:
                model_eval = self._evaluate_single_model(model_result, ground_truth, node_id)
                evaluation['model_evaluations'][model] = model_eval
        
        # Calculate aggregate metrics across models
        evaluation['aggregate_metrics'] = self._calculate_aggregate_metrics(
            evaluation['model_evaluations']
        )
        
        return evaluation
    
    def _evaluate_single_model(self, model_result: Dict[str, Any],
                              ground_truth: Dict[str, Any],
                              node_id: str) -> Dict[str, Any]:
        """Evaluate a single model's predictions using LLM extraction and real ground truth."""
        model_name = model_result.get('model', 'unknown')

        # Get ground truth questions for this node
        node_questions = self.ground_truth.get(node_id, [])

        # Build a mapping from question text to per-question response
        # (new format: model_result has 'question_results' with per-question responses)
        per_question_responses = {}
        if 'question_results' in model_result:
            for qr in model_result['question_results']:
                if qr.get('success') and qr.get('question_type') == 'verifiable':
                    per_question_responses[qr['question']] = qr.get('response', '')

        # Fallback: concatenated response for old-format results
        response_text = model_result.get('response') or ''

        # Extract probability for the node's primary question using LLM
        node_desc = NODE_DESCRIPTIONS.get(node_id, '')
        primary_question = f"What is the probability that '{node_desc}' will occur within the next 24 hours?"
        extraction = self.extractor.extract_probability(
            response_text, primary_question, model_name, node_id
        )

        # Build per-question predictions using LLM extraction
        # Use per-question response if available (paper protocol), else fall back to full response
        question_results = []
        for q in node_questions:
            q_response = per_question_responses.get(q["question"], response_text)
            q_extraction = self.extractor.extract_probability(
                q_response, q["question"], model_name, node_id + f"_q{q['id']}"
            )
            question_results.append({
                'question_id': q['id'],
                'question': q['question'],
                'predicted_probability': q_extraction['probability'],
                'ground_truth': 1.0 if q['ground_truth'] else 0.0,
                'raw_quote': q_extraction.get('raw_quote'),
                'source': q_extraction.get('source'),
            })

        # Calculate 1-MAE and Brier score from question results
        valid_results = [r for r in question_results if r['predicted_probability'] is not None]

        accuracy_scores = {}
        calibration_metrics = {}

        if valid_results:
            predictions = [r['predicted_probability'] for r in valid_results]
            outcomes = [r['ground_truth'] for r in valid_results]

            # 1-MAE (paper's metric)
            mae = np.mean([abs(p - o) for p, o in zip(predictions, outcomes)])
            one_mae = 1.0 - mae

            # Brier score
            brier = np.mean([(p - o) ** 2 for p, o in zip(predictions, outcomes)])

            # Binary accuracy (predict > 0.5 = event occurs)
            binary_correct = sum(
                1 for p, o in zip(predictions, outcomes)
                if (p > 0.5 and o == 1.0) or (p <= 0.5 and o == 0.0)
            )
            binary_accuracy = binary_correct / len(valid_results)

            accuracy_scores = {
                'overall_binary_accuracy': binary_accuracy,
                'overall_brier_score': brier,
                'one_mae': one_mae,
                'mae': mae,
                'questions_evaluated': len(valid_results),
            }
            calibration_metrics = {
                'expected_calibration_error': mae,
                'mean_confidence': np.mean(predictions),
                'one_mae': one_mae,
            }

        evaluation = {
            'model': model_name,
            'response_length': len(response_text),
            'primary_probability': extraction.get('probability'),
            'primary_source': extraction.get('source'),
            'question_results': question_results,
            'probability_predictions': [],  # legacy compat
            'scenario_assessments': self._extract_scenario_assessments(response_text),
            'accuracy_scores': accuracy_scores,
            'calibration_metrics': calibration_metrics,
            'qualitative_assessment': self._assess_reasoning_quality(response_text, ground_truth)
        }

        return evaluation
    
    def _extract_scenario_assessments(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract qualitative scenario assessments."""
        assessments = []
        
        # Look for structured analysis sections
        sections = self._split_response_sections(response_text)
        
        for section_title, content in sections:
            assessment = {
                'section': section_title,
                'key_points': self._extract_key_points(content),
                'sentiment': self._assess_section_sentiment(content),
                'confidence_indicators': self._extract_confidence_indicators(content)
            }
            assessments.append(assessment)
        
        return assessments
    
    def _extract_confidence_level(self, position: int, text: str, window: int = 200) -> str:
        """Extract confidence level near a probability prediction."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end].lower()
        
        if any(term in context for term in ['high confidence', 'very confident', 'certain']):
            return 'high'
        elif any(term in context for term in ['low confidence', 'uncertain', 'unclear']):
            return 'low'
        else:
            return 'medium'
    
    def _extract_context(self, position: int, text: str, window: int) -> str:
        """Extract context around a position in text."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()
    
    def _split_response_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split response into titled sections."""
        sections = []
        
        # Look for markdown headers or numbered sections
        section_pattern = r'^(#{1,3}\s*.+|^\d+\.\s*.+)$'
        lines = text.split('\n')
        
        current_section = "introduction"
        current_content = []
        
        for line in lines:
            if re.match(section_pattern, line, re.MULTILINE):
                # Save previous section
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                
                # Start new section
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save final section
        if current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        return sections
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from section content."""
        # Look for bullet points, numbered lists, or strong statements
        patterns = [
            r'^\s*[-•]\s*(.+)$',  # Bullet points
            r'^\s*\d+\.\s*(.+)$',  # Numbered lists
            r'\*\*(.+?)\*\*',      # Bold text
        ]
        
        key_points = []
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                point = match.group(1).strip()
                if len(point) > 10:  # Filter out very short points
                    key_points.append(point)
        
        return key_points[:5]  # Limit to top 5 key points
    
    def _assess_section_sentiment(self, content: str) -> str:
        """Assess the sentiment/tone of a section."""
        escalation_terms = ['escalation', 'threat', 'risk', 'danger', 'conflict', 'attack']
        deescalation_terms = ['peaceful', 'diplomatic', 'stable', 'calm', 'negotiation']
        
        content_lower = content.lower()
        
        escalation_count = sum(1 for term in escalation_terms if term in content_lower)
        deescalation_count = sum(1 for term in deescalation_terms if term in content_lower)
        
        if escalation_count > deescalation_count:
            return 'escalatory'
        elif deescalation_count > escalation_count:
            return 'de-escalatory'
        else:
            return 'neutral'
    
    def _extract_confidence_indicators(self, content: str) -> List[str]:
        """Extract confidence indicators from content."""
        high_confidence = ['certain', 'confident', 'clear evidence', 'strong indicators']
        low_confidence = ['uncertain', 'unclear', 'limited data', 'insufficient evidence']
        
        content_lower = content.lower()
        indicators = []
        
        for term in high_confidence:
            if term in content_lower:
                indicators.append(f"high: {term}")
        
        for term in low_confidence:
            if term in content_lower:
                indicators.append(f"low: {term}")
        
        return indicators
    
    def _assess_reasoning_quality(self, response_text: str, 
                                ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of reasoning in the response."""
        assessment = {
            'evidence_usage': self._assess_evidence_usage(response_text),
            'logical_structure': self._assess_logical_structure(response_text),
            'specificity': self._assess_specificity(response_text),
            'uncertainty_handling': self._assess_uncertainty_handling(response_text)
        }
        
        return assessment
    
    def _assess_evidence_usage(self, text: str) -> Dict[str, Any]:
        """Assess how well the response uses evidence."""
        data_references = len(re.findall(r'(data|evidence|indicator|signal)', text, re.IGNORECASE))
        specific_numbers = len(re.findall(r'\$\d+|\d+%|\d+\.\d+', text))
        
        return {
            'data_references': data_references,
            'specific_numbers': specific_numbers,
            'score': min(1.0, (data_references + specific_numbers) / 20.0)
        }
    
    def _assess_logical_structure(self, text: str) -> Dict[str, Any]:
        """Assess the logical structure of the response."""
        # Look for reasoning connectors
        connectors = ['because', 'therefore', 'however', 'furthermore', 'consequently']
        connector_count = sum(1 for conn in connectors if conn in text.lower())
        
        # Look for structured sections
        sections = len(re.findall(r'^(#{1,3}|\d+\.|\*\*)', text, re.MULTILINE))
        
        return {
            'reasoning_connectors': connector_count,
            'structured_sections': sections,
            'score': min(1.0, (connector_count + sections) / 15.0)
        }
    
    def _assess_specificity(self, text: str) -> Dict[str, Any]:
        """Assess specificity of predictions."""
        specific_terms = ['within 24 hours', 'next 48 hours', 'by tomorrow', 'specific', 'exactly']
        specificity_count = sum(1 for term in specific_terms if term in text.lower())
        
        vague_terms = ['might', 'could', 'possibly', 'perhaps', 'maybe']
        vague_count = sum(1 for term in vague_terms if term in text.lower())
        
        return {
            'specific_terms': specificity_count,
            'vague_terms': vague_count,
            'score': max(0.0, min(1.0, (specificity_count - vague_count) / 10.0))
        }
    
    def _assess_uncertainty_handling(self, text: str) -> Dict[str, Any]:
        """Assess how well uncertainty is handled."""
        uncertainty_terms = ['uncertainty', 'unknown', 'unclear', 'limited data', 'confidence']
        uncertainty_count = sum(1 for term in uncertainty_terms if term in text.lower())
        
        confidence_terms = ['high confidence', 'medium confidence', 'low confidence']
        confidence_count = sum(1 for term in confidence_terms if term in text.lower())
        
        return {
            'uncertainty_mentions': uncertainty_count,
            'confidence_levels': confidence_count,
            'score': min(1.0, (uncertainty_count + confidence_count) / 8.0)
        }
    
    def _summarize_ground_truth(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize ground truth events."""
        summary = {
            'event_count': 0,
            'event_types': [],
            'severity_indicators': {},
            'temporal_distribution': {}
        }
        
        if 'events' in ground_truth:
            events = ground_truth['events']
            summary['event_count'] = len(events)
            
            # Categorize event types
            for event in events:
                event_type = self._categorize_event(event.get('description', ''))
                summary['event_types'].append(event_type)
                
                # Severity indicators
                fatalities = event.get('fatalities', 0)
                if fatalities > 0:
                    severity = 'high' if fatalities > 50 else 'medium' if fatalities > 10 else 'low'
                    summary['severity_indicators'][event.get('date', 'unknown')] = severity
        
        return summary
    
    def _categorize_event(self, description: str) -> str:
        """Categorize an event based on its description."""
        desc_lower = description.lower()
        
        if any(term in desc_lower for term in ['nuclear', 'facility', 'reactor']):
            return 'nuclear_incident'
        elif any(term in desc_lower for term in ['strike', 'attack', 'missile']):
            return 'military_action'
        elif any(term in desc_lower for term in ['oil', 'energy', 'production']):
            return 'economic_disruption'
        elif any(term in desc_lower for term in ['evacuation', 'diplomatic']):
            return 'diplomatic_action'
        else:
            return 'other'
    
    def _calculate_aggregate_metrics(self, model_evaluations: Dict[str, Any]) -> Dict[str, float]:
        """Calculate aggregate metrics across all models."""
        if not model_evaluations:
            return {}
        
        # Collect metrics from all models
        all_accuracy_scores = []
        all_brier_scores = []
        all_calibration_errors = []
        
        for model_eval in model_evaluations.values():
            accuracy_scores = model_eval.get('accuracy_scores', {})
            
            # Binary accuracy scores
            binary_accuracies = [v for k, v in accuracy_scores.items() if 'binary_accuracy' in k]
            all_accuracy_scores.extend(binary_accuracies)
            
            # Brier scores
            brier_scores = [v for k, v in accuracy_scores.items() if 'brier_score' in k]
            all_brier_scores.extend(brier_scores)
            
            # Calibration errors
            calib_metrics = model_eval.get('calibration_metrics', {})
            if 'expected_calibration_error' in calib_metrics:
                all_calibration_errors.append(calib_metrics['expected_calibration_error'])
        
        aggregate = {}
        
        if all_accuracy_scores:
            aggregate['mean_accuracy'] = np.mean(all_accuracy_scores)
            aggregate['std_accuracy'] = np.std(all_accuracy_scores)
        
        if all_brier_scores:
            aggregate['mean_brier_score'] = np.mean(all_brier_scores)
            aggregate['std_brier_score'] = np.std(all_brier_scores)
        
        if all_calibration_errors:
            aggregate['mean_calibration_error'] = np.mean(all_calibration_errors)
            aggregate['std_calibration_error'] = np.std(all_calibration_errors)
        
        return aggregate
    
    def evaluate_sequence(self, sequence_results: Dict[str, Any],
                         ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an entire temporal sequence.
        
        Args:
            sequence_results: Results from run_inference temporal sequence
            ground_truth_data: Ground truth events for all nodes
            
        Returns:
            Complete sequence evaluation
        """
        self.logger.info("Evaluating temporal sequence")
        
        sequence_evaluation = {
            'sequence_id': sequence_results.get('sequence_id'),
            'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
            'node_evaluations': {},
            'sequence_metrics': {},
            'model_comparison': {},
            'thematic_analysis': {}
        }
        
        # Evaluate each node
        for node_id, node_results in sequence_results.get('node_results', {}).items():
            node_ground_truth = ground_truth_data.get(node_id, {})
            
            node_eval = self.evaluate_node_predictions(node_results, node_ground_truth)
            sequence_evaluation['node_evaluations'][node_id] = node_eval
        
        # Calculate sequence-level metrics
        sequence_evaluation['sequence_metrics'] = self._calculate_sequence_metrics(
            sequence_evaluation['node_evaluations']
        )
        
        # Model comparison across sequence
        sequence_evaluation['model_comparison'] = self._compare_models_across_sequence(
            sequence_evaluation['node_evaluations']
        )
        
        # Thematic analysis
        sequence_evaluation['thematic_analysis'] = self._analyze_themes(
            sequence_evaluation['node_evaluations']
        )
        
        return sequence_evaluation
    
    def _calculate_sequence_metrics(self, node_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics across the entire temporal sequence."""
        # Aggregate accuracy over time
        temporal_accuracy = {}
        temporal_calibration = {}
        
        for node_id, node_eval in node_evaluations.items():
            target_date = node_eval.get('target_date')
            if target_date:
                agg_metrics = node_eval.get('aggregate_metrics', {})
                if 'mean_accuracy' in agg_metrics:
                    temporal_accuracy[target_date] = agg_metrics['mean_accuracy']
                if 'mean_calibration_error' in agg_metrics:
                    temporal_calibration[target_date] = agg_metrics['mean_calibration_error']
        
        return {
            'temporal_accuracy_trend': temporal_accuracy,
            'temporal_calibration_trend': temporal_calibration,
            'overall_sequence_accuracy': np.mean(list(temporal_accuracy.values())) if temporal_accuracy else 0.0,
            'accuracy_improvement_rate': self._calculate_trend_slope(temporal_accuracy),
            'total_nodes_evaluated': len(node_evaluations)
        }
    
    def _compare_models_across_sequence(self, node_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Compare model performance across the sequence."""
        model_performance = {}
        
        # Collect performance metrics for each model
        all_models = set()
        for node_eval in node_evaluations.values():
            all_models.update(node_eval.get('model_evaluations', {}).keys())
        
        for model in all_models:
            model_stats = {
                'nodes_evaluated': 0,
                'accuracy_scores': [],
                'brier_scores': [],
                'one_mae_scores': [],
                'reasoning_scores': []
            }

            for node_eval in node_evaluations.values():
                model_eval = node_eval.get('model_evaluations', {}).get(model)
                if model_eval:
                    model_stats['nodes_evaluated'] += 1

                    accuracy_scores = model_eval.get('accuracy_scores', {})
                    if 'overall_binary_accuracy' in accuracy_scores:
                        model_stats['accuracy_scores'].append(accuracy_scores['overall_binary_accuracy'])
                    if 'overall_brier_score' in accuracy_scores:
                        model_stats['brier_scores'].append(accuracy_scores['overall_brier_score'])
                    if 'one_mae' in accuracy_scores:
                        model_stats['one_mae_scores'].append(accuracy_scores['one_mae'])

                    qual_assess = model_eval.get('qualitative_assessment', {})
                    reasoning_score = np.mean([
                        qual_assess.get('evidence_usage', {}).get('score', 0),
                        qual_assess.get('logical_structure', {}).get('score', 0),
                        qual_assess.get('specificity', {}).get('score', 0),
                        qual_assess.get('uncertainty_handling', {}).get('score', 0)
                    ])
                    model_stats['reasoning_scores'].append(reasoning_score)

            model_performance[model] = {
                'nodes_evaluated': model_stats['nodes_evaluated'],
                'mean_accuracy': np.mean(model_stats['accuracy_scores']) if model_stats['accuracy_scores'] else 0.0,
                'mean_brier_score': np.mean(model_stats['brier_scores']) if model_stats['brier_scores'] else 0.0,
                'mean_one_mae': np.mean(model_stats['one_mae_scores']) if model_stats['one_mae_scores'] else 0.0,
                'mean_reasoning_score': np.mean(model_stats['reasoning_scores']) if model_stats['reasoning_scores'] else 0.0,
                'consistency': 1.0 - np.std(model_stats['accuracy_scores']) if model_stats['accuracy_scores'] else 0.0
            }
        
        return model_performance
    
    def _analyze_themes(self, node_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by evaluation themes."""
        theme_analysis = {}
        
        for theme in EVALUATION_THEMES:
            theme_analysis[theme] = {
                'relevant_nodes': [],
                'accuracy_scores': [],
                'common_patterns': []
            }
        
        # Map nodes to themes per paper Table 8
        # Theme I: Initial Outbreak (T0, T1, T2)
        # Theme II: Threshold Crossings (T3, T6, T7, T8)
        # Theme III: Economic Shockwaves (T4, T5)
        # Theme IV: Political Signaling (T9, T10)
        theme_mapping = {
            'T0': 'Initial Outbreak',
            'T1': 'Initial Outbreak',
            'T2': 'Initial Outbreak',
            'T3': 'Threshold Crossings',
            'T4': 'Economic Shockwaves',
            'T5': 'Economic Shockwaves',
            'T6': 'Threshold Crossings',
            'T7': 'Threshold Crossings',
            'T8': 'Threshold Crossings',
            'T9': 'Political Signaling',
            'T10': 'Political Signaling',
        }
        
        for node_id, node_eval in node_evaluations.items():
            theme = theme_mapping.get(node_id, 'Other')
            if theme in theme_analysis:
                theme_analysis[theme]['relevant_nodes'].append(node_id)
                
                agg_metrics = node_eval.get('aggregate_metrics', {})
                if 'mean_accuracy' in agg_metrics:
                    theme_analysis[theme]['accuracy_scores'].append(agg_metrics['mean_accuracy'])
        
        # Calculate theme-level statistics
        for theme, data in theme_analysis.items():
            if data['accuracy_scores']:
                data['mean_accuracy'] = np.mean(data['accuracy_scores'])
                data['std_accuracy'] = np.std(data['accuracy_scores'])
            else:
                data['mean_accuracy'] = 0.0
                data['std_accuracy'] = 0.0
        
        return theme_analysis
    
    def _calculate_trend_slope(self, temporal_data: Dict[str, float]) -> float:
        """Calculate the slope of a temporal trend."""
        if len(temporal_data) < 2:
            return 0.0
        
        try:
            # Sort by timestamp
            sorted_items = sorted(temporal_data.items())
            x = np.arange(len(sorted_items))
            y = np.array([item[1] for item in sorted_items])
            
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        except:
            return 0.0
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a markdown evaluation report."""
        report = f"""# Fog of War Pipeline Evaluation Report

**Generated:** {evaluation_results.get('evaluation_timestamp', 'Unknown')}
**Sequence ID:** {evaluation_results.get('sequence_id', 'Unknown')}

## Executive Summary

"""
        
        sequence_metrics = evaluation_results.get('sequence_metrics', {})
        
        report += f"- **Overall Accuracy:** {sequence_metrics.get('overall_sequence_accuracy', 0.0):.2%}\n"
        report += f"- **Nodes Evaluated:** {sequence_metrics.get('total_nodes_evaluated', 0)}\n"
        report += f"- **Accuracy Trend:** {'Improving' if sequence_metrics.get('accuracy_improvement_rate', 0) > 0 else 'Declining' if sequence_metrics.get('accuracy_improvement_rate', 0) < 0 else 'Stable'}\n\n"
        
        # Model comparison
        report += "## Model Performance Comparison\n\n"
        model_comparison = evaluation_results.get('model_comparison', {})
        
        if model_comparison:
            report += "| Model | 1-MAE | Accuracy | Brier Score | Reasoning | Consistency |\n"
            report += "|-------|-------|----------|-------------|-----------|-------------|\n"

            for model, metrics in model_comparison.items():
                report += f"| {model} | {metrics.get('mean_one_mae', 0):.3f} | {metrics.get('mean_accuracy', 0):.2%} | {metrics.get('mean_brier_score', 0):.3f} | {metrics.get('mean_reasoning_score', 0):.2f} | {metrics.get('consistency', 0):.2f} |\n"
        
        # Thematic analysis
        report += "\n## Thematic Analysis\n\n"
        theme_analysis = evaluation_results.get('thematic_analysis', {})
        
        for theme, data in theme_analysis.items():
            if data.get('relevant_nodes'):
                report += f"### {theme}\n"
                report += f"- **Nodes:** {', '.join(data['relevant_nodes'])}\n"
                report += f"- **Mean Accuracy:** {data.get('mean_accuracy', 0):.2%}\n"
                report += f"- **Std Deviation:** {data.get('std_accuracy', 0):.3f}\n\n"
        
        # Node-by-node results
        report += "## Node-by-Node Results\n\n"
        node_evaluations = evaluation_results.get('node_evaluations', {})
        
        for node_id, node_eval in sorted(node_evaluations.items()):
            target_date = node_eval.get('target_date', 'Unknown')
            report += f"### {node_id} - {target_date}\n"
            
            agg_metrics = node_eval.get('aggregate_metrics', {})
            if agg_metrics:
                report += f"- **Accuracy:** {agg_metrics.get('mean_accuracy', 0):.2%}\n"
                report += f"- **Brier Score:** {agg_metrics.get('mean_brier_score', 0):.3f}\n"
                report += f"- **Calibration Error:** {agg_metrics.get('mean_calibration_error', 0):.3f}\n"
            
            # Ground truth summary
            gt_summary = node_eval.get('ground_truth_summary', {})
            if gt_summary.get('event_count', 0) > 0:
                report += f"- **Actual Events:** {gt_summary['event_count']}\n"
            
            report += "\n"
        
        report += """## Methodology Notes

- **Binary Accuracy:** Percentage of correct binary predictions (event occurs/doesn't occur)
- **Brier Score:** Mean squared difference between predicted probabilities and actual outcomes (lower is better)
- **Calibration Error:** Average absolute difference between predicted probabilities and actual frequencies
- **Reasoning Score:** Composite score based on evidence usage, logical structure, specificity, and uncertainty handling

## Methodology Notes (cont.)

- **1-MAE (Calibration Consistency):** 1 minus mean absolute error between predicted probability and binary outcome (paper's primary metric, higher is better)
- Probability extraction via LLM only (qualitative language mapped to calibrated probabilities)
- Ground truth from 42 verifiable questions with binary labels
"""
        
        return report
    
    def save_evaluation_results(self, evaluation_results: Dict[str, Any], 
                               output_path: Optional[Path] = None):
        """Save evaluation results to disk."""
        if not output_path:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            output_path = OUTPUT_DIR / f"evaluation_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Also save markdown report
        report = self.generate_evaluation_report(evaluation_results)
        report_path = output_path.with_suffix('.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Evaluation results saved to {output_path}")
        self.logger.info(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    # Test the evaluator
    import logging
    from datetime import datetime, timezone
    
    logging.basicConfig(level=logging.INFO)
    
    # Create sample prediction and ground truth data
    sample_predictions = {
        'node_id': 'T1',
        'target_date': '2026-02-28T00:00:00+00:00',
        'model_results': {
            'claude-3.5-sonnet': {
                'success': True,
                'response': 'Based on the intelligence briefing, I assess a 75% probability of military escalation within 24 hours. High confidence in this assessment due to convergent tactical and economic indicators.',
                'model': 'claude-3.5-sonnet'
            }
        }
    }
    
    sample_ground_truth = {
        'events': [
            {
                'description': 'Israeli-US coordinated strikes on Iranian facilities',
                'date': '2026-02-28',
                'fatalities': 15
            }
        ]
    }
    
    evaluator = Evaluator()
    evaluation = evaluator.evaluate_node_predictions(sample_predictions, sample_ground_truth)
    
    print("Sample evaluation completed:")
    print(f"- Model evaluated: {list(evaluation['model_evaluations'].keys())}")
    print(f"- Aggregate accuracy: {evaluation['aggregate_metrics'].get('mean_accuracy', 0):.2%}")
    
    # Generate report
    sample_sequence_eval = {
        'sequence_id': 'test_sequence',
        'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
        'node_evaluations': {'T1': evaluation},
        'sequence_metrics': {'overall_sequence_accuracy': 0.75, 'total_nodes_evaluated': 1},
        'model_comparison': {
            'claude-3.5-sonnet': {'mean_accuracy': 0.75, 'mean_brier_score': 0.15}
        },
        'thematic_analysis': {}
    }
    
    report = evaluator.generate_evaluation_report(sample_sequence_eval)
    print("\nSample report generated (first 500 chars):")
    print(report[:500] + "...")