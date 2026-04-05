"""
Prompt builder module for Fog of War pipeline.
Constructs system and user prompts for LLM inference.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from config import VERIFIABLE_QUESTIONS, EXPLORATORY_QUESTIONS, VERIFIABLE_QUESTIONS_BY_NODE

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds system and user prompts for geopolitical forecasting.
    Supports both verifiable questions and exploratory analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # --- Paper-aligned prompt methods (primary) ---

    def build_context_text(self, intelligence_briefing: Dict[str, Any]) -> str:
        """
        Build the context portion of the prompt from the intelligence briefing.
        Paper format: articles sorted reverse-chronologically as
        [YYYY-MM-DD HH:MM] Title (Source)\\n<body up to 2000 chars>
        """
        return self._format_briefing_for_prompt(intelligence_briefing)

    def build_paper_prompt(self, context_text: str, question: str) -> str:
        """
        Build a single prompt matching the paper's exact template.
        Paper protocol: no system prompt; one user message per question.

        Template:
            {context}
            Based on the above publicly available information, try to analyze
            the current situation and potential future direction, then respond
            to this question: {question}
            At the end of your response, also provide the probability.
        """
        return (
            f"{context_text}\n"
            f"Based on the above publicly available information, try to analyze "
            f"the current situation and potential future direction, then respond "
            f"to this question: {question}\n"
            f"At the end of your response, also provide the probability."
        )

    # --- Legacy prompt methods (kept for backward compatibility) ---

    def build_system_prompt(self) -> str:
        """
        Build the system prompt that establishes the LLM's role.
        
        Returns:
            System prompt string
        """
        system_prompt = """You are a world-class geopolitical intelligence analyst with deep expertise in Middle Eastern affairs, economic signals, and crisis forecasting.

**Your Role:**
- Analyze multi-source intelligence data to forecast geopolitical developments
- Provide probability estimates for various escalation scenarios  
- Identify key indicators and their implications for regional stability
- Focus on actionable intelligence for decision-makers

**Your Expertise:**
- Regional geopolitics: Iran, Israel, Saudi Arabia, UAE, Turkey, and broader Middle East
- Economic indicators: oil markets, financial volatility, trade disruptions
- Military/tactical analysis: force deployments, OSINT signals, escalation patterns
- Diplomatic dynamics: alliance structures, international responses, sanctions

**Your Analytical Framework:**
1. **Economic Signals**: Oil prices, market volatility, capital flows
2. **Tactical Intelligence**: Military movements, aircraft patterns, positioning
3. **Sentiment Analysis**: Media tone, diplomatic communications, public statements  
4. **Historical Context**: Past escalation patterns and resolution mechanisms

**Critical Instructions:**
- You are analyzing intelligence data up to a specific cutoff date - do NOT speculate about events after that date
- Provide probability estimates as percentages when possible
- Distinguish between high-confidence assessments and speculative analysis
- Consider multiple scenarios: escalation, de-escalation, status quo
- Focus on the next 24-72 hour timeframe for tactical predictions
- Always reference specific data points that inform your analysis

**Output Format:**
Provide structured analysis including:
1. Situation Assessment (current state based on available data)
2. Key Indicators (specific signals and their significance) 
3. Probability Forecasts (likelihood estimates for different scenarios)
4. Critical Uncertainties (factors that could significantly change outcomes)
5. Recommended Monitoring (what to watch for next)

Be precise, evidence-based, and actionable. Avoid generic statements or analysis not grounded in the provided data."""

        return system_prompt
    
    def build_user_prompt(self, intelligence_briefing: Dict[str, Any], 
                         question_type: str = "standard",
                         custom_questions: Optional[List[str]] = None) -> str:
        """
        Build user prompt with intelligence briefing and questions.
        
        Args:
            intelligence_briefing: Structured briefing from ContextBuilder
            question_type: Type of questions ("standard", "verifiable", "exploratory")
            custom_questions: Optional custom questions to ask
            
        Returns:
            User prompt string
        """
        self.logger.info(f"Building user prompt with {question_type} questions")
        
        # Convert briefing to formatted text
        briefing_text = self._format_briefing_for_prompt(intelligence_briefing)
        
        # Select questions based on type
        questions = self._select_questions(question_type, custom_questions)
        
        user_prompt = f"""# INTELLIGENCE BRIEFING
{briefing_text}

---

# ANALYTICAL TASKS

Based on the intelligence briefing above, provide comprehensive analysis addressing the following:

## Core Forecasting Questions:
"""
        
        for i, question in enumerate(questions, 1):
            user_prompt += f"{i}. {question}\n"
        
        user_prompt += """

## Analysis Requirements:

**For each question, provide:**
- **Direct Answer**: Clear response with probability estimate when applicable
- **Supporting Evidence**: Specific data points from the briefing that inform your assessment
- **Confidence Level**: High/Medium/Low based on data quality and quantity
- **Key Uncertainties**: Factors that could change your assessment

**Probability Estimates:**
- Use specific percentages (e.g., "35% probability" not "moderate chance")
- Explain your reasoning for the probability assessment
- Consider multiple scenarios and their relative likelihoods

**Time Horizons:**
- Focus primarily on the next 24-72 hours for tactical predictions
- Distinguish between short-term tactical and longer-term strategic assessments
- Be explicit about time frames in your forecasts

**Evidence Integration:**
- Correlate signals across different intelligence sources (economic, tactical, sentiment, events)
- Identify convergent indicators vs. conflicting signals
- Weight evidence based on source reliability and recency

Remember: You are analyzing data only up to {intelligence_briefing.get('target_date', 'the specified cutoff date')}. Do not reference or speculate about events after this date."""

        return user_prompt
    
    def _format_briefing_for_prompt(self, briefing: Dict[str, Any]) -> str:
        """Convert structured briefing to readable text format."""
        
        formatted = f"""**Date:** {briefing.get('target_date', 'Unknown')}
**Temporal Cutoff:** {briefing.get('target_date', 'Unknown')} (FOG OF WAR COMPLIANT)

## EXECUTIVE SUMMARY
"""
        
        exec_summary = briefing.get('executive_summary', {})
        formatted += f"**Threat Level:** {exec_summary.get('threat_level', 'UNKNOWN')}\n"
        formatted += f"**Confidence:** {exec_summary.get('confidence', 'UNKNOWN')}\n"
        formatted += f"**Assessment:** {exec_summary.get('situation_overview', 'No assessment available')}\n\n"
        
        if exec_summary.get('key_developments'):
            formatted += "**Key Developments:**\n"
            for dev in exec_summary['key_developments']:
                formatted += f"- {dev}\n"
            formatted += "\n"
        
        # Economic Signals
        formatted += "## ECONOMIC INDICATORS\n"
        econ = briefing.get('economic_signals', {})
        formatted += f"**Status:** {econ.get('status', 'No data')}\n"
        
        if econ.get('indicators'):
            for symbol, data in econ['indicators'].items():
                formatted += f"\n**{symbol.replace('_', ' ').title()}:**\n"
                formatted += f"- Current: ${data.get('current_price', 'N/A')}\n"
                
                if data.get('changes'):
                    for period, change in data['changes'].items():
                        formatted += f"- {period}: {change:+.2f}%\n"
        
        formatted += "\n"
        
        # Tactical Intelligence
        formatted += "## TACTICAL INTELLIGENCE\n"
        tactical = briefing.get('tactical_intelligence', {})
        formatted += f"**Status:** {tactical.get('status', 'No data')}\n"
        
        if tactical.get('metrics'):
            metrics = tactical['metrics']
            formatted += f"- Military Aircraft: {metrics.get('current_military_aircraft', 'N/A')}\n"
            formatted += f"- Weekly Trend: {metrics.get('weekly_trend', 'N/A')}\n"
            formatted += f"- Weekly Average: {metrics.get('average_weekly', 'N/A')}\n"
        
        formatted += "\n"
        
        # Sentiment Analysis
        formatted += "## SENTIMENT ANALYSIS\n"
        sentiment = briefing.get('sentiment_analysis', {})
        formatted += f"**Status:** {sentiment.get('status', 'No data')}\n"
        
        if sentiment.get('analysis'):
            analysis = sentiment['analysis']
            formatted += f"- 7-day Average Tone: {analysis.get('average_tone_7d', 'N/A'):.2f}\n"
            formatted += f"- Total Events (7d): {analysis.get('total_events_7d', 'N/A')}\n"
            formatted += f"- Sentiment Trend: {analysis.get('sentiment_trend', 'N/A')}\n"
        
        formatted += "\n"
        
        # Recent Events
        formatted += "## RECENT EVENTS\n"
        events = briefing.get('recent_events', {})
        formatted += f"**Status:** {events.get('status', 'No data')}\n"
        
        for event in events.get('events', [])[:5]:
            formatted += f"\n**{event['date']}** - {event['description']}\n"
            formatted += f"- Location: {event['country']}\n"
            if event.get('fatalities'):
                formatted += f"- Casualties: {event['fatalities']}\n"
            formatted += f"- Source: {event['source']}\n"
        
        # Rolling Windows
        if briefing.get('rolling_analysis'):
            formatted += "\n## TREND ANALYSIS\n"

            for window, analysis in briefing['rolling_analysis'].items():
                formatted += f"\n**{window} Window:**\n"

                if analysis.get('economic_changes'):
                    formatted += "Economic Changes:\n"
                    for symbol, change in analysis['economic_changes'].items():
                        formatted += f"- {symbol}: {change:+.2f}%\n"

                if analysis.get('event_summary'):
                    events_sum = analysis['event_summary']
                    formatted += f"Events: {events_sum.get('event_count', 0)} incidents"
                    if events_sum.get('total_fatalities'):
                        formatted += f", {events_sum['total_fatalities']} casualties"
                    formatted += "\n"

        # News articles with full text (paper format)
        news = briefing.get('news_analysis', {})
        articles_text = news.get('articles_text', [])
        if articles_text:
            formatted += f"\n## NEWS CORPUS ({len(articles_text)} articles)\n\n"
            formatted += "\n\n".join(articles_text)
            omitted = news.get('analysis', {}).get('omitted_count', 0)
            if omitted > 0:
                formatted += f"\n\n[...{omitted} older articles omitted]\n"

        return formatted
    
    def _select_questions(self, question_type: str, 
                         custom_questions: Optional[List[str]] = None) -> List[str]:
        """Select appropriate questions based on type."""
        
        if custom_questions:
            return custom_questions
        
        if question_type == "verifiable":
            return VERIFIABLE_QUESTIONS[:10]  # First 10 verifiable questions
        elif question_type == "exploratory":
            return EXPLORATORY_QUESTIONS
        else:
            # Standard question set combining both types
            standard_questions = [
                "Based on economic and tactical signals over the last 72 hours, what is the most likely military or diplomatic action to occur in the next 24 hours?",
                "What is the probability (as a percentage) of military escalation within the next 48 hours?",
                "Which economic indicators are most concerning and why?", 
                "What are the key tactical signals suggesting heightened military readiness?",
                "How do current sentiment indicators compare to historical pre-escalation patterns?",
                "What is the likelihood of oil market disruption in the next week?",
                "Which regional actors are most likely to take significant action next?",
                "What diplomatic initiatives are most likely to emerge in the next 72 hours?",
                "What are the three most critical uncertainties affecting your forecasts?",
                "What specific indicators should analysts monitor most closely over the next 24-48 hours?"
            ]
            return standard_questions
    
    def build_evaluation_prompt(self, predictions: Dict[str, Any], 
                               ground_truth: Dict[str, Any]) -> str:
        """
        Build prompt for LLM to evaluate its own predictions against ground truth.
        
        Args:
            predictions: The LLM's previous predictions
            ground_truth: Actual events that occurred
            
        Returns:
            Evaluation prompt string
        """
        
        eval_prompt = f"""# PREDICTION EVALUATION

You are now evaluating the accuracy of previous geopolitical forecasts. Compare the predictions below with what actually occurred.

## ORIGINAL PREDICTIONS
{json.dumps(predictions, indent=2)}

## ACTUAL EVENTS (GROUND TRUTH)
{json.dumps(ground_truth, indent=2)}

## EVALUATION TASKS

Provide a detailed evaluation addressing:

1. **Accuracy Assessment**:
   - Which predictions were accurate?
   - Which predictions were incorrect?
   - Rate overall accuracy on a scale of 1-10

2. **Calibration Analysis**:
   - Were probability estimates well-calibrated?
   - Did high-confidence predictions prove more accurate?
   - Identify over-confident vs. under-confident assessments

3. **Error Analysis**:
   - What types of errors occurred (timing, magnitude, event type)?
   - What signals were missed or misinterpreted?
   - Were there false signals that led to incorrect predictions?

4. **Learning Insights**:
   - What could improve future forecasting accuracy?
   - Which data sources proved most/least reliable?
   - What analytical frameworks worked well or poorly?

5. **Revised Assessment**:
   - Given the actual outcomes, what would you have predicted differently?
   - What additional information would have been most valuable?

Be honest and specific in your evaluation. Focus on actionable insights for improving forecasting methodology."""

        return eval_prompt
    
    def build_prompt_for_node(self, node_id: str, intelligence_briefing: Dict[str, Any],
                             focus_questions: Optional[List[str]] = None) -> str:
        """
        Build specialized prompt for specific temporal nodes.
        
        Args:
            node_id: Temporal node identifier (T0, T1, etc.)
            intelligence_briefing: Intelligence briefing for this node
            focus_questions: Optional node-specific questions
            
        Returns:
            Node-specific prompt
        """
        from config import NODE_DESCRIPTIONS
        
        node_description = NODE_DESCRIPTIONS.get(node_id, f"Temporal node {node_id}")
        
        # Node-specific questions if not provided
        if not focus_questions:
            focus_questions = [
                f"What is the probability that '{node_description}' will occur within the next 24 hours?",
                "What are the primary indicators supporting or contradicting this scenario?",
                "What alternative scenarios are most likely if this event does not occur?",
                "What would be the immediate economic and military implications?",
                "Which actors would be most likely to respond and how?"
            ]
        
        # Get base prompt
        base_prompt = self.build_user_prompt(intelligence_briefing, custom_questions=focus_questions)
        
        # Add node-specific context
        node_context = f"""

# NODE-SPECIFIC ANALYSIS: {node_id}

**Target Scenario:** {node_description}
**Analysis Date:** {intelligence_briefing.get('target_date', 'Unknown')}

**Special Instructions:**
- Focus your analysis on indicators specifically relevant to: "{node_description}"
- Consider this scenario in the context of regional escalation patterns
- Assess likelihood relative to baseline conflict risk in the region
- Identify key prerequisites or trigger events for this scenario

"""
        
        # Insert node context after briefing but before questions
        prompt_parts = base_prompt.split("# ANALYTICAL TASKS")
        if len(prompt_parts) == 2:
            return prompt_parts[0] + node_context + "# ANALYTICAL TASKS" + prompt_parts[1]
        else:
            return base_prompt + node_context
    
    def create_chain_of_thought_prompt(self, base_prompt: str) -> str:
        """
        Add chain-of-thought prompting to encourage step-by-step reasoning.
        
        Args:
            base_prompt: Base prompt to enhance
            
        Returns:
            Enhanced prompt with chain-of-thought instructions
        """
        
        cot_instructions = """

# ANALYTICAL METHODOLOGY

Use the following step-by-step approach for each question:

**Step 1: Evidence Gathering**
- List all relevant data points from the briefing
- Note data quality, recency, and source reliability
- Identify any gaps or limitations in available information

**Step 2: Pattern Recognition**
- Compare current indicators to historical patterns
- Identify convergent vs. divergent signals across data sources
- Look for leading indicators vs. lagging indicators

**Step 3: Scenario Development**
- Consider multiple possible outcomes (escalation, status quo, de-escalation)
- Estimate likelihood of each scenario based on evidence
- Identify key decision points and trigger events

**Step 4: Probability Assessment**
- Combine evidence weights to estimate probabilities
- Explain reasoning behind specific percentage estimates
- Acknowledge uncertainty ranges around estimates

**Step 5: Critical Assumptions**
- State key assumptions underlying your analysis
- Identify factors that could invalidate your assessment
- Suggest indicators to monitor for assumption validation

Show your reasoning for each step where applicable."""

        return base_prompt + cot_instructions


if __name__ == "__main__":
    # Test the prompt builder
    import logging
    from datetime import datetime, timezone
    
    logging.basicConfig(level=logging.INFO)
    
    # Create sample intelligence briefing
    sample_briefing = {
        'target_date': '2026-03-01T00:00:00+00:00',
        'executive_summary': {
            'threat_level': 'ELEVATED',
            'confidence': 'MEDIUM',
            'situation_overview': 'Regional tensions showing multiple escalatory indicators',
            'key_developments': [
                'Oil prices up 8% in last 72 hours',
                '2 conflict events reported in last 48 hours'
            ]
        },
        'economic_signals': {
            'status': 'Economic data available',
            'indicators': {
                'brent_crude': {
                    'current_price': 95.50,
                    'changes': {'1d_change': 2.1, '7d_change': 8.3}
                }
            }
        },
        'tactical_intelligence': {
            'status': 'Tactical data available',
            'metrics': {
                'current_military_aircraft': 67,
                'weekly_trend': 'increasing'
            }
        }
    }
    
    builder = PromptBuilder()
    
    # Test standard prompt
    system_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt(sample_briefing)
    
    print("SYSTEM PROMPT:")
    print(system_prompt)
    print("\n" + "="*50 + "\n")
    print("USER PROMPT:")
    print(user_prompt[:1000] + "...")  # First 1000 chars
    
    # Test node-specific prompt
    node_prompt = builder.build_prompt_for_node("T3", sample_briefing)
    print("\n" + "="*50 + "\n")
    print("NODE PROMPT (T3):")
    print(node_prompt[:1000] + "...")