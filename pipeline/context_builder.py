"""
Context builder module for Fog of War pipeline.
Implements strict temporal gating and builds intelligence briefings.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from config import TEMPORAL_NODES, NODE_DESCRIPTIONS, MIDDLE_EAST_COUNTRIES

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds temporally-gated intelligence briefings for specific target dates.
    CRITICAL: Only includes data timestamped before or at target date T.
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize context builder with all available data.
        
        Args:
            data: Dictionary containing all data sources from DataFetcher
        """
        self.data = data
        self.logger = logging.getLogger(__name__)
    
    def build_context(self, target_date: datetime, 
                     include_rolling_windows: bool = True) -> Dict[str, Any]:
        """
        Build intelligence briefing for target date with strict temporal gating.
        
        Args:
            target_date: Target date T - only include data timestamped <= T
            include_rolling_windows: Whether to include 24h, 72h, 7d rolling windows
            
        Returns:
            Structured intelligence briefing dictionary
        """
        self.logger.info(f"Building context for {target_date} with temporal gating")
        
        # Strict temporal filtering - NEVER include data after target_date
        filtered_data = self._apply_temporal_gating(target_date)
        
        # Build structured briefing
        briefing = {
            'target_date': target_date.isoformat(),
            'briefing_timestamp': target_date.isoformat(),
            'temporal_constraints': {
                'strict_cutoff': target_date.isoformat(),
                'data_sources_filtered': True,
                'fog_of_war_compliance': True
            },
            'executive_summary': self._build_executive_summary(filtered_data, target_date),
            'economic_signals': self._build_economic_section(filtered_data, target_date),
            'tactical_intelligence': self._build_tactical_section(filtered_data, target_date),
            'sentiment_analysis': self._build_sentiment_section(filtered_data, target_date),
            'recent_events': self._build_events_section(filtered_data, target_date),
            'news_analysis': self._build_news_section(filtered_data, target_date)
        }
        
        if include_rolling_windows:
            briefing['rolling_analysis'] = self._build_rolling_windows(filtered_data, target_date)
        
        # Add metadata
        briefing['metadata'] = self._build_metadata(filtered_data, target_date)
        
        return briefing
    
    def _apply_temporal_gating(self, target_date: datetime) -> Dict[str, Any]:
        """
        Apply strict temporal gating - only include data <= target_date.
        This is the MOST CRITICAL function for "Fog of War" compliance.
        """
        self.logger.debug(f"Applying temporal gating for cutoff: {target_date}")
        
        filtered_data = {}
        
        # Filter DataFrames with timestamp columns
        df_sources = ['economic', 'osint', 'sentiment', 'ground_truth']
        
        for source in df_sources:
            if source in self.data and isinstance(self.data[source], pd.DataFrame):
                df = self.data[source]
                if not df.empty and 'timestamp' in df.columns:
                    # CRITICAL: Only include records where timestamp <= target_date
                    mask = pd.to_datetime(df['timestamp']) <= target_date
                    filtered_df = df[mask].copy()
                    filtered_data[source] = filtered_df
                    
                    excluded_count = len(df) - len(filtered_df)
                    if excluded_count > 0:
                        self.logger.debug(f"Excluded {excluded_count} future records from {source}")
                else:
                    filtered_data[source] = df
            else:
                filtered_data[source] = self.data.get(source, pd.DataFrame())
        
        # Filter news articles by publication date
        filtered_data['news_articles'] = self._filter_news_articles(target_date)
        
        self.logger.info(f"Temporal gating complete - all data <= {target_date}")
        return filtered_data
    
    def _filter_news_articles(self, target_date: datetime) -> Dict[str, List[Dict]]:
        """Filter news articles by publication date."""
        news_data = self.data.get('news_articles', {})
        filtered_news = {'google': [], 'gdelt': []}
        
        for source in ['google', 'gdelt']:
            articles = news_data.get(source, [])
            for article in articles:
                # Try to extract publication date from various fields
                pub_date = self._extract_article_date(article)
                if pub_date and pub_date <= target_date:
                    filtered_news[source].append(article)
        
        return filtered_news
    
    def _extract_article_date(self, article: Dict) -> Optional[datetime]:
        """Extract publication date from article metadata."""
        # Try various date fields common in news articles
        date_fields = ['published_date', 'date_published', 'pub_date', 'timestamp', 'date']
        
        for field in date_fields:
            if field in article and article[field]:
                try:
                    if isinstance(article[field], str):
                        return pd.to_datetime(article[field], utc=True)
                    elif isinstance(article[field], datetime):
                        return article[field]
                except:
                    continue
        
        return None
    
    def _build_executive_summary(self, data: Dict[str, Any], target_date: datetime) -> Dict[str, Any]:
        """Build executive summary section."""
        summary = {
            'situation_overview': 'Regional tensions in Middle East showing variable indicators',
            'key_developments': [],
            'threat_level': 'MODERATE',
            'confidence': 'MEDIUM'
        }
        
        # Check recent ground truth events
        gt_df = data.get('ground_truth', pd.DataFrame())
        if not gt_df.empty:
            recent_events = gt_df[
                gt_df['timestamp'] >= (target_date - timedelta(days=3))
            ]
            if len(recent_events) > 0:
                summary['threat_level'] = 'ELEVATED'
                summary['key_developments'].append(f"{len(recent_events)} conflict events in last 72 hours")
        
        # Check economic indicators
        econ_df = data.get('economic', pd.DataFrame())
        if not econ_df.empty:
            recent_econ = econ_df[
                econ_df['timestamp'] >= (target_date - timedelta(days=7))
            ]
            if not recent_econ.empty:
                brent_data = recent_econ[recent_econ['symbol'] == 'brent_crude']
                if not brent_data.empty:
                    price_change = (brent_data['close'].iloc[-1] / brent_data['close'].iloc[0] - 1) * 100
                    if abs(price_change) > 5:
                        summary['key_developments'].append(f"Oil prices {'rose' if price_change > 0 else 'fell'} {abs(price_change):.1f}% this week")
        
        return summary
    
    def _build_economic_section(self, data: Dict[str, Any], target_date: datetime) -> Dict[str, Any]:
        """Build economic signals section."""
        econ_data = data.get('economic', pd.DataFrame())
        
        if econ_data.empty:
            return {'status': 'No economic data available', 'indicators': {}}
        
        # Get latest available data for each symbol
        indicators = {}
        
        for symbol in ['brent_crude', 'tel_aviv_35']:
            symbol_data = econ_data[econ_data['symbol'] == symbol].sort_values('timestamp')
            
            if not symbol_data.empty:
                latest = symbol_data.iloc[-1]
                
                # Calculate changes over different periods
                changes = {}
                for days in [1, 7, 30]:
                    cutoff = target_date - timedelta(days=days)
                    historical = symbol_data[symbol_data['timestamp'] <= cutoff]
                    
                    if not historical.empty:
                        old_price = historical.iloc[-1]['close']
                        change_pct = (latest['close'] / old_price - 1) * 100
                        changes[f'{days}d_change'] = round(change_pct, 2)
                
                indicators[symbol] = {
                    'current_price': latest['close'],
                    'timestamp': latest['timestamp'].isoformat(),
                    'changes': changes,
                    'volume': latest['volume']
                }
        
        return {
            'status': 'Economic data available',
            'last_updated': target_date.isoformat(),
            'indicators': indicators
        }
    
    def _build_tactical_section(self, data: Dict[str, Any], target_date: datetime) -> Dict[str, Any]:
        """Build tactical intelligence section."""
        osint_data = data.get('osint', pd.DataFrame())
        
        if osint_data.empty:
            return {'status': 'No tactical data available', 'metrics': {}}
        
        # Get recent aircraft counts
        recent_data = osint_data[
            osint_data['timestamp'] >= (target_date - timedelta(days=7))
        ].sort_values('timestamp')
        
        if recent_data.empty:
            return {'status': 'No recent tactical data', 'metrics': {}}
        
        latest = recent_data.iloc[-1]
        
        # Calculate trend
        if len(recent_data) >= 2:
            trend_change = latest['military_aircraft_count'] - recent_data.iloc[0]['military_aircraft_count']
            trend = 'increasing' if trend_change > 5 else 'decreasing' if trend_change < -5 else 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'status': 'Tactical data available',
            'last_updated': latest['timestamp'].isoformat(),
            'metrics': {
                'current_military_aircraft': latest['military_aircraft_count'],
                'weekly_trend': trend,
                'average_weekly': recent_data['military_aircraft_count'].mean()
            }
        }
    
    def _build_sentiment_section(self, data: Dict[str, Any], target_date: datetime) -> Dict[str, Any]:
        """Build sentiment analysis section."""
        sentiment_data = data.get('sentiment', pd.DataFrame())
        
        if sentiment_data.empty:
            return {'status': 'No sentiment data available', 'analysis': {}}
        
        # Get recent sentiment data
        recent_sentiment = sentiment_data[
            sentiment_data['timestamp'] >= (target_date - timedelta(days=7))
        ].sort_values('timestamp')
        
        if recent_sentiment.empty:
            return {'status': 'No recent sentiment data', 'analysis': {}}
        
        analysis = {
            'average_tone_7d': recent_sentiment['average_tone'].mean(),
            'total_events_7d': recent_sentiment['event_count'].sum(),
            'daily_average_events': recent_sentiment['event_count'].mean(),
            'sentiment_trend': self._calculate_sentiment_trend(recent_sentiment)
        }
        
        return {
            'status': 'Sentiment data available',
            'last_updated': recent_sentiment.iloc[-1]['timestamp'].isoformat(),
            'analysis': analysis
        }
    
    def _calculate_sentiment_trend(self, sentiment_df: pd.DataFrame) -> str:
        """Calculate sentiment trend from recent data."""
        if len(sentiment_df) < 2:
            return 'insufficient_data'
        
        recent_avg = sentiment_df.iloc[-3:]['average_tone'].mean()
        older_avg = sentiment_df.iloc[:-3]['average_tone'].mean() if len(sentiment_df) > 3 else sentiment_df.iloc[0]['average_tone']
        
        diff = recent_avg - older_avg
        
        if diff > 0.5:
            return 'improving'
        elif diff < -0.5:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _build_events_section(self, data: Dict[str, Any], target_date: datetime) -> Dict[str, Any]:
        """Build recent events section."""
        events_data = data.get('ground_truth', pd.DataFrame())
        
        if events_data.empty:
            return {'status': 'No events data available', 'events': []}
        
        # Get events from last 7 days
        recent_events = events_data[
            events_data['timestamp'] >= (target_date - timedelta(days=7))
        ].sort_values('timestamp', ascending=False)
        
        events_list = []
        for _, event in recent_events.iterrows():
            events_list.append({
                'date': event['timestamp'].strftime('%Y-%m-%d'),
                'description': event['description'],
                'country': event['country'],
                'fatalities': event.get('fatalities', 0),
                'source': event['source'],
                'node_id': event.get('node_id', None)
            })
        
        return {
            'status': f'{len(events_list)} recent events',
            'events': events_list
        }
    
    def _build_news_section(self, data: Dict[str, Any], target_date: datetime) -> Dict[str, Any]:
        """Build news analysis section with full article text (matching paper format)."""
        news_data = data.get('news_articles', {})
        all_articles = news_data.get('google', []) + news_data.get('gdelt', [])

        # Filter articles before target date and sort most recent first
        dated_articles = []
        for article in all_articles:
            pub_date = self._extract_article_date(article)
            if pub_date and pub_date <= target_date:
                dated_articles.append((pub_date, article))
        dated_articles.sort(key=lambda x: x[0], reverse=True)

        # Build article entries with body text (paper format: up to 2000 chars each)
        # Cap total context at ~480,000 chars (~120K tokens) per the paper
        articles_text = []
        total_chars = 0
        max_chars = 480000

        for date, article in dated_articles:
            body = article.get('body', '') or ''
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown')
            entry = f"[{date.strftime('%Y-%m-%d %H:%M')}] {title} ({source})\n{body[:2000]}"
            if total_chars + len(entry) > max_chars:
                break
            articles_text.append(entry)
            total_chars += len(entry)

        omitted = len(dated_articles) - len(articles_text)

        return {
            'status': 'News data available',
            'analysis': {
                'article_count': len(articles_text),
                'omitted_count': omitted,
                'total_chars': total_chars,
                'key_topics': self._extract_key_topics(news_data),
            },
            'articles_text': articles_text,
        }
    
    def _extract_key_topics(self, news_data: Dict[str, List]) -> List[str]:
        """Extract key topics from news articles."""
        # Simple keyword frequency analysis
        keywords = ['Iran', 'Israel', 'missile', 'strike', 'nuclear', 'oil', 'conflict']
        topics = []
        
        all_articles = news_data.get('google', []) + news_data.get('gdelt', [])
        
        for keyword in keywords:
            count = 0
            for article in all_articles:
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                if keyword.lower() in title or keyword.lower() in description:
                    count += 1
            
            if count > len(all_articles) * 0.1:  # 10% threshold
                topics.append(f"{keyword} ({count} mentions)")
        
        return topics
    
    def _get_latest_headlines(self, news_data: Dict[str, List], target_date: datetime, 
                            limit: int = 5) -> List[Dict]:
        """Get most recent headlines before target date."""
        headlines = []
        
        all_articles = news_data.get('google', []) + news_data.get('gdelt', [])
        
        # Filter and sort by date
        dated_articles = []
        for article in all_articles:
            pub_date = self._extract_article_date(article)
            if pub_date and pub_date <= target_date:
                dated_articles.append((pub_date, article))
        
        # Sort by date (most recent first) and take top N
        dated_articles.sort(key=lambda x: x[0], reverse=True)
        
        for i, (date, article) in enumerate(dated_articles[:limit]):
            headlines.append({
                'title': article.get('title', 'No title'),
                'source': article.get('source', 'Unknown'),
                'date': date.strftime('%Y-%m-%d %H:%M'),
                'url': article.get('url', '')
            })
        
        return headlines
    
    def _build_rolling_windows(self, data: Dict[str, Any], target_date: datetime) -> Dict[str, Any]:
        """Build rolling window analysis (24h, 72h, 7d)."""
        windows = {}
        
        for window_hours in [24, 72, 168]:  # 24h, 72h, 7d
            window_start = target_date - timedelta(hours=window_hours)
            window_name = f'{window_hours}h'
            
            if window_hours == 168:
                window_name = '7d'
            
            windows[window_name] = {
                'period': f'{window_start.isoformat()} to {target_date.isoformat()}',
                'economic_changes': self._analyze_economic_window(data, window_start, target_date),
                'event_summary': self._analyze_events_window(data, window_start, target_date),
                'sentiment_summary': self._analyze_sentiment_window(data, window_start, target_date)
            }
        
        return windows
    
    def _analyze_economic_window(self, data: Dict[str, Any], start: datetime, end: datetime) -> Dict:
        """Analyze economic changes in time window."""
        econ_df = data.get('economic', pd.DataFrame())
        
        if econ_df.empty:
            return {'status': 'no_data'}
        
        window_data = econ_df[
            (econ_df['timestamp'] >= start) & (econ_df['timestamp'] <= end)
        ]
        
        changes = {}
        for symbol in ['brent_crude', 'tel_aviv_35']:
            symbol_data = window_data[window_data['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) >= 2:
                first_price = symbol_data.iloc[0]['close']
                last_price = symbol_data.iloc[-1]['close']
                change_pct = (last_price / first_price - 1) * 100
                changes[symbol] = round(change_pct, 2)
        
        return changes
    
    def _analyze_events_window(self, data: Dict[str, Any], start: datetime, end: datetime) -> Dict:
        """Analyze events in time window."""
        events_df = data.get('ground_truth', pd.DataFrame())
        
        if events_df.empty:
            return {'event_count': 0}
        
        window_events = events_df[
            (events_df['timestamp'] >= start) & (events_df['timestamp'] <= end)
        ]
        
        return {
            'event_count': len(window_events),
            'total_fatalities': window_events['fatalities'].sum() if 'fatalities' in window_events.columns else 0
        }
    
    def _analyze_sentiment_window(self, data: Dict[str, Any], start: datetime, end: datetime) -> Dict:
        """Analyze sentiment in time window."""
        sentiment_df = data.get('sentiment', pd.DataFrame())
        
        if sentiment_df.empty:
            return {'status': 'no_data'}
        
        window_sentiment = sentiment_df[
            (sentiment_df['timestamp'] >= start) & (sentiment_df['timestamp'] <= end)
        ]
        
        if window_sentiment.empty:
            return {'status': 'no_data'}
        
        return {
            'average_tone': window_sentiment['average_tone'].mean(),
            'total_events': window_sentiment['event_count'].sum()
        }
    
    def _build_metadata(self, data: Dict[str, Any], target_date: datetime) -> Dict[str, Any]:
        """Build metadata about the context."""
        metadata = {
            'temporal_cutoff': target_date.isoformat(),
            'data_completeness': {},
            'fog_of_war_compliance': True,
            'generation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Check data completeness
        for source in ['economic', 'osint', 'sentiment', 'ground_truth']:
            df = data.get(source, pd.DataFrame())
            metadata['data_completeness'][source] = {
                'available': not df.empty,
                'record_count': len(df),
                'latest_timestamp': df['timestamp'].max().isoformat() if not df.empty and 'timestamp' in df.columns else None
            }
        
        # Check news data
        news_data = data.get('news_articles', {})
        metadata['data_completeness']['news'] = {
            'google_articles': len(news_data.get('google', [])),
            'gdelt_articles': len(news_data.get('gdelt', []))
        }
        
        return metadata
    
    def format_as_markdown(self, briefing: Dict[str, Any]) -> str:
        """Format briefing as markdown intelligence report."""
        md = f"""# Intelligence Briefing
## Target Date: {briefing['target_date']}
### Temporal Constraints: ✅ FOG OF WAR COMPLIANT

---

## Executive Summary
**Threat Level:** {briefing['executive_summary']['threat_level']}  
**Confidence:** {briefing['executive_summary']['confidence']}

{briefing['executive_summary']['situation_overview']}

### Key Developments:
"""
        for dev in briefing['executive_summary']['key_developments']:
            md += f"- {dev}\n"
        
        md += f"""

---

## Economic Signals
**Status:** {briefing['economic_signals']['status']}

"""
        
        if briefing['economic_signals'].get('indicators'):
            for symbol, data in briefing['economic_signals']['indicators'].items():
                md += f"### {symbol.replace('_', ' ').title()}\n"
                md += f"- Current Price: ${data['current_price']:.2f}\n"
                
                if data.get('changes'):
                    for period, change in data['changes'].items():
                        md += f"- {period}: {change:+.2f}%\n"
                
                md += "\n"
        
        md += f"""---

## Tactical Intelligence
**Status:** {briefing['tactical_intelligence']['status']}

"""
        
        if briefing['tactical_intelligence'].get('metrics'):
            metrics = briefing['tactical_intelligence']['metrics']
            md += f"- Military Aircraft Count: {metrics.get('current_military_aircraft', 'N/A')}\n"
            md += f"- Weekly Trend: {metrics.get('weekly_trend', 'N/A')}\n"
            md += f"- Weekly Average: {metrics.get('average_weekly', 'N/A'):.1f}\n"
        
        md += f"""

---

## Recent Events
**Status:** {briefing['recent_events']['status']}

"""
        
        for event in briefing['recent_events'].get('events', [])[:5]:
            md += f"### {event['date']}\n"
            md += f"**{event['description']}**\n"
            md += f"- Location: {event['country']}\n"
            if event.get('fatalities'):
                md += f"- Fatalities: {event['fatalities']}\n"
            md += f"- Source: {event['source']}\n\n"
        
        md += """---

## Rolling Window Analysis
"""
        
        if 'rolling_analysis' in briefing:
            for window, analysis in briefing['rolling_analysis'].items():
                md += f"\n### {window} Window\n"
                md += f"Period: {analysis['period']}\n\n"
                
                if analysis['economic_changes']:
                    md += "**Economic Changes:**\n"
                    for symbol, change in analysis['economic_changes'].items():
                        md += f"- {symbol}: {change:+.2f}%\n"
                
                if analysis['event_summary']:
                    md += f"**Events:** {analysis['event_summary']['event_count']} incidents\n"
        
        md += f"""

---

## Metadata
- **Temporal Cutoff:** {briefing['metadata']['temporal_cutoff']}
- **FOG OF WAR Compliance:** ✅ {briefing['metadata']['fog_of_war_compliance']}
- **Generated:** {briefing['metadata']['generation_timestamp']}

### Data Sources:
"""
        
        for source, info in briefing['metadata']['data_completeness'].items():
            if isinstance(info, dict) and info.get('available'):
                md += f"- **{source}:** {info['record_count']} records\n"
        
        return md


if __name__ == "__main__":
    # Test the context builder
    import logging
    from datetime import datetime, timezone
    from data_fetcher import DataFetcher
    
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    fetcher = DataFetcher()
    test_date = datetime(2026, 3, 1, tzinfo=timezone.utc)
    test_data = fetcher.fetch_all_data(test_date - timedelta(days=5), test_date)
    
    # Build context
    builder = ContextBuilder(test_data)
    briefing = builder.build_context(test_date)
    
    # Print as markdown
    markdown = builder.format_as_markdown(briefing)
    print(markdown)