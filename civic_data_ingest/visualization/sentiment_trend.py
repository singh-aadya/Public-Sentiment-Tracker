"""
Sentiment Trend Visualizer using Plotly
Creates interactive charts showing sentiment trends over time
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from collections import defaultdict, Counter

class SentimentTrendVisualizer:
    def __init__(self):
        """Initialize sentiment trend visualizer"""
        self.color_palette = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#4682B4',   # Steel Blue
            'high': '#FF4500',      # Orange Red
            'medium': '#FFA500',    # Orange
            'low': '#32CD32'        # Lime Green
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_time_series_data(self, data: List[Dict]) -> pd.DataFrame:
        """
        Prepare data for time series analysis
        
        Args:
            data: List of processed civic data records
            
        Returns:
            DataFrame with time series data
        """
        records = []
        
        for record in data:
            # Extract timestamp
            timestamp = record.get('timestamp')
            if not timestamp:
                continue
            
            # Parse timestamp
            try:
                if isinstance(timestamp, str):
                    # Handle different timestamp formats
                    if 'T' in timestamp:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    else:
                        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                else:
                    dt = timestamp
            except:
                continue
            
            # Extract relevant fields
            records.append({
                'timestamp': dt,
                'date': dt.date(),
                'hour': dt.hour,
                'day_of_week': dt.weekday(),
                'sentiment': record.get('sentiment', 'neutral'),
                'urgency': record.get('urgency', 'low'),
                'vader_compound': record.get('vader_compound', 0),
                'sentiment_confidence': record.get('sentiment_confidence', 0),
                'source': record.get('source', 'unknown'),
                'location': record.get('extracted_locations', ['Unknown'])[0] if record.get('extracted_locations') else 'Unknown',
                'keywords': [kw['keyword'] if isinstance(kw, dict) else str(kw) 
                           for kw in record.get('keywords', [])]
            })
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('timestamp')
            df['cumulative_count'] = range(1, len(df) + 1)
        
        return df
    
    def create_sentiment_timeline(self, data: List[Dict], save_path: str = "sentiment_timeline.html") -> go.Figure:
        """
        Create sentiment timeline chart
        
        Args:
            data: List of processed civic data records
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure
        """
        self.logger.info("Creating sentiment timeline")
        
        df = self.prepare_time_series_data(data)
        
        if df.empty:
            self.logger.warning("No data available for sentiment timeline")
            return go.Figure()
        
        # Group by date and sentiment
        daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in daily_sentiment.columns:
                fig.add_trace(go.Scatter(
                    x=daily_sentiment.index,
                    y=daily_sentiment[sentiment],
                    mode='lines+markers',
                    name=sentiment.title(),
                    line=dict(color=self.color_palette[sentiment], width=2),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{sentiment.title()}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Count: %{y}<br>' +
                                 '<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Civic Issues Sentiment Timeline',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Date',
            yaxis_title='Number of Issues',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        # Save figure
        fig.write_html(save_path)
        self.logger.info(f"Sentiment timeline saved to {save_path}")
        
        return fig
    
    def create_urgency_distribution(self, data: List[Dict], save_path: str = "urgency_distribution.html") -> go.Figure:
        """
        Create urgency level distribution chart
        
        Args:
            data: List of processed civic data records
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure
        """
        self.logger.info("Creating urgency distribution chart")
        
        df = self.prepare_time_series_data(data)
        
        if df.empty:
            self.logger.warning("No data available for urgency distribution")
            return go.Figure()
        
        # Count urgency levels
        urgency_counts = df['urgency'].value_counts()
        
        # Create pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=urgency_counts.index,
                values=urgency_counts.values,
                hole=0.4,
                marker=dict(
                    colors=[self.color_palette.get(urgency, '#808080') for urgency in urgency_counts.index]
                ),
                textinfo='label+percent',
                textfont=dict(size=12),
                hovertemplate='<b>%{label}</b><br>' +
                             'Count: %{value}<br>' +
                             'Percentage: %{percent}<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Civic Issues Urgency Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            template='plotly_white',
            height=500,
            annotations=[dict(text='Urgency<br>Levels', x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        
        # Save figure
        fig.write_html(save_path)
        self.logger.info(f"Urgency distribution saved to {save_path}")
        
        return fig
    
    def create_hourly_pattern(self, data: List[Dict], save_path: str = "hourly_pattern.html") -> go.Figure:
        """
        Create hourly pattern chart
        
        Args:
            data: List of processed civic data records
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure
        """
        self.logger.info("Creating hourly pattern chart")
        
        df = self.prepare_time_series_data(data)
        
        if df.empty:
            self.logger.warning("No data available for hourly pattern")
            return go.Figure()
        
        # Group by hour and sentiment
        hourly_data = df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in hourly_data.columns:
                fig.add_trace(go.Bar(
                    name=sentiment.title(),
                    x=hourly_data.index,
                    y=hourly_data[sentiment],
                    marker_color=self.color_palette[sentiment],
                    hovertemplate=f'<b>{sentiment.title()}</b><br>' +
                                 'Hour: %{x}:00<br>' +
                                 'Count: %{y}<br>' +
                                 '<extra></extra>'
                ))
        
        fig.update_layout(
            title={
                'text': 'Civic Issues by Hour of Day',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Hour of Day',
            yaxis_title='Number of Issues',
            barmode='stack',
            template='plotly_white',
            height=500,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(0, 24, 2)),
                ticktext=[f"{i}:00" for i in range(0, 24, 2)]
            )
        )
        
        # Save figure
        fig.write_html(save_path)
        self.logger.info(f"Hourly pattern saved to {save_path}")
        
        return fig
    
    def create_location_sentiment_heatmap(self, data: List[Dict], save_path: str = "location_sentiment_heatmap.html") -> go.Figure:
        """
        Create location vs sentiment heatmap
        
        Args:
            data: List of processed civic data records
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure
        """
        self.logger.info("Creating location sentiment heatmap")
        
        df = self.prepare_time_series_data(data)
        
        if df.empty:
            self.logger.warning("No data available for location sentiment heatmap")
            return go.Figure()
        
        # Create pivot table
        location_sentiment = df.groupby(['location', 'sentiment']).size().unstack(fill_value=0)
        
        # Calculate percentages
        location_sentiment_pct = location_sentiment.div(location_sentiment.sum(axis=1), axis=0) * 100
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=location_sentiment_pct.values,
            x=location_sentiment_pct.columns,
            y=location_sentiment_pct.index,
            colorscale='RdYlGn_r',
            hovertemplate='Location: %{y}<br>' +
                         'Sentiment: %{x}<br>' +
                         'Percentage: %{z:.1f}%<br>' +
                         '<extra></extra>',
            colorbar=dict(title="Percentage")
        ))
        
        fig.update_layout(
            title={
                'text': 'Sentiment Distribution by Location',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Sentiment',
            yaxis_title='Location',
            template='plotly_white',
            height=max(400, len(location_sentiment_pct.index) * 30)
        )
        
        # Save figure
        fig.write_html(save_path)
        self.logger.info(f"Location sentiment heatmap saved to {save_path}")
        
        return fig
    
    def create_compound_score_distribution(self, data: List[Dict], save_path: str = "compound_score_distribution.html") -> go.Figure:
        """
        Create VADER compound score distribution
        
        Args:
            data: List of processed civic data records
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure
        """
        self.logger.info("Creating compound score distribution")
        
        df = self.prepare_time_series_data(data)
        
        if df.empty:
            self.logger.warning("No data available for compound score distribution")
            return go.Figure()
        
        # Create histogram
        fig = go.Figure(data=[
            go.Histogram(
                x=df['vader_compound'],
                nbinsx=30,
                marker_color='lightblue',
                opacity=0.7,
                hovertemplate='Score Range: %{x}<br>' +
                             'Count: %{y}<br>' +
                             '<extra></extra>'
            )
        ])
        
        # Add vertical lines for sentiment boundaries
        fig.add_vline(x=-0.05, line_dash="dash", line_color="red", 
                     annotation_text="Negative Threshold")
        fig.add_vline(x=0.05, line_dash="dash", line_color="green", 
                     annotation_text="Positive Threshold")
        
        fig.update_layout(
            title={
                'text': 'VADER Compound Score Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='VADER Compound Score',
            yaxis_title='Frequency',
            template='plotly_white',
            height=500
        )
        
        # Save figure
        fig.write_html(save_path)
        self.logger.info(f"Compound score distribution saved to {save_path}")
        
        return fig
    
    def create_comprehensive_dashboard(self, data: List[Dict], save_path: str = "sentiment_dashboard.html") -> go.Figure:
        """
        Create comprehensive sentiment analysis dashboard
        
        Args:
            data: List of processed civic data records
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure with subplots
        """
        self.logger.info("Creating comprehensive sentiment dashboard")
        
        df = self.prepare_time_series_data(data)
        
        if df.empty:
            self.logger.warning("No data available for dashboard")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Timeline', 'Urgency Distribution', 
                          'Hourly Pattern', 'Location Sentiment'),
            specs=[[{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # 1. Sentiment Timeline
        daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in daily_sentiment.columns:
                fig.add_trace(
                    go.Scatter(
                        x=daily_sentiment.index,
                        y=daily_sentiment[sentiment],
                        mode='lines+markers',
                        name=sentiment.title(),
                        line=dict(color=self.color_palette[sentiment]),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # 2. Urgency Distribution
        urgency_counts = df['urgency'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=urgency_counts.index,
                values=urgency_counts.values,
                marker=dict(colors=[self.color_palette.get(u, '#808080') for u in urgency_counts.index]),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Hourly Pattern
        hourly_total = df.groupby('hour').size()
        fig.add_trace(
            go.Bar(
                x=hourly_total.index,
                y=hourly_total.values,
                marker_color='lightblue',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Location Sentiment Heatmap
        location_sentiment = df.groupby(['location', 'sentiment']).size().unstack(fill_value=0)
        if not location_sentiment.empty:
            location_sentiment_pct = location_sentiment.div(location_sentiment.sum(axis=1), axis=0) * 100
            
            fig.add_trace(
                go.Heatmap(
                    z=location_sentiment_pct.values,
                    x=location_sentiment_pct.columns,
                    y=location_sentiment_pct.index,
                    colorscale='RdYlGn_r',
                    showscale=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Civic Issues Sentiment Analysis Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            template='plotly_white',
            height=800,
            showlegend=True
        )
        
        # Save figure
        fig.write_html(save_path)
        self.logger.info(f"Comprehensive dashboard saved to {save_path}")
        
        return fig

def main():
    """Test the sentiment trend visualizer"""
    # Create sample data with timestamps
    base_date = datetime.now() - timedelta(days=7)
    sample_data = []
    
    for i in range(20):
        timestamp = base_date + timedelta(hours=i*6)
        sample_data.append({
            'timestamp': timestamp.isoformat(),
            'sentiment': np.random.choice(['positive', 'negative', 'neutral'], p=[0.3, 0.5, 0.2]),
            'urgency': np.random.choice(['high', 'medium', 'low'], p=[0.2, 0.3, 0.5]),
            'vader_compound': np.random.uniform(-1, 1),
            'sentiment_confidence': np.random.uniform(0.5, 1.0),
            'source': np.random.choice(['twitter', 'news']),
            'extracted_locations': [np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai'])],
            'keywords': [{'keyword': 'test'}]
        })
    
    print("Testing Sentiment Trend Visualizer:")
    print("=" * 45)
    
    # Initialize visualizer
    visualizer = SentimentTrendVisualizer()
    
    # Create various charts
    print("Creating sentiment timeline...")
    timeline_fig = visualizer.create_sentiment_timeline(sample_data, "test_sentiment_timeline.html")
    print("✓ Sentiment timeline created")
    
    print("Creating urgency distribution...")
    urgency_fig = visualizer.create_urgency_distribution(sample_data, "test_urgency_distribution.html")
    print("✓ Urgency distribution created")
    
    print("Creating hourly pattern...")
    hourly_fig = visualizer.create_hourly_pattern(sample_data, "test_hourly_pattern.html")
    print("✓ Hourly pattern created")
    
    print("Creating location sentiment heatmap...")
    heatmap_fig = visualizer.create_location_sentiment_heatmap(sample_data, "test_location_heatmap.html")
    print("✓ Location sentiment heatmap created")
    
    print("Creating compound score distribution...")
    compound_fig = visualizer.create_compound_score_distribution(sample_data, "test_compound_distribution.html")
    print("✓ Compound score distribution created")
    
    print("Creating comprehensive dashboard...")
    dashboard_fig = visualizer.create_comprehensive_dashboard(sample_data, "test_sentiment_dashboard.html")
    print("✓ Comprehensive dashboard created")
    
    print(f"\n📊 All visualizations created successfully!")
    print("Open the HTML files in your browser to view the interactive charts.")

if __name__ == "__main__":
    main()