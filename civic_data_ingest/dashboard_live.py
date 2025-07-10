"""
Live Streamlit Dashboard with API Integration
Real-time civic data visualization connected to FastAPI backend
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Live Civic Issues Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #00ff00;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 10px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .status-healthy { background-color: #d4edda; color: #155724; }
    .status-error { background-color: #f8d7da; color: #721c24; }
    .status-processing { background-color: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_api_health():
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json(), True
        else:
            return {"status": "error", "message": "API not responding"}, False
    except requests.exceptions.RequestException:
        return {"status": "error", "message": "Cannot connect to API"}, False

@st.cache_data(ttl=10)  # Cache for 10 seconds for live data
def get_live_data(limit=100):
    """Fetch live data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/live?limit={limit}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"data": [], "total_records": 0}
    except requests.exceptions.RequestException:
        return {"data": [], "total_records": 0}

@st.cache_data(ttl=15)  # Cache for 15 seconds
def get_live_statistics():
    """Fetch live statistics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/stats")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except requests.exceptions.RequestException:
        return {}

def get_recent_data(minutes=60):
    """Fetch recent data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/recent/{minutes}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"data": [], "records_found": 0}
    except requests.exceptions.RequestException:
        return {"data": [], "records_found": 0}

def simulate_data(count=10, interval=5):
    """Trigger data simulation"""
    try:
        payload = {"count": count, "interval_seconds": interval}
        response = requests.post(f"{API_BASE_URL}/simulate-data", json=payload)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def process_text_via_api(text, source="manual"):
    """Process text through API"""
    try:
        payload = {"text": text, "source": source}
        response = requests.post(f"{API_BASE_URL}/process-text", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def create_live_metrics(stats):
    """Create live metrics display"""
    if not stats:
        st.warning("No statistics available")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = stats.get('total_records', 0)
        recent = stats.get('recent_records_24h', 0)
        st.metric(
            label="Total Issues",
            value=total,
            delta=f"+{recent} in 24h"
        )
    
    with col2:
        sentiment_dist = stats.get('sentiment_distribution', {})
        negative_count = sentiment_dist.get('negative', 0)
        negative_pct = (negative_count / total * 100) if total > 0 else 0
        st.metric(
            label="Negative Issues",
            value=negative_count,
            delta=f"{negative_pct:.1f}% of total"
        )
    
    with col3:
        urgency_dist = stats.get('urgency_distribution', {})
        high_urgency = urgency_dist.get('high', 0)
        st.metric(
            label="High Urgency",
            value=high_urgency,
            delta=f"{(high_urgency/total*100):.1f}% of total" if total > 0 else "0%"
        )
    
    with col4:
        avg_sentiment = stats.get('average_sentiment_score', 0)
        sentiment_trend = "📈" if avg_sentiment > 0 else "📉" if avg_sentiment < 0 else "➡️"
        st.metric(
            label="Avg Sentiment Score",
            value=f"{avg_sentiment:.3f}",
            delta=f"{sentiment_trend} Sentiment trend"
        )

def create_live_charts(data, stats):
    """Create live data visualizations"""
    if not data:
        st.warning("No data available for visualization")
        return
    
    df = pd.DataFrame(data)
    
    # Real-time sentiment timeline
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Real-time Sentiment Timeline")
        
        # Convert timestamps and create timeline
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('datetime')
        
        # Create cumulative sentiment count
        sentiment_timeline = []
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_data = df_sorted[df_sorted['sentiment'] == sentiment]
            if not sentiment_data.empty:
                sentiment_timeline.append(
                    go.Scatter(
                        x=sentiment_data['datetime'],
                        y=range(1, len(sentiment_data) + 1),
                        mode='lines+markers',
                        name=sentiment.title(),
                        line=dict(width=2)
                    )
                )
        
        if sentiment_timeline:
            fig_timeline = go.Figure(sentiment_timeline)
            fig_timeline.update_layout(
                title="Cumulative Sentiment Over Time",
                xaxis_title="Time",
                yaxis_title="Cumulative Count",
                height=400
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Current Distribution")
        
        # Sentiment pie chart
        sentiment_dist = stats.get('sentiment_distribution', {})
        if sentiment_dist:
            fig_pie = px.pie(
                values=list(sentiment_dist.values()),
                names=list(sentiment_dist.keys()),
                title="Current Sentiment Distribution",
                color_discrete_map={
                    'positive': '#2E8B57',
                    'negative': '#DC143C', 
                    'neutral': '#4682B4'
                }
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

def create_location_analysis(data, stats):
    """Create location-based analysis"""
    if not data:
        return
    
    st.subheader("🗺️ Geographic Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top locations bar chart
        top_locations = stats.get('top_locations', {})
        if top_locations:
            locations = list(top_locations.keys())[:10]
            counts = list(top_locations.values())[:10]
            
            fig_locations = px.bar(
                x=counts,
                y=locations,
                orientation='h',
                title="Issues by Location",
                labels={'x': 'Number of Issues', 'y': 'Location'}
            )
            fig_locations.update_layout(height=400)
            st.plotly_chart(fig_locations, use_container_width=True)
    
    with col2:
        # Urgency distribution by location
        if data:
            df = pd.DataFrame(data)
            
            # Create location-urgency matrix
            location_urgency = []
            for record in data:
                locations = record.get('extracted_locations', ['Unknown'])
                location = locations[0] if locations else 'Unknown'
                urgency = record.get('urgency', 'low')
                location_urgency.append({'Location': location, 'Urgency': urgency})
            
            if location_urgency:
                df_urgency = pd.DataFrame(location_urgency)
                urgency_counts = df_urgency.groupby(['Location', 'Urgency']).size().unstack(fill_value=0)
                
                # Top 8 locations
                top_8_locations = urgency_counts.sum(axis=1).nlargest(8).index
                urgency_counts_top = urgency_counts.loc[top_8_locations]
                
                fig_urgency = px.bar(
                    urgency_counts_top,
                    title="Urgency Levels by Location",
                    color_discrete_map={
                        'high': '#FF4500',
                        'medium': '#FFA500', 
                        'low': '#32CD32'
                    }
                )
                fig_urgency.update_layout(height=400)
                st.plotly_chart(fig_urgency, use_container_width=True)

def create_live_data_table(data):
    """Create live data table with recent records"""
    if not data:
        st.warning("No data available")
        return
    
    st.subheader("📋 Live Data Stream")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Format for display
    display_df = df[[
        'timestamp', 'original_text', 'source', 'extracted_locations',
        'sentiment', 'urgency', 'sentiment_confidence'
    ]].copy()
    
    # Format timestamp
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
    
    # Format locations
    display_df['location'] = display_df['extracted_locations'].apply(
        lambda x: x[0] if x and len(x) > 0 else 'Unknown'
    )
    
    # Reorder columns
    display_df = display_df[[
        'timestamp', 'original_text', 'source', 'location', 
        'sentiment', 'urgency', 'sentiment_confidence'
    ]]
    
    # Rename columns
    display_df.columns = [
        'Time', 'Text', 'Source', 'Location', 
        'Sentiment', 'Urgency', 'Confidence'
    ]
    
    # Show latest 20 records
    st.dataframe(
        display_df.head(20),
        use_container_width=True,
        height=400
    )
    
    st.info(f"Showing latest 20 of {len(data)} total records")

def main():
    """Main dashboard application"""
    
    # Header with live indicator
    st.markdown(
        '<h1 class="main-header">'
        '<span class="live-indicator"></span>'
        '🏛️ Live Civic Issues Dashboard'
        '</h1>', 
        unsafe_allow_html=True
    )
    
    # Check API health
    health_data, is_healthy = get_api_health()
    
    # Status indicator
    if is_healthy:
        st.markdown(
            '<div class="status-box status-healthy">'
            '✅ API Connected - Live data streaming'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-box status-error">'
            '❌ API Connection Failed - Please start the API server'
            '</div>',
            unsafe_allow_html=True
        )
        st.info("To start the API server, run: `python api/main.py` or `uvicorn api.main:app --reload`")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Live Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    
    if auto_refresh:
        # Auto-refresh every 30 seconds
        time.sleep(0.1)  # Small delay to prevent too frequent updates
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Data simulation controls
    st.sidebar.subheader("📊 Data Simulation")
    
    sim_count = st.sidebar.number_input("Records to generate", min_value=1, max_value=50, value=10)
    sim_interval = st.sidebar.number_input("Interval (seconds)", min_value=1, max_value=60, value=5)
    
    if st.sidebar.button("🚀 Start Simulation"):
        if simulate_data(sim_count, sim_interval):
            st.sidebar.success(f"Started generating {sim_count} records every {sim_interval}s")
        else:
            st.sidebar.error("Failed to start simulation")
    
    # Manual text input
    st.sidebar.subheader("✍️ Add Manual Entry")
    
    manual_text = st.sidebar.text_area("Enter civic issue text:")
    
    if st.sidebar.button("📝 Process Text") and manual_text:
        result = process_text_via_api(manual_text, "manual")
        if result:
            st.sidebar.success("Text processed successfully!")
            st.cache_data.clear()  # Clear cache to show new data
        else:
            st.sidebar.error("Failed to process text")
    
    # Get live data and statistics
    live_data_response = get_live_data(limit=100)
    stats = get_live_statistics()
    
    live_data = live_data_response.get('data', [])
    total_records = live_data_response.get('total_records', 0)
    
    # Main content
    st.markdown("---")
    
    # Overview metrics
    st.header("📊 Live Overview")
    create_live_metrics(stats)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Real-time Analytics",
        "🗺️ Geographic Analysis", 
        "📋 Live Data Stream",
        "⏱️ Recent Activity",
        "📊 System Status"
    ])
    
    with tab1:
        st.header("Real-time Analytics")
        create_live_charts(live_data, stats)
    
    with tab2:
        st.header("Geographic Distribution")
        create_location_analysis(live_data, stats)
    
    with tab3:
        st.header("Live Data Stream")
        create_live_data_table(live_data)
    
    with tab4:
        st.header("Recent Activity")
        
        # Time filter
        time_filter = st.selectbox(
            "Show data from last:",
            [15, 30, 60, 120, 360],
            format_func=lambda x: f"{x} minutes"
        )
        
        recent_data_response = get_recent_data(time_filter)
        recent_data = recent_data_response.get('data', [])
        recent_count = recent_data_response.get('records_found', 0)
        
        st.info(f"Found {recent_count} records in the last {time_filter} minutes")
        
        if recent_data:
            # Recent activity summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                recent_sentiments = [r.get('sentiment', 'neutral') for r in recent_data]
                sentiment_counts = Counter(recent_sentiments)
                
                fig_recent_sentiment = px.pie(
                    values=list(sentiment_counts.values()),
                    names=list(sentiment_counts.keys()),
                    title=f"Sentiment (Last {time_filter}min)"
                )
                st.plotly_chart(fig_recent_sentiment, use_container_width=True)
            
            with col2:
                recent_urgencies = [r.get('urgency', 'low') for r in recent_data]
                urgency_counts = Counter(recent_urgencies)
                
                fig_recent_urgency = px.bar(
                    x=list(urgency_counts.keys()),
                    y=list(urgency_counts.values()),
                    title=f"Urgency (Last {time_filter}min)"
                )
                st.plotly_chart(fig_recent_urgency, use_container_width=True)
            
            with col3:
                recent_locations = []
                for r in recent_data:
                    locs = r.get('extracted_locations', ['Unknown'])
                    if locs:
                        recent_locations.append(locs[0])
                
                location_counts = Counter(recent_locations)
                top_recent_locations = dict(location_counts.most_common(5))
                
                if top_recent_locations:
                    fig_recent_locations = px.bar(
                        x=list(top_recent_locations.values()),
                        y=list(top_recent_locations.keys()),
                        orientation='h',
                        title=f"Top Locations (Last {time_filter}min)"
                    )
                    st.plotly_chart(fig_recent_locations, use_container_width=True)
            
            # Recent data table
            create_live_data_table(recent_data)
    
    with tab5:
        st.header("System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("API Health")
            if is_healthy:
                st.success("✅ API is running normally")
                st.json(health_data)
            else:
                st.error("❌ API connection issues")
                st.json(health_data)
        
        with col2:
            st.subheader("Data Summary")
            if stats:
                st.metric("Total Records", stats.get('total_records', 0))
                st.metric("Records (24h)", stats.get('recent_records_24h', 0))
                st.metric("Average Sentiment", f"{stats.get('average_sentiment_score', 0):.3f}")
                
                source_dist = stats.get('source_distribution', {})
                if source_dist:
                    st.write("**Source Distribution:**")
                    for source, count in source_dist.items():
                        st.write(f"- {source}: {count}")
    
    # Footer with last update time
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Total records: {total_records} | "
        f"API Status: {'✅ Connected' if is_healthy else '❌ Disconnected'}"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()