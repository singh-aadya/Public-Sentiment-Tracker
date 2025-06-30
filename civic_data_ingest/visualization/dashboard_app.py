"""
Streamlit Dashboard for Civic Data Ingest System
Interactive web interface for exploring civic issues data
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.map_visualizer import CivicMapVisualizer
from visualization.sentiment_trend import SentimentTrendVisualizer
from shared.utils import load_json
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="CivicLense: Public Sentiment Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #fafafa;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    # Create realistic sample data
    base_date = datetime.now() - timedelta(days=30)
    
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Pune', 'Hyderabad']
    categories = ['water', 'power', 'roads', 'garbage', 'transport']
    sentiments = ['positive', 'negative', 'neutral']
    urgencies = ['high', 'medium', 'low']
    sources = ['twitter', 'news']
    
    sample_data = []
    for i in range(100):
        timestamp = base_date + timedelta(hours=np.random.randint(0, 720))
        
        record = {
            'timestamp': timestamp.isoformat(),
            'original_text': f'Sample civic issue {i+1}',
            'cleaned_text': f'Sample civic issue {i+1}',
            'source': np.random.choice(sources),
            'extracted_locations': [np.random.choice(cities)],
            'sentiment': np.random.choice(sentiments, p=[0.2, 0.6, 0.2]),
            'urgency': np.random.choice(urgencies, p=[0.3, 0.4, 0.3]),
            'vader_compound': np.random.uniform(-1, 1),
            'sentiment_confidence': np.random.uniform(0.5, 1.0),
            'keywords': [
                {'keyword': np.random.choice(categories)},
                {'keyword': 'civic'},
                {'keyword': 'municipal'}
            ]
        }
        sample_data.append(record)
    
    return sample_data

@st.cache_data
def load_data_from_file(file_path):
    """Load data from uploaded file"""
    try:
        if file_path.endswith('.json'):
            return load_json(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path).to_dict('records')
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return []

def create_metrics_overview(data):
    """Create overview metrics"""
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Issues",
            value=len(data),
            delta=f"+{len(data)} issues"
        )
    
    with col2:
        negative_count = len([d for d in data if d.get('sentiment') == 'negative'])
        negative_pct = (negative_count / len(data)) * 100 if data else 0
        st.metric(
            label="Negative Issues",
            value=negative_count,
            delta=f"{negative_pct:.1f}% of total"
        )
    
    with col3:
        high_urgency = len([d for d in data if d.get('urgency') == 'high'])
        st.metric(
            label="High Urgency",
            value=high_urgency,
            delta=f"{(high_urgency/len(data)*100):.1f}% of total" if data else "0%"
        )
    
    with col4:
        locations = set()
        for d in data:
            locs = d.get('extracted_locations', [])
            if locs:
                locations.add(locs[0])
        st.metric(
            label="Affected Locations",
            value=len(locations),
            delta=f"{len(locations)} cities"
        )

def create_sentiment_charts(data):
    """Create sentiment analysis charts"""
    if not data:
        st.warning("No data available for sentiment analysis")
        return
    
    # Sentiment distribution pie chart
    sentiments = [d.get('sentiment', 'neutral') for d in data]
    sentiment_counts = Counter(sentiments)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=list(sentiment_counts.values()),
            names=list(sentiment_counts.keys()),
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C',
                'neutral': '#4682B4'
            }
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Urgency distribution
        urgencies = [d.get('urgency', 'low') for d in data]
        urgency_counts = Counter(urgencies)
        
        fig_urgency = px.bar(
            x=list(urgency_counts.keys()),
            y=list(urgency_counts.values()),
            title="Urgency Level Distribution",
            color=list(urgency_counts.keys()),
            color_discrete_map={
                'high': '#FF4500',
                'medium': '#FFA500',
                'low': '#32CD32'
            }
        )
        fig_urgency.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_urgency, use_container_width=True)

def create_location_analysis(data):
    """Create location-based analysis"""
    if not data:
        st.warning("No data available for location analysis")
        return
    
    # Extract location data
    location_data = {}
    for d in data:
        locations = d.get('extracted_locations', ['Unknown'])
        location = locations[0] if locations else 'Unknown'
        
        if location not in location_data:
            location_data[location] = {
                'count': 0,
                'sentiments': [],
                'urgencies': []
            }
        
        location_data[location]['count'] += 1
        location_data[location]['sentiments'].append(d.get('sentiment', 'neutral'))
        location_data[location]['urgencies'].append(d.get('urgency', 'low'))
    
    # Location-wise issue count
    locations = list(location_data.keys())
    counts = [location_data[loc]['count'] for loc in locations]
    
    fig_locations = px.bar(
        x=locations,
        y=counts,
        title="Issues by Location",
        labels={'x': 'Location', 'y': 'Number of Issues'}
    )
    fig_locations.update_layout(height=400)
    st.plotly_chart(fig_locations, use_container_width=True)
    
    # Location sentiment heatmap
    if len(locations) > 1:
        heatmap_data = []
        sentiments = ['positive', 'negative', 'neutral']
        
        for location in locations:
            sentiment_counts = Counter(location_data[location]['sentiments'])
            total = sum(sentiment_counts.values())
            
            for sentiment in sentiments:
                percentage = (sentiment_counts.get(sentiment, 0) / total) * 100 if total > 0 else 0
                heatmap_data.append({
                    'Location': location,
                    'Sentiment': sentiment,
                    'Percentage': percentage
                })
        
        df_heatmap = pd.DataFrame(heatmap_data)
        
        if not df_heatmap.empty:
            fig_heatmap = px.density_heatmap(
                df_heatmap,
                x='Sentiment',
                y='Location',
                z='Percentage',
                title="Sentiment Distribution by Location (%)",
                color_continuous_scale='RdYlGn_r'
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)

def create_keyword_analysis(data):
    """Create keyword analysis"""
    if not data:
        st.warning("No data available for keyword analysis")
        return
    
    # Extract all keywords
    all_keywords = []
    for d in data:
        keywords = d.get('keywords', [])
        for kw in keywords:
            if isinstance(kw, dict):
                all_keywords.append(kw.get('keyword', ''))
            else:
                all_keywords.append(str(kw))
    
    # Count keywords
    keyword_counts = Counter(all_keywords)
    top_keywords = dict(keyword_counts.most_common(15))
    
    if top_keywords:
        fig_keywords = px.bar(
            x=list(top_keywords.values()),
            y=list(top_keywords.keys()),
            orientation='h',
            title="Top Keywords",
            labels={'x': 'Frequency', 'y': 'Keywords'}
        )
        fig_keywords.update_layout(height=500)
        st.plotly_chart(fig_keywords, use_container_width=True)
    else:
        st.info("No keywords found in the data")

def create_time_analysis(data):
    """Create time-based analysis"""
    if not data:
        st.warning("No data available for time analysis")
        return
    
    # Parse timestamps and create DataFrame
    time_data = []
    for d in data:
        timestamp = d.get('timestamp')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                
                time_data.append({
                    'datetime': dt,
                    'date': dt.date(),
                    'hour': dt.hour,
                    'day_of_week': dt.strftime('%A'),
                    'sentiment': d.get('sentiment', 'neutral')
                })
            except:
                continue
    
    if not time_data:
        st.info("No valid timestamp data found")
        return
    
    df_time = pd.DataFrame(time_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily trend
        daily_counts = df_time.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        fig_daily = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Daily Issue Trend",
            markers=True
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        # Hourly pattern
        hourly_counts = df_time.groupby('hour').size().reset_index(name='count')
        
        fig_hourly = px.bar(
            hourly_counts,
            x='hour',
            y='count',
            title="Issues by Hour of Day"
        )
        fig_hourly.update_layout(height=400)
        st.plotly_chart(fig_hourly, use_container_width=True)

def create_data_table(data):
    """Create searchable data table"""
    if not data:
        st.warning("No data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Select relevant columns for display
    display_columns = []
    if 'timestamp' in df.columns:
        display_columns.append('timestamp')
    if 'original_text' in df.columns:
        display_columns.append('original_text')
    if 'source' in df.columns:
        display_columns.append('source')
    if 'extracted_locations' in df.columns:
        df['location'] = df['extracted_locations'].apply(lambda x: x[0] if x else 'Unknown')
        display_columns.append('location')
    if 'sentiment' in df.columns:
        display_columns.append('sentiment')
    if 'urgency' in df.columns:
        display_columns.append('urgency')
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'sentiment' in df.columns:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                options=['All'] + list(df['sentiment'].unique())
            )
    
    with col2:
        if 'urgency' in df.columns:
            urgency_filter = st.selectbox(
                "Filter by Urgency", 
                options=['All'] + list(df['urgency'].unique())
            )
    
    with col3:
        if 'location' in df.columns:
            location_filter = st.selectbox(
                "Filter by Location",
                options=['All'] + list(df['location'].unique())
            )
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'sentiment_filter' in locals() and sentiment_filter != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
    
    if 'urgency_filter' in locals() and urgency_filter != 'All':
        filtered_df = filtered_df[filtered_df['urgency'] == urgency_filter]
    
    if 'location_filter' in locals() and location_filter != 'All':
        filtered_df = filtered_df[filtered_df['location'] == location_filter]
    
    # Display table
    st.dataframe(
        filtered_df[display_columns] if display_columns else filtered_df,
        use_container_width=True,
        height=400
    )
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} records")

def main():
    """Main dashboard application"""
    # Header
    st.markdown('<h1 class="main-header">🏛️ Civic Issues Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for data loading
    st.sidebar.header("Data Source")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Sample Data", "Upload File", "Load from Output"]
    )
    
    data = []
    
    if data_source == "Sample Data":
        st.sidebar.info("Using generated sample data for demonstration")
        data = load_sample_data()
    
    elif data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=['json', 'csv'],
            help="Upload processed civic data in JSON or CSV format"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            data = load_data_from_file(temp_path)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
    
    elif data_source == "Load from Output":
        output_dir = "output"
        if os.path.exists(output_dir):
            json_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and 'results' in f]
            
            if json_files:
                selected_file = st.sidebar.selectbox(
                    "Select results file:",
                    json_files
                )
                
                if selected_file:
                    file_path = os.path.join(output_dir, selected_file)
                    data = load_data_from_file(file_path)
            else:
                st.sidebar.warning("No result files found in output directory")
        else:
            st.sidebar.warning("Output directory not found")
    
    if not data:
        st.warning("No data loaded. Please select a data source from the sidebar.")
        return
    
    # Main content
    st.markdown("---")
    
    # Overview metrics
    st.header("📊 Overview")
    create_metrics_overview(data)
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Sentiment Analysis", 
        "🗺️ Location Analysis", 
        "🔑 Keyword Analysis",
        "⏰ Time Analysis",
        "📋 Data Table",
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("Sentiment Analysis")
        create_sentiment_charts(data)
    
    with tab2:
        st.header("Location Analysis")
        create_location_analysis(data)
    
    with tab3:
        st.header("Keyword Analysis")
        create_keyword_analysis(data)
    
    with tab4:
        st.header("Time-based Analysis")
        create_time_analysis(data)
    
    with tab5:
        st.header("Data Explorer")
        create_data_table(data)
    
    with tab6:
        st.header("About This Dashboard")
        st.markdown("""
        ### Civic Data Ingest System Dashboard
        
        This dashboard provides an interactive interface for exploring civic issues data collected from social media and news sources.
        
        **Features:**
        - **Real-time sentiment analysis** using VADER sentiment analyzer
        - **Geographic visualization** of issues by location
        - **Keyword extraction** to identify common themes
        - **Time-based analysis** to track trends
        - **Interactive filtering** and data exploration
        
        **Data Sources:**
        - Twitter API for social media posts
        - News scraping from local news websites
        - Automatic location extraction and geocoding
        
        **Analysis Components:**
        - Text preprocessing and cleaning
        - TF-IDF vectorization for feature extraction
        - Civic-specific sentiment scoring
        - Urgency classification based on sentiment and keywords
        
        **Visualization Types:**
        - Interactive maps showing issue density
        - Time series charts for trend analysis
        - Distribution charts for sentiment and urgency
        - Heatmaps for location-sentiment correlation
        
        ---
        
        **Instructions:**
        1. Select a data source from the sidebar
        2. Explore different tabs for various analyses
        3. Use filters in the Data Table tab to drill down
        4. Hover over charts for detailed information
        
        **Note:** Sample data is used for demonstration when no real data is available.
        """)

if __name__ == "__main__":
    main()