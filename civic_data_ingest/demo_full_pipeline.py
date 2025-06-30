"""
Full Pipeline Demo for Civic Data Ingest System
Demonstrates the complete end-to-end workflow with visualizations
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.cleaner import TextCleaner
from preprocessing.vectorizer import TextVectorizer
from nlp_modeling.keyword_extractor import KeywordExtractor
from nlp_modeling.sentiment_analyzer import SentimentAnalyzer
from data_ingestion.geo_extractor import GeoExtractor
from visualization.map_visualizer import CivicMapVisualizer
from visualization.sentiment_trend import SentimentTrendVisualizer
from shared.utils import save_json, create_timestamp

def create_realistic_sample_data():
    """Create realistic sample civic data for demonstration"""
    
    # Real civic issue templates
    issue_templates = [
        {
            'text': 'Water supply has been disrupted in {location} for the past {days} days. Residents are facing severe shortage. @{authority} please take immediate action! #WaterCrisis #{location_tag}',
            'sentiment_bias': 'negative',
            'urgency_bias': 'high',
            'category': 'water'
        },
        {
            'text': 'Terrible road conditions in {location}. Potholes everywhere causing traffic problems and vehicle damage. The municipal corporation needs to fix this URGENTLY!!! #RoadRepair #{location_tag}',
            'sentiment_bias': 'negative', 
            'urgency_bias': 'high',
            'category': 'roads'
        },
        {
            'text': 'Power outage in {location} for {hours} hours. This is completely unacceptable. When will the electricity board take action? #PowerOutage',
            'sentiment_bias': 'negative',
            'urgency_bias': 'medium',
            'category': 'power'
        },
        {
            'text': 'Garbage collection has improved significantly in {location} over the past month. Thanks to the efficient work by the sanitation department. #ImprovedServices',
            'sentiment_bias': 'positive',
            'urgency_bias': 'low', 
            'category': 'sanitation'
        },
        {
            'text': 'The municipal corporation in {location} announced new measures to address the water supply issues. The initiative includes installation of new pipelines and improvement of existing infrastructure.',
            'sentiment_bias': 'positive',
            'urgency_bias': 'low',
            'category': 'water'
        },
        {
            'text': 'Traffic signals not working at the main intersection in {location}. Creating chaos and safety concerns for commuters. #TrafficProblems',
            'sentiment_bias': 'negative',
            'urgency_bias': 'medium',
            'category': 'traffic'
        },
        {
            'text': 'Public transport service has become very unreliable in {location}. Buses are irregular and overcrowded. Need better management. #PublicTransport',
            'sentiment_bias': 'negative',
            'urgency_bias': 'medium',
            'category': 'transport'
        },
        {
            'text': 'Street lights are not working in our area in {location}. Safety concerns at night. Request immediate repair work. #StreetLights',
            'sentiment_bias': 'negative',
            'urgency_bias': 'medium',
            'category': 'infrastructure'
        },
        {
            'text': 'Happy to report that the park maintenance in {location} has improved. Clean and well-maintained green spaces for families. #ParkMaintenance',
            'sentiment_bias': 'positive',
            'urgency_bias': 'low',
            'category': 'parks'
        },
        {
            'text': 'Sewage overflow near residential area in {location}. Health hazard for residents. Emergency action needed by municipal authorities! #SewageProblems',
            'sentiment_bias': 'negative',
            'urgency_bias': 'high',
            'category': 'sanitation'
        }
    ]
    
    # Indian cities with realistic context
    cities = [
        {'name': 'Mumbai', 'tag': 'Mumbai', 'authority': 'BMC'},
        {'name': 'Delhi', 'tag': 'Delhi', 'authority': 'DelhiGov'},
        {'name': 'Bangalore', 'tag': 'Bangalore', 'authority': 'BBMP'},
        {'name': 'Chennai', 'tag': 'Chennai', 'authority': 'ChennaiCorp'},
        {'name': 'Kolkata', 'tag': 'Kolkata', 'authority': 'KMC'},
        {'name': 'Pune', 'tag': 'Pune', 'authority': 'PMC'},
        {'name': 'Hyderabad', 'tag': 'Hyderabad', 'authority': 'GHMC'}
    ]
    
    sources = ['twitter', 'news']
    
    sample_data = []
    base_date = datetime.now()
    
    # Generate 50 realistic records
    for i in range(50):
        template = np.random.choice(issue_templates)
        city = np.random.choice(cities)
        source = np.random.choice(sources, p=[0.7, 0.3])  # More Twitter data
        
        # Generate realistic variations
        days = np.random.randint(1, 8)
        hours = np.random.randint(2, 24)
        
        # Create text based on template
        text = template['text'].format(
            location=city['name'],
            location_tag=city['tag'],
            authority=city['authority'],
            days=days,
            hours=hours
        )
        
        # Adjust for news vs social media
        if source == 'news':
            # Remove hashtags and mentions for news
            text = text.split('#')[0].split('@')[0].strip()
            text = f"News: {text}"
        
        # Generate timestamp (last 30 days)
        timestamp = base_date - pd.Timedelta(days=np.random.randint(0, 30), 
                                           hours=np.random.randint(0, 24))
        
        record = {
            'text': text,
            'source': source,
            'timestamp': timestamp.isoformat(),
            'author_location': f"{city['name']}, India" if source == 'twitter' else None,
            'template_category': template['category'],
            'template_sentiment': template['sentiment_bias'],
            'template_urgency': template['urgency_bias']
        }
        
        sample_data.append(record)
    
    return sample_data

def run_complete_pipeline():
    """Run the complete civic data pipeline with visualizations"""
    
    print("🏛️ Civic Data Ingest System - Complete Pipeline Demo")
    print("=" * 60)
    
    # Step 1: Create realistic sample data
    print("\n📊 Step 1: Creating realistic sample data...")
    raw_data = create_realistic_sample_data()
    print(f"✓ Created {len(raw_data)} sample civic records")
    
    # Step 2: Initialize components
    print("\n🔧 Step 2: Initializing pipeline components...")
    text_cleaner = TextCleaner()
    keyword_extractor = KeywordExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    geo_extractor = GeoExtractor()
    map_visualizer = CivicMapVisualizer()
    trend_visualizer = SentimentTrendVisualizer()
    print("✓ All components initialized")
    
    # Step 3: Text Preprocessing
    print("\n🧹 Step 3: Text preprocessing...")
    processed_data = []
    for record in raw_data:
        if record['source'] == 'twitter':
            cleaned_text = text_cleaner.clean_twitter_text(record['text'], remove_hashtags=True)
        else:
            cleaned_text = text_cleaner.clean_news_text(record['text'])
        
        processed_record = {
            **record,
            'cleaned_text': cleaned_text
        }
        processed_data.append(processed_record)
    
    print(f"✓ Preprocessed {len(processed_data)} records")
    
    # Step 4: Keyword Extraction
    print("\n🔑 Step 4: Extracting keywords...")
    texts = [record['cleaned_text'] for record in processed_data]
    keyword_results = keyword_extractor.extract_keywords_batch(texts, method='civic', top_k=5)
    
    for i, result in enumerate(keyword_results):
        processed_data[i]['keywords'] = result['keywords']
        processed_data[i]['keyword_categories'] = keyword_extractor.get_keyword_categories(
            [kw['keyword'] for kw in result['keywords']]
        )
    
    print(f"✓ Extracted keywords for {len(processed_data)} records")
    
    # Step 5: Sentiment Analysis
    print("\n😊 Step 5: Analyzing sentiment...")
    sentiment_results = sentiment_analyzer.analyze_batch_sentiment(texts, apply_civic_adjustment=True)
    
    for i, result in enumerate(sentiment_results):
        processed_data[i]['sentiment'] = result['final_sentiment']
        processed_data[i]['sentiment_confidence'] = result['confidence']
        processed_data[i]['vader_compound'] = result['vader']['compound']
        processed_data[i]['urgency'] = sentiment_analyzer.classify_urgency_by_sentiment(result)
        processed_data[i]['emotional_indicators'] = sentiment_analyzer.extract_emotional_indicators(
            processed_data[i]['text']
        )
    
    print(f"✓ Analyzed sentiment for {len(processed_data)} records")
    
    # Step 6: Location Extraction
    print("\n🗺️ Step 6: Extracting locations...")
    for record in processed_data:
        text_locations = geo_extractor.extract_locations_from_text(record['text'])
        user_location = record.get('author_location')
        
        locations = []
        if text_locations:
            locations.extend(text_locations)
        if user_location:
            locations.append(user_location.split(',')[0].strip())
        
        record['extracted_locations'] = list(set(locations)) if locations else ['Unknown']
    
    print(f"✓ Extracted locations for {len(processed_data)} records")
    
    # Step 7: Generate Summary Statistics
    print("\n📈 Step 7: Generating summary statistics...")
    
    # Basic stats
    total_records = len(processed_data)
    sentiment_dist = {}
    urgency_dist = {}
    location_dist = {}
    category_dist = {}
    
    for record in processed_data:
        # Sentiment distribution
        sentiment = record['sentiment']
        sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1
        
        # Urgency distribution
        urgency = record['urgency']
        urgency_dist[urgency] = urgency_dist.get(urgency, 0) + 1
        
        # Location distribution
        location = record['extracted_locations'][0] if record['extracted_locations'] else 'Unknown'
        location_dist[location] = location_dist.get(location, 0) + 1
        
        # Category distribution (from template)
        category = record.get('template_category', 'other')
        category_dist[category] = category_dist.get(category, 0) + 1
    
    summary_stats = {
        'total_records': total_records,
        'sentiment_distribution': sentiment_dist,
        'urgency_distribution': urgency_dist,
        'location_distribution': location_dist,
        'category_distribution': category_dist,
        'processing_timestamp': create_timestamp()
    }
    
    print("✓ Summary statistics generated")
    
    # Step 8: Create Visualizations
    print("\n📊 Step 8: Creating visualizations...")
    
    # Ensure output directory exists
    os.makedirs('demo_output', exist_ok=True)
    
    # Map visualizations
    print("   Creating map visualizations...")
    map_visualizer.create_heatmap(processed_data, 'demo_output/demo_heatmap.html')
    map_visualizer.create_category_map(processed_data, 'demo_output/demo_category_map.html')
    
    # Trend visualizations
    print("   Creating trend visualizations...")
    trend_visualizer.create_sentiment_timeline(processed_data, 'demo_output/demo_sentiment_timeline.html')
    trend_visualizer.create_urgency_distribution(processed_data, 'demo_output/demo_urgency_distribution.html')
    trend_visualizer.create_hourly_pattern(processed_data, 'demo_output/demo_hourly_pattern.html')
    trend_visualizer.create_location_sentiment_heatmap(processed_data, 'demo_output/demo_location_heatmap.html')
    trend_visualizer.create_comprehensive_dashboard(processed_data, 'demo_output/demo_comprehensive_dashboard.html')
    
    print("✓ All visualizations created")
    
    # Step 9: Save Results
    print("\n💾 Step 9: Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save processed data
    processed_file = f'demo_output/demo_processed_data_{timestamp}.json'
    save_json(processed_data, processed_file)
    
    # Save summary statistics
    summary_file = f'demo_output/demo_summary_stats_{timestamp}.json'
    save_json(summary_stats, summary_file)
    
    print(f"✓ Results saved to demo_output/ directory")
    
    # Step 10: Display Results Summary
    print("\n📋 Step 10: Results Summary")
    print("-" * 40)
    print(f"Total Issues Processed: {summary_stats['total_records']}")
    print(f"Sentiment Distribution:")
    for sentiment, count in summary_stats['sentiment_distribution'].items():
        percentage = (count / total_records) * 100
        print(f"  - {sentiment.title()}: {count} ({percentage:.1f}%)")
    
    print(f"\nUrgency Distribution:")
    for urgency, count in summary_stats['urgency_distribution'].items():
        percentage = (count / total_records) * 100
        print(f"  - {urgency.title()}: {count} ({percentage:.1f}%)")
    
    print(f"\nTop Locations:")
    sorted_locations = sorted(summary_stats['location_distribution'].items(), 
                             key=lambda x: x[1], reverse=True)
    for location, count in sorted_locations[:5]:
        percentage = (count / total_records) * 100
        print(f"  - {location}: {count} ({percentage:.1f}%)")
    
    print(f"\nIssue Categories:")
    for category, count in summary_stats['category_distribution'].items():
        percentage = (count / total_records) * 100
        print(f"  - {category.title()}: {count} ({percentage:.1f}%)")
    
    # Display sample processed record
    print(f"\n📄 Sample Processed Record:")
    print("-" * 40)
    sample = processed_data[0]
    print(f"Original Text: {sample['text'][:100]}...")
    print(f"Cleaned Text: {sample['cleaned_text'][:100]}...")
    print(f"Source: {sample['source']}")
    print(f"Location: {sample['extracted_locations']}")
    print(f"Sentiment: {sample['sentiment']} (confidence: {sample['sentiment_confidence']:.3f})")
    print(f"VADER Score: {sample['vader_compound']:.3f}")
    print(f"Urgency: {sample['urgency']}")
    print(f"Keywords: {[kw['keyword'] for kw in sample['keywords'][:3]]}")
    
    # Files created
    print(f"\n📁 Generated Files:")
    print("-" * 40)
    print("📊 Visualizations:")
    print("  - demo_output/demo_heatmap.html")
    print("  - demo_output/demo_category_map.html")
    print("  - demo_output/demo_sentiment_timeline.html")
    print("  - demo_output/demo_urgency_distribution.html")
    print("  - demo_output/demo_hourly_pattern.html")
    print("  - demo_output/demo_location_heatmap.html")
    print("  - demo_output/demo_comprehensive_dashboard.html")
    print("\n💾 Data Files:")
    print(f"  - {processed_file}")
    print(f"  - {summary_file}")
    
    print(f"\n🎉 Pipeline Demo Completed Successfully!")
    print("Open the HTML files in your browser to explore the interactive visualizations.")
    
    return processed_data, summary_stats

def main():
    """Run the complete pipeline demo"""
    try:
        processed_data, summary_stats = run_complete_pipeline()
        
        print(f"\n🚀 To launch the interactive dashboard:")
        print("Run: streamlit run visualization/dashboard_app.py")
        print("Then select 'Load from Output' and choose the generated JSON file.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()