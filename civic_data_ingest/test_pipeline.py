"""
Test script for the Civic Data Pipeline
Demonstrates basic functionality without requiring API keys
"""

import sys
import os
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.cleaner import TextCleaner
from preprocessing.vectorizer import TextVectorizer
from nlp_modeling.keyword_extractor import KeywordExtractor
from nlp_modeling.sentiment_analyzer import SentimentAnalyzer
from data_ingestion.geo_extractor import GeoExtractor
from shared.utils import save_json, create_timestamp

def create_sample_data():
    """Create sample civic data for testing"""
    return [
        {
            'text': 'Water crisis in Mumbai! 😰 Residents facing shortage for 3 days. @MumbaiMunicipal please take immediate action! #WaterCrisis #Mumbai',
            'source': 'twitter',
            'timestamp': '2024-01-15T10:30:00Z',
            'author_location': 'Mumbai, Maharashtra'
        },
        {
            'text': 'Terrible road conditions in Bangalore. Potholes everywhere causing traffic problems. The municipal corporation needs to fix this URGENTLY!!! #BangaloreRoads',
            'source': 'twitter', 
            'timestamp': '2024-01-15T11:45:00Z',
            'author_location': 'Bangalore, Karnataka'
        },
        {
            'text': 'Power outage in our area for the third consecutive day. This is completely unacceptable. When will the electricity board take action?',
            'source': 'twitter',
            'timestamp': '2024-01-15T12:15:00Z',
            'author_location': 'Delhi'
        },
        {
            'text': 'The municipal corporation announced new measures to address the water supply issues in the eastern districts. The initiative includes installation of new pipelines and improvement of existing infrastructure.',
            'source': 'news',
            'timestamp': '2024-01-15T09:00:00Z',
            'url': 'https://example-news.com/water-supply-measures'
        },
        {
            'text': 'Garbage collection has improved significantly in our locality over the past month. Thanks to the efficient work by the sanitation department.',
            'source': 'twitter',
            'timestamp': '2024-01-15T13:20:00Z',
            'author_location': 'Pune, Maharashtra'
        }
    ]

def test_basic_pipeline():
    """Test the basic pipeline functionality"""
    print("Testing Civic Data Pipeline - Basic Components")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_data()
    print(f"Created {len(sample_data)} sample records")
    
    # Initialize components
    text_cleaner = TextCleaner()
    text_vectorizer = TextVectorizer(max_features=500)
    keyword_extractor = KeywordExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    geo_extractor = GeoExtractor()
    
    # Phase 1: Text Cleaning
    print(f"\nPhase 1: Text Cleaning")
    print("-" * 30)
    
    cleaned_data = []
    for record in sample_data:
        if record['source'] == 'twitter':
            cleaned_text = text_cleaner.clean_twitter_text(record['text'], remove_hashtags=True)
        else:
            cleaned_text = text_cleaner.clean_news_text(record['text'])
        
        record['cleaned_text'] = cleaned_text
        cleaned_data.append(record)
        
        print(f"Original: {record['text'][:80]}...")
        print(f"Cleaned:  {cleaned_text[:80]}...")
        print()
    
    # Phase 2: Keyword Extraction
    print(f"Phase 2: Keyword Extraction")
    print("-" * 30)
    
    texts = [record['cleaned_text'] for record in cleaned_data]
    keyword_results = keyword_extractor.extract_keywords_batch(texts, method='civic', top_k=5)
    
    for i, result in enumerate(keyword_results):
        cleaned_data[i]['keywords'] = result['keywords']
        print(f"Text {i+1}: {result['text_preview'][:60]}...")
        print(f"Keywords: {[kw['keyword'] for kw in result['keywords']]}")
        print()
    
    # Phase 3: Sentiment Analysis
    print(f"Phase 3: Sentiment Analysis")
    print("-" * 30)
    
    sentiment_results = sentiment_analyzer.analyze_batch_sentiment(texts, apply_civic_adjustment=True)
    
    for i, result in enumerate(sentiment_results):
        cleaned_data[i]['sentiment'] = result['final_sentiment']
        cleaned_data[i]['sentiment_confidence'] = result['confidence']
        cleaned_data[i]['vader_compound'] = result['vader']['compound']
        
        # Classify urgency
        urgency = sentiment_analyzer.classify_urgency_by_sentiment(result)
        cleaned_data[i]['urgency'] = urgency
        
        print(f"Text {i+1}: {result['text_preview'][:60]}...")
        print(f"Sentiment: {result['final_sentiment']} (confidence: {result['confidence']:.3f})")
        print(f"VADER Score: {result['vader']['compound']:.3f}")
        print(f"Urgency: {urgency}")
        print()
    
    # Phase 4: Location Extraction
    print(f"Phase 4: Location Extraction")
    print("-" * 30)
    
    for i, record in enumerate(cleaned_data):
        # Extract locations from text
        text_locations = geo_extractor.extract_locations_from_text(record['text'])
        
        # Process user location if available
        user_location = record.get('author_location')
        
        locations = []
        if text_locations:
            locations.extend(text_locations)
        if user_location:
            locations.append(user_location)
        
        record['extracted_locations'] = list(set(locations))  # Remove duplicates
        
        print(f"Text {i+1}: {record['text'][:60]}...")
        print(f"Locations: {record['extracted_locations']}")
        print()
    
    # Phase 5: Create Summary Statistics
    print(f"Phase 5: Summary Statistics")
    print("-" * 30)
    
    # Sentiment distribution
    sentiments = [record['sentiment'] for record in cleaned_data]
    sentiment_counts = {}
    for sentiment in sentiments:
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    # Urgency distribution
    urgencies = [record['urgency'] for record in cleaned_data]
    urgency_counts = {}
    for urgency in urgencies:
        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
    
    # Source distribution
    sources = [record['source'] for record in cleaned_data]
    source_counts = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    
    summary = {
        'total_records': len(cleaned_data),
        'sentiment_distribution': sentiment_counts,
        'urgency_distribution': urgency_counts,
        'source_distribution': source_counts,
        'records_with_locations': sum(1 for record in cleaned_data if record['extracted_locations']),
        'processed_at': create_timestamp()
    }
    
    print(f"Total Records: {summary['total_records']}")
    print(f"Sentiment Distribution: {summary['sentiment_distribution']}")
    print(f"Urgency Distribution: {summary['urgency_distribution']}")
    print(f"Source Distribution: {summary['source_distribution']}")
    print(f"Records with Locations: {summary['records_with_locations']}")
    
    # Phase 6: Save Results
    print(f"\nPhase 6: Save Results")
    print("-" * 30)
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"output/test_pipeline_results_{timestamp}.json"
    summary_file = f"output/test_pipeline_summary_{timestamp}.json"
    
    save_json(cleaned_data, results_file)
    save_json(summary, summary_file)
    
    print(f"Processed data saved to: {results_file}")
    print(f"Summary statistics saved to: {summary_file}")
    
    return results_file, summary_file

def display_sample_record(record):
    """Display a nicely formatted sample record"""
    print("\nSample Processed Record:")
    print("=" * 40)
    print(f"Original Text: {record['text']}")
    print(f"Cleaned Text: {record['cleaned_text']}")
    print(f"Source: {record['source']}")
    print(f"Timestamp: {record['timestamp']}")
    print(f"Keywords: {[kw['keyword'] for kw in record.get('keywords', [])]}")
    print(f"Sentiment: {record.get('sentiment', 'N/A')} (confidence: {record.get('sentiment_confidence', 0):.3f})")
    print(f"VADER Score: {record.get('vader_compound', 0):.3f}")
    print(f"Urgency: {record.get('urgency', 'N/A')}")
    print(f"Locations: {record.get('extracted_locations', [])}")

def main():
    """Main function to run the test"""
    try:
        # Run the test pipeline
        results_file, summary_file = test_basic_pipeline()
        
        # Load and display a sample result
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if results:
            display_sample_record(results[0])
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Check the output files for detailed results:")
        print(f"   - {results_file}")
        print(f"   - {summary_file}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()