"""
Main Pipeline Runner for Civic Data Ingest System
Orchestrates the end-to-end ETL pipeline
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.utils import (
    setup_logging, load_config, save_json, create_timestamp,
    create_data_pipeline_status, update_pipeline_status
)

from data_ingestion.twitter_scraper import TwitterScraper
from data_ingestion.news_scraper import NewsScraper
from data_ingestion.geo_extractor import GeoExtractor
from data_ingestion.db_connector import DatabaseManager

from preprocessing.cleaner import TextCleaner
from preprocessing.vectorizer import TextVectorizer

from nlp_modeling.keyword_extractor import KeywordExtractor
from nlp_modeling.sentiment_analyzer import SentimentAnalyzer

class CivicDataPipeline:
    def __init__(self, config_file: str = ".env"):
        """
        Initialize the civic data pipeline
        
        Args:
            config_file: Path to configuration file
        """
        self.config = load_config(config_file)
        self.setup_logging()
        
        # Initialize components
        self.twitter_scraper = TwitterScraper()
        self.news_scraper = NewsScraper()
        self.geo_extractor = GeoExtractor()
        self.text_cleaner = TextCleaner()
        self.text_vectorizer = TextVectorizer()
        self.keyword_extractor = KeywordExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Database manager
        db_type = self.config.get('DATABASE_TYPE', 'mongodb')
        self.db_manager = DatabaseManager(db_type)
        
        # Pipeline status
        self.status = create_data_pipeline_status()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Civic Data Pipeline initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('LOG_LEVEL', 'INFO')
        log_file = self.config.get('LOG_FILE', 'civic_pipeline.log')
        setup_logging(log_level, log_file)
    
    def collect_data(self, 
                    collect_twitter: bool = True,
                    collect_news: bool = True,
                    max_tweets: int = 100,
                    max_articles: int = 50) -> List[Dict]:
        """
        Collect data from various sources
        
        Args:
            collect_twitter: Whether to collect Twitter data
            collect_news: Whether to collect news data
            max_tweets: Maximum tweets to collect
            max_articles: Maximum news articles to collect
            
        Returns:
            List of collected data records
        """
        self.logger.info("Starting data collection phase")
        
        try:
            all_data = []
            
            # Collect Twitter data
            if collect_twitter:
                self.logger.info(f"Collecting Twitter data (max: {max_tweets})")
                tweets = self.twitter_scraper.scrape_tweets(max_results=max_tweets)
                all_data.extend(tweets)
                self.logger.info(f"Collected {len(tweets)} tweets")
            
            # Collect news data
            if collect_news:
                self.logger.info(f"Collecting news data (max: {max_articles})")
                articles = self.news_scraper.scrape_news_batch(max_articles_per_source=max_articles)
                # Filter civic-related articles
                civic_articles = self.news_scraper.filter_civic_articles(articles)
                all_data.extend(civic_articles)
                self.logger.info(f"Collected {len(civic_articles)} civic news articles")
            
            # Update status
            self.status = update_pipeline_status(
                self.status, 'data_collection', 'completed', len(all_data)
            )
            
            return all_data
            
        except Exception as e:
            error_msg = f"Error in data collection: {str(e)}"
            self.logger.error(error_msg)
            self.status = update_pipeline_status(
                self.status, 'data_collection', 'failed', error=error_msg
            )
            return []
    
    def preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """
        Preprocess collected data
        
        Args:
            data: Raw data records
            
        Returns:
            Preprocessed data records
        """
        self.logger.info("Starting preprocessing phase")
        
        try:
            if not data:
                self.logger.warning("No data to preprocess")
                return []
            
            processed_data = []
            
            for record in data:
                # Extract text based on source
                if record.get('source') == 'twitter':
                    text = record.get('text', '')
                    cleaned_text = self.text_cleaner.clean_twitter_text(text, remove_hashtags=True)
                else:
                    text = record.get('text', '')
                    cleaned_text = self.text_cleaner.clean_news_text(text)
                
                # Create processed record
                processed_record = {
                    'original_text': text,
                    'cleaned_text': cleaned_text,
                    'source': record.get('source'),
                    'timestamp': record.get('timestamp') or record.get('created_at'),
                    'raw_data': record
                }
                
                processed_data.append(processed_record)
            
            # Fit TF-IDF vectorizer on cleaned texts
            cleaned_texts = [r['cleaned_text'] for r in processed_data if r['cleaned_text']]
            if cleaned_texts:
                self.logger.info("Fitting TF-IDF vectorizer")
                self.text_vectorizer.fit_tfidf(cleaned_texts)
            
            # Update status
            self.status = update_pipeline_status(
                self.status, 'preprocessing', 'completed', len(processed_data)
            )
            
            self.logger.info(f"Preprocessed {len(processed_data)} records")
            return processed_data
            
        except Exception as e:
            error_msg = f"Error in preprocessing: {str(e)}"
            self.logger.error(error_msg)
            self.status = update_pipeline_status(
                self.status, 'preprocessing', 'failed', error=error_msg
            )
            return []
    
    def extract_keywords(self, data: List[Dict]) -> List[Dict]:
        """
        Extract keywords from preprocessed data
        
        Args:
            data: Preprocessed data records
            
        Returns:
            Data records with extracted keywords
        """
        self.logger.info("Starting keyword extraction phase")
        
        try:
            if not data:
                self.logger.warning("No data for keyword extraction")
                return []
            
            # Extract texts for batch processing
            texts = [record['cleaned_text'] for record in data if record.get('cleaned_text')]
            
            if not texts:
                self.logger.warning("No valid texts for keyword extraction")
                return data
            
            # Extract keywords using combined method
            self.logger.info("Extracting keywords using combined methods")
            keyword_results = self.keyword_extractor.extract_keywords_batch(
                texts, method='combined', top_k=10
            )
            
            # Add keywords to data records
            for i, record in enumerate(data):
                if i < len(keyword_results) and record.get('cleaned_text'):
                    record['keywords'] = keyword_results[i]['keywords']
                    record['keyword_categories'] = self.keyword_extractor.get_keyword_categories(
                        [kw['keyword'] for kw in keyword_results[i]['keywords']]
                    )
                else:
                    record['keywords'] = []
                    record['keyword_categories'] = {}
            
            # Update status
            self.status = update_pipeline_status(
                self.status, 'keyword_extraction', 'completed', len(data)
            )
            
            self.logger.info(f"Extracted keywords for {len(data)} records")
            return data
            
        except Exception as e:
            error_msg = f"Error in keyword extraction: {str(e)}"
            self.logger.error(error_msg)
            self.status = update_pipeline_status(
                self.status, 'keyword_extraction', 'failed', error=error_msg
            )
            return data
    
    def analyze_sentiment(self, data: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment of preprocessed data
        
        Args:
            data: Preprocessed data records
            
        Returns:
            Data records with sentiment analysis
        """
        self.logger.info("Starting sentiment analysis phase")
        
        try:
            if not data:
                self.logger.warning("No data for sentiment analysis")
                return []
            
            # Extract texts for batch processing
            texts = [record['cleaned_text'] for record in data if record.get('cleaned_text')]
            
            if not texts:
                self.logger.warning("No valid texts for sentiment analysis")
                return data
            
            # Analyze sentiment
            self.logger.info("Analyzing sentiment using VADER")
            sentiment_results = self.sentiment_analyzer.analyze_batch_sentiment(
                texts, apply_civic_adjustment=True
            )
            
            # Add sentiment analysis to data records
            for i, record in enumerate(data):
                if i < len(sentiment_results) and record.get('cleaned_text'):
                    sentiment_data = sentiment_results[i]
                    record['sentiment'] = {
                        'final_sentiment': sentiment_data['final_sentiment'],
                        'confidence': sentiment_data['confidence'],
                        'vader_compound': sentiment_data['vader']['compound'],
                        'vader_scores': sentiment_data['vader']
                    }
                    
                    # Add urgency classification
                    record['urgency'] = self.sentiment_analyzer.classify_urgency_by_sentiment(sentiment_data)
                    
                    # Extract emotional indicators
                    record['emotional_indicators'] = self.sentiment_analyzer.extract_emotional_indicators(
                        record['original_text']
                    )
                else:
                    record['sentiment'] = {
                        'final_sentiment': 'neutral',
                        'confidence': 0.0,
                        'vader_compound': 0.0,
                        'vader_scores': {}
                    }
                    record['urgency'] = 'low'
                    record['emotional_indicators'] = {}
            
            # Update status
            self.status = update_pipeline_status(
                self.status, 'sentiment_analysis', 'completed', len(data)
            )
            
            self.logger.info(f"Analyzed sentiment for {len(data)} records")
            return data
            
        except Exception as e:
            error_msg = f"Error in sentiment analysis: {str(e)}"
            self.logger.error(error_msg)
            self.status = update_pipeline_status(
                self.status, 'sentiment_analysis', 'failed', error=error_msg
            )
            return data
    
    def extract_locations(self, data: List[Dict]) -> List[Dict]:
        """
        Extract geographic locations from data
        
        Args:
            data: Processed data records
            
        Returns:
            Data records with location information
        """
        self.logger.info("Starting location extraction phase")
        
        try:
            if not data:
                self.logger.warning("No data for location extraction")
                return []
            
            for record in data:
                locations = []
                
                # Extract from original text
                original_text = record.get('original_text', '')
                if original_text:
                    text_locations = self.geo_extractor.extract_and_geocode(original_text)
                    locations.extend(text_locations)
                
                # Process social media location data
                raw_data = record.get('raw_data', {})
                if raw_data.get('source') == 'twitter':
                    user_location = raw_data.get('author_location')
                    geo_data = raw_data.get('geo')
                    
                    if user_location or geo_data:
                        social_location = self.geo_extractor.process_social_media_location(
                            user_location, geo_data
                        )
                        if social_location:
                            locations.append(social_location)
                
                record['locations'] = locations
                
                # Add primary location (first valid location)
                if locations:
                    record['primary_location'] = locations[0]
                else:
                    record['primary_location'] = None
            
            # Update status
            self.status = update_pipeline_status(
                self.status, 'geo_extraction', 'completed', len(data)
            )
            
            self.logger.info(f"Extracted locations for {len(data)} records")
            return data
            
        except Exception as e:
            error_msg = f"Error in location extraction: {str(e)}"
            self.logger.error(error_msg)
            self.status = update_pipeline_status(
                self.status, 'geo_extraction', 'failed', error=error_msg
            )
            return data
    
    def save_results(self, data: List[Dict], output_dir: str = "output") -> str:
        """
        Save pipeline results to files
        
        Args:
            data: Processed data
            output_dir: Output directory
            
        Returns:
            Path to main output file
        """
        self.logger.info("Saving pipeline results")
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Create timestamp for files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save main results
            main_output_file = os.path.join(output_dir, f"civic_data_results_{timestamp}.json")
            save_json(data, main_output_file)
            
            # Save pipeline status
            status_file = os.path.join(output_dir, f"pipeline_status_{timestamp}.json")
            save_json(self.status, status_file)
            
            # Create summary statistics
            summary = self.create_summary_stats(data)
            summary_file = os.path.join(output_dir, f"summary_stats_{timestamp}.json")
            save_json(summary, summary_file)
            
            self.logger.info(f"Results saved to {main_output_file}")
            return main_output_file
            
        except Exception as e:
            error_msg = f"Error saving results: {str(e)}"
            self.logger.error(error_msg)
            return ""
    
    def create_summary_stats(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Create summary statistics of processed data
        
        Args:
            data: Processed data
            
        Returns:
            Summary statistics
        """
        if not data:
            return {}
        
        # Basic statistics
        total_records = len(data)
        sources = [record.get('source', 'unknown') for record in data]
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Sentiment statistics
        sentiments = [record.get('sentiment', {}).get('final_sentiment', 'neutral') for record in data]
        sentiment_counts = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Urgency statistics
        urgencies = [record.get('urgency', 'low') for record in data]
        urgency_counts = {}
        for urgency in urgencies:
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
        
        # Location statistics
        records_with_location = sum(1 for record in data if record.get('primary_location'))
        
        return {
            'total_records': total_records,
            'source_distribution': source_counts,
            'sentiment_distribution': sentiment_counts,
            'urgency_distribution': urgency_counts,
            'records_with_location': records_with_location,
            'location_coverage': (records_with_location / total_records) * 100 if total_records > 0 else 0,
            'pipeline_status': self.status['status'],
            'created_at': create_timestamp()
        }
    
    def run_pipeline(self, 
                    collect_twitter: bool = True,
                    collect_news: bool = True,
                    max_tweets: int = 100,
                    max_articles: int = 50,
                    output_dir: str = "output") -> str:
        """
        Run the complete civic data pipeline
        
        Args:
            collect_twitter: Whether to collect Twitter data
            collect_news: Whether to collect news data
            max_tweets: Maximum tweets to collect
            max_articles: Maximum news articles to collect
            output_dir: Output directory for results
            
        Returns:
            Path to main output file
        """
        self.logger.info("Starting Civic Data Pipeline")
        
        try:
            # Phase 1: Data Collection
            raw_data = self.collect_data(collect_twitter, collect_news, max_tweets, max_articles)
            
            if not raw_data:
                self.logger.error("No data collected. Stopping pipeline.")
                return ""
            
            # Phase 2: Preprocessing
            processed_data = self.preprocess_data(raw_data)
            
            # Phase 3: Keyword Extraction
            processed_data = self.extract_keywords(processed_data)
            
            # Phase 4: Sentiment Analysis
            processed_data = self.analyze_sentiment(processed_data)
            
            # Phase 5: Location Extraction
            processed_data = self.extract_locations(processed_data)
            
            # Phase 6: Save Results
            output_file = self.save_results(processed_data, output_dir)
            
            self.logger.info("Civic Data Pipeline completed successfully")
            return output_file
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            self.status['status'] = 'failed'
            self.status['errors'].append({
                'stage': 'pipeline',
                'error': error_msg,
                'timestamp': create_timestamp()
            })
            return ""

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='Civic Data Ingest Pipeline')
    parser.add_argument('--no-twitter', action='store_true', help='Skip Twitter data collection')
    parser.add_argument('--no-news', action='store_true', help='Skip news data collection')
    parser.add_argument('--max-tweets', type=int, default=50, help='Maximum tweets to collect')
    parser.add_argument('--max-articles', type=int, default=20, help='Maximum news articles to collect')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--config', default='.env', help='Configuration file')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CivicDataPipeline(config_file=args.config)
    
    # Run pipeline
    output_file = pipeline.run_pipeline(
        collect_twitter=not args.no_twitter,
        collect_news=not args.no_news,
        max_tweets=args.max_tweets,
        max_articles=args.max_articles,
        output_dir=args.output_dir
    )
    
    if output_file:
        print(f"Pipeline completed successfully. Results saved to: {output_file}")
    else:
        print("Pipeline failed. Check logs for details.")

if __name__ == "__main__":
    main()