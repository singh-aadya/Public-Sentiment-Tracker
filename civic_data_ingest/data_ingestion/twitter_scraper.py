"""
Twitter Data Scraper using Tweepy API
Collects tweets based on civic-related keywords and hashtags
"""

import tweepy
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class TwitterScraper:
    def __init__(self):
        """Initialize Twitter API client"""
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        # Initialize Tweepy client
        self.client = tweepy.Client(
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            wait_on_rate_limit=True
        )
        
        # Civic-related keywords and hashtags
        self.civic_keywords = [
            'water supply', 'electricity outage', 'potholes', 'street lights',
            'garbage collection', 'sewage', 'road repair', 'traffic signal',
            'park maintenance', 'public transport'
        ]
        
        self.civic_hashtags = [
            '#pothole', '#powercut', '#watersupply', '#trafficjam',
            '#garbagecollection', '#streetlights', '#publictransport',
            '#roadrepair', '#sewage', '#parkmaintenance'
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def build_search_query(self, keywords: List[str] = None, hashtags: List[str] = None) -> str:
        """Build search query from keywords and hashtags"""
        if keywords is None:
            keywords = self.civic_keywords[:5]  # Limit to avoid query length issues
        if hashtags is None:
            hashtags = self.civic_hashtags[:5]
        
        # Combine keywords and hashtags with OR operator
        query_parts = []
        query_parts.extend([f'"{keyword}"' for keyword in keywords])
        query_parts.extend(hashtags)
        
        # Join with OR and add language filter
        query = ' OR '.join(query_parts)
        query += ' -is:retweet lang:en'  # Exclude retweets, English only
        
        return query
    
    def scrape_tweets(self, 
                     max_results: int = 100,
                     days_back: int = 1,
                     keywords: List[str] = None,
                     hashtags: List[str] = None) -> List[Dict]:
        """
        Scrape tweets based on civic-related keywords and hashtags
        
        Args:
            max_results: Maximum number of tweets to fetch
            days_back: How many days back to search
            keywords: Custom keywords (optional)
            hashtags: Custom hashtags (optional)
            
        Returns:
            List of tweet dictionaries
        """
        try:
            query = self.build_search_query(keywords, hashtags)
            
            # Calculate start time
            start_time = datetime.utcnow() - timedelta(days=days_back)
            
            self.logger.info(f"Searching tweets with query: {query}")
            
            # Search tweets
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'geo', 'context_annotations'],
                user_fields=['location', 'verified'],
                expansions=['author_id'],
                start_time=start_time,
                max_results=min(max_results, 100)  # API limit per request
            ).flatten(limit=max_results)
            
            # Process tweets
            processed_tweets = []
            users_dict = {}
            
            # Get user information if available
            for tweet in tweets:
                if hasattr(tweet, 'includes') and 'users' in tweet.includes:
                    for user in tweet.includes['users']:
                        users_dict[user.id] = user
            
            for tweet in tweets:
                # Get user info
                user_info = users_dict.get(tweet.author_id, {})
                
                tweet_data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at.isoformat() if tweet.created_at else None,
                    'author_id': tweet.author_id,
                    'author_location': getattr(user_info, 'location', None),
                    'author_verified': getattr(user_info, 'verified', False),
                    'retweet_count': tweet.public_metrics.get('retweet_count', 0) if tweet.public_metrics else 0,
                    'like_count': tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                    'reply_count': tweet.public_metrics.get('reply_count', 0) if tweet.public_metrics else 0,
                    'geo': tweet.geo,
                    'context_annotations': tweet.context_annotations,
                    'source': 'twitter',
                    'scraped_at': datetime.utcnow().isoformat()
                }
                
                processed_tweets.append(tweet_data)
            
            self.logger.info(f"Successfully scraped {len(processed_tweets)} tweets")
            return processed_tweets
            
        except Exception as e:
            self.logger.error(f"Error scraping tweets: {str(e)}")
            return []
    
    def save_tweets_to_json(self, tweets: List[Dict], filename: str = None) -> str:
        """Save tweets to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tweets_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(tweets, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Tweets saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving tweets: {str(e)}")
            return ""

def main():
    """Test the Twitter scraper"""
    scraper = TwitterScraper()
    
    # Test scraping
    tweets = scraper.scrape_tweets(max_results=50, days_back=1)
    
    if tweets:
        # Save to file
        filename = scraper.save_tweets_to_json(tweets)
        print(f"Scraped {len(tweets)} tweets and saved to {filename}")
        
        # Print sample tweet
        print("\nSample tweet:")
        print(json.dumps(tweets[0], indent=2))
    else:
        print("No tweets found")

if __name__ == "__main__":
    main()