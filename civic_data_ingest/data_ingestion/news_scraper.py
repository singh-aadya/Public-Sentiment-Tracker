"""
News Scraper using Newspaper3k and BeautifulSoup
Scrapes civic-related news from local news websites
"""

import requests
from newspaper import Article, Config
from bs4 import BeautifulSoup
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import time
import re

class NewsScraper:
    def __init__(self):
        """Initialize news scraper with configuration"""
        self.config = Config()
        self.config.request_timeout = 10
        self.config.number_threads = 3
        
        # Common news sources (you can extend this list)
        self.news_sources = [
            'https://timesofindia.indiatimes.com',
            'https://indianexpress.com',
            'https://hindustantimes.com',
            'https://thehindu.com'
        ]
        
        # Keywords for civic issues
        self.civic_keywords = [
            'water crisis', 'power outage', 'pothole', 'road condition',
            'garbage collection', 'sewage problem', 'street light',
            'traffic congestion', 'public transport', 'park maintenance',
            'municipal', 'civic body', 'corporation', 'infrastructure'
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_civic_news_urls(self, base_url: str, max_pages: int = 3) -> List[str]:
        """
        Extract URLs of civic-related news articles from a news website
        
        Args:
            base_url: Base URL of the news website
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of article URLs
        """
        article_urls = []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Try different section URLs that might contain civic news
            sections = ['/city', '/local', '/state', '/mumbai', '/delhi', '/bangalore', '/chennai']
            
            for section in sections:
                try:
                    url = urljoin(base_url, section)
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Find all article links
                        article_links = soup.find_all('a', href=True)
                        
                        for link in article_links:
                            href = link.get('href')
                            if href:
                                # Convert relative URLs to absolute
                                full_url = urljoin(base_url, href)
                                
                                # Check if the link text or URL contains civic keywords
                                link_text = link.get_text().lower()
                                url_text = href.lower()
                                
                                if any(keyword in link_text or keyword in url_text 
                                      for keyword in self.civic_keywords):
                                    article_urls.append(full_url)
                        
                        time.sleep(1)  # Be respectful to the server
                        
                except Exception as e:
                    self.logger.warning(f"Error scraping section {section}: {str(e)}")
                    continue
            
            # Remove duplicates
            article_urls = list(set(article_urls))
            self.logger.info(f"Found {len(article_urls)} civic-related articles from {base_url}")
            
        except Exception as e:
            self.logger.error(f"Error getting news URLs from {base_url}: {str(e)}")
        
        return article_urls
    
    def scrape_article(self, url: str) -> Optional[Dict]:
        """
        Scrape a single news article
        
        Args:
            url: URL of the article
            
        Returns:
            Dictionary containing article data
        """
        try:
            article = Article(url, config=self.config)
            article.download()
            article.parse()
            
            # Extract basic information
            article_data = {
                'url': url,
                'title': article.title,
                'text': article.text,
                'summary': article.summary,
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'top_image': article.top_image,
                'keywords': article.keywords,
                'source': urlparse(url).netloc,
                'scraped_at': datetime.utcnow().isoformat()
            }
            
            # Additional processing
            article_data['word_count'] = len(article.text.split()) if article.text else 0
            article_data['has_civic_keywords'] = any(
                keyword in article.text.lower() 
                for keyword in self.civic_keywords
            ) if article.text else False
            
            return article_data
            
        except Exception as e:
            self.logger.error(f"Error scraping article {url}: {str(e)}")
            return None
    
    def scrape_news_batch(self, 
                         sources: List[str] = None, 
                         max_articles_per_source: int = 20) -> List[Dict]:
        """
        Scrape news articles from multiple sources
        
        Args:
            sources: List of news source URLs
            max_articles_per_source: Maximum articles to scrape per source
            
        Returns:
            List of article dictionaries
        """
        if sources is None:
            sources = self.news_sources
        
        all_articles = []
        
        for source in sources:
            self.logger.info(f"Scraping from {source}")
            
            try:
                # Get article URLs
                article_urls = self.get_civic_news_urls(source)
                
                # Limit the number of articles per source
                article_urls = article_urls[:max_articles_per_source]
                
                # Scrape each article
                for url in article_urls:
                    article_data = self.scrape_article(url)
                    if article_data:
                        all_articles.append(article_data)
                    
                    time.sleep(0.5)  # Be respectful to the server
                
                self.logger.info(f"Scraped {len([a for a in all_articles if a['source'] == urlparse(source).netloc])} articles from {source}")
                
            except Exception as e:
                self.logger.error(f"Error scraping {source}: {str(e)}")
                continue
        
        self.logger.info(f"Total articles scraped: {len(all_articles)}")
        return all_articles
    
    def filter_civic_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Filter articles to keep only civic-related ones
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Filtered list of civic-related articles
        """
        civic_articles = []
        
        for article in articles:
            # Check if article contains civic keywords
            text_to_check = f"{article.get('title', '')} {article.get('text', '')}"
            text_to_check = text_to_check.lower()
            
            # Count civic keyword matches
            keyword_matches = sum(1 for keyword in self.civic_keywords if keyword in text_to_check)
            
            if keyword_matches >= 2:  # Require at least 2 keyword matches
                article['civic_keyword_count'] = keyword_matches
                civic_articles.append(article)
        
        self.logger.info(f"Filtered to {len(civic_articles)} civic-related articles")
        return civic_articles
    
    def save_articles_to_json(self, articles: List[Dict], filename: str = None) -> str:
        """Save articles to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_articles_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Articles saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving articles: {str(e)}")
            return ""

def main():
    """Test the news scraper"""
    scraper = NewsScraper()
    
    # Test with a smaller subset for demonstration
    test_sources = ['https://timesofindia.indiatimes.com']
    
    # Scrape articles
    articles = scraper.scrape_news_batch(sources=test_sources, max_articles_per_source=5)
    
    if articles:
        # Filter civic-related articles
        civic_articles = scraper.filter_civic_articles(articles)
        
        # Save to file
        filename = scraper.save_articles_to_json(civic_articles)
        print(f"Scraped {len(civic_articles)} civic articles and saved to {filename}")
        
        # Print sample article
        if civic_articles:
            print("\nSample article:")
            sample = civic_articles[0]
            print(f"Title: {sample.get('title')}")
            print(f"Source: {sample.get('source')}")
            print(f"URL: {sample.get('url')}")
            print(f"Text preview: {sample.get('text', '')[:200]}...")
    else:
        print("No articles found")

if __name__ == "__main__":
    main()