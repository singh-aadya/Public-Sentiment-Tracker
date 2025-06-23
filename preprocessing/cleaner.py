"""
Text Cleaner
Removes emojis, URLs, mentions, hashtags, and other unwanted characters from text
"""

import re
import string
import logging
from typing import List, Dict, Optional
import unicodedata

class TextCleaner:
    def __init__(self):
        """Initialize text cleaner with patterns and configurations"""
        # Emoji pattern
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+", flags=re.UNICODE
        )
        
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Mention pattern (@username)
        self.mention_pattern = re.compile(r'@\w+')
        
        # Hashtag pattern (#hashtag)
        self.hashtag_pattern = re.compile(r'#\w+')
        
        # Multiple whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Special characters to remove (keeping basic punctuation)
        self.special_chars_pattern = re.compile(r'[^\w\s\.\,\!\?\;\:\'\"\(\)\[\]\-]')
        
        # Repeated characters pattern (e.g., "sooooo" -> "so")
        self.repeated_chars_pattern = re.compile(r'(.)\1{2,}')
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text"""
        return self.emoji_pattern.sub('', text)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        return self.url_pattern.sub('', text)
    
    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from text"""
        return self.mention_pattern.sub('', text)
    
    def remove_hashtags(self, text: str, keep_text: bool = True) -> str:
        """
        Remove hashtags from text
        
        Args:
            text: Input text
            keep_text: If True, keep the text part of hashtag (remove only #)
        """
        if keep_text:
            return re.sub(r'#(\w+)', r'\1', text)
        else:
            return self.hashtag_pattern.sub('', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize multiple whitespaces to single space"""
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def remove_special_characters(self, text: str) -> str:
        """Remove special characters except basic punctuation"""
        return self.special_chars_pattern.sub('', text)
    
    def fix_repeated_characters(self, text: str) -> str:
        """Fix repeated characters (e.g., 'sooooo' -> 'so')"""
        return self.repeated_chars_pattern.sub(r'\1\1', text)
    
    def remove_accents(self, text: str) -> str:
        """Remove accents while preserving the base characters"""
        return ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn'
        )
    
    def clean_twitter_text(self, text: str, 
                          remove_hashtags: bool = False,
                          keep_hashtag_text: bool = True) -> str:
        """
        Clean Twitter text with common preprocessing steps
        
        Args:
            text: Tweet text
            remove_hashtags: Whether to remove hashtags
            keep_hashtag_text: If removing hashtags, whether to keep the text part
        """
        if not text:
            return ""
        
        # Apply cleaning steps in order
        cleaned = text
        
        # Remove URLs
        cleaned = self.remove_urls(cleaned)
        
        # Remove mentions
        cleaned = self.remove_mentions(cleaned)
        
        # Handle hashtags
        if remove_hashtags:
            cleaned = self.remove_hashtags(cleaned, keep_text=keep_hashtag_text)
        
        # Remove emojis
        cleaned = self.remove_emojis(cleaned)
        
        # Fix repeated characters
        cleaned = self.fix_repeated_characters(cleaned)
        
        # Remove special characters
        cleaned = self.remove_special_characters(cleaned)
        
        # Normalize whitespace
        cleaned = self.normalize_whitespace(cleaned)
        
        return cleaned
    
    def clean_news_text(self, text: str) -> str:
        """
        Clean news article text
        
        Args:
            text: News article text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Remove URLs
        cleaned = self.remove_urls(cleaned)
        
        # Remove excessive whitespace and line breaks
        cleaned = re.sub(r'\n+', ' ', cleaned)
        cleaned = self.normalize_whitespace(cleaned)
        
        # Remove special characters (more lenient for news)
        cleaned = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\(\)\[\]\-\—\–]', '', cleaned)
        
        # Fix repeated characters (less aggressive for news)
        cleaned = re.sub(r'(.)\1{3,}', r'\1\1', cleaned)
        
        return cleaned
    
    def basic_clean(self, text: str) -> str:
        """
        Basic text cleaning for general use
        
        Args:
            text: Input text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Convert to lowercase
        cleaned = cleaned.lower()
        
        # Remove URLs
        cleaned = self.remove_urls(cleaned)
        
        # Remove emojis
        cleaned = self.remove_emojis(cleaned)
        
        # Remove special characters
        cleaned = self.remove_special_characters(cleaned)
        
        # Normalize whitespace
        cleaned = self.normalize_whitespace(cleaned)
        
        return cleaned
    
    def clean_batch(self, texts: List[str], 
                   source_type: str = 'mixed',
                   **kwargs) -> List[str]:
        """
        Clean a batch of texts
        
        Args:
            texts: List of texts to clean
            source_type: 'twitter', 'news', or 'mixed'
            **kwargs: Additional arguments for specific cleaning methods
        """
        cleaned_texts = []
        
        for text in texts:
            if source_type == 'twitter':
                cleaned = self.clean_twitter_text(text, **kwargs)
            elif source_type == 'news':
                cleaned = self.clean_news_text(text)
            else:
                cleaned = self.basic_clean(text)
            
            cleaned_texts.append(cleaned)
        
        return cleaned_texts
    
    def get_cleaning_stats(self, original: str, cleaned: str) -> Dict:
        """
        Get statistics about the cleaning process
        
        Args:
            original: Original text
            cleaned: Cleaned text
        """
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'chars_removed': len(original) - len(cleaned),
            'removal_ratio': (len(original) - len(cleaned)) / len(original) if len(original) > 0 else 0,
            'original_words': len(original.split()),
            'cleaned_words': len(cleaned.split()),
            'words_removed': len(original.split()) - len(cleaned.split())
        }

def main():
    """Test the text cleaner"""
    cleaner = TextCleaner()
    
    # Test texts
    test_texts = [
        "Water crisis in Mumbai! 😰 Check this link: https://example.com @MumbaiMunicipal #WaterCrisis #Mumbai",
        "Potholessss everywhere in Bangalore 😡😡😡 Why can't @BBMP fix them??? #PotholeProblems #BangaloreTraffic",
        "Power outage in my area for 3rd time this week!!! When will @PowerGrid fix this??? #PowerCut #Frustrated",
        "News Article: The municipal corporation announced new measures to address the water supply issues in the eastern districts..."
    ]
    
    print("Testing Text Cleaner:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Original: {text}")
        
        # Clean as Twitter text
        twitter_cleaned = cleaner.clean_twitter_text(text, remove_hashtags=True)
        print(f"Twitter cleaned: {twitter_cleaned}")
        
        # Clean as news text
        news_cleaned = cleaner.clean_news_text(text)
        print(f"News cleaned: {news_cleaned}")
        
        # Basic clean
        basic_cleaned = cleaner.basic_clean(text)
        print(f"Basic cleaned: {basic_cleaned}")
        
        # Get stats
        stats = cleaner.get_cleaning_stats(text, twitter_cleaned)
        print(f"Stats: {stats['chars_removed']} chars removed ({stats['removal_ratio']:.2%})")
    
    # Test batch cleaning
    print(f"\n\nBatch cleaning test:")
    batch_results = cleaner.clean_batch(test_texts, source_type='twitter', remove_hashtags=True)
    for i, result in enumerate(batch_results, 1):
        print(f"{i}. {result}")

if __name__ == "__main__":
    main()