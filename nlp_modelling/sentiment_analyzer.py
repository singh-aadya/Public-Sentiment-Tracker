"""
Sentiment Analyzer using VADER and HuggingFace transformers
Analyzes sentiment of civic-related text data
"""

import logging
from typing import List, Dict, Optional, Union
import numpy as np

# VADER Sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# HuggingFace Transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import re
from collections import Counter

class SentimentAnalyzer:
    def __init__(self, use_transformer: bool = False, transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize sentiment analyzer
        
        Args:
            use_transformer: Whether to use transformer model in addition to VADER
            transformer_model: HuggingFace model name for transformer-based sentiment
        """
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.transformer_model_name = transformer_model
        
        # Initialize VADER
        self.vader_analyzer = None
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformer pipeline
        self.transformer_pipeline = None
        if self.use_transformer:
            try:
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model=transformer_model,
                    tokenizer=transformer_model
                )
            except Exception as e:
                logging.warning(f"Could not load transformer model: {e}")
                self.use_transformer = False
        
        # Civic-specific sentiment modifiers
        self.civic_negative_intensifiers = [
            'crisis', 'emergency', 'urgent', 'terrible', 'awful', 'disaster',
            'failure', 'broken', 'worst', 'horrible', 'nightmare', 'chaos'
        ]
        
        self.civic_positive_intensifiers = [
            'excellent', 'improved', 'better', 'fixed', 'resolved', 'working',
            'good', 'great', 'satisfied', 'happy', 'pleased', 'thankful'
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_vader_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.vader_analyzer or not text:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            self.logger.error(f"Error in VADER sentiment analysis: {str(e)}")
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
    
    def analyze_transformer_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using transformer model
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment prediction
        """
        if not self.transformer_pipeline or not text:
            return {
                'label': 'NEUTRAL',
                'score': 0.0
            }
        
        try:
            # Truncate text if too long (most models have token limits)
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.transformer_pipeline(text)[0]
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            self.logger.error(f"Error in transformer sentiment analysis: {str(e)}")
            return {
                'label': 'NEUTRAL',
                'score': 0.0
            }
    
    def apply_civic_sentiment_adjustment(self, text: str, vader_scores: Dict) -> Dict:
        """
        Apply civic-specific sentiment adjustments
        
        Args:
            text: Input text
            vader_scores: Original VADER scores
            
        Returns:
            Adjusted sentiment scores
        """
        text_lower = text.lower()
        adjustment = 0.0
        
        # Check for civic negative intensifiers
        negative_count = sum(1 for word in self.civic_negative_intensifiers if word in text_lower)
        if negative_count > 0:
            adjustment -= 0.1 * negative_count
        
        # Check for civic positive intensifiers
        positive_count = sum(1 for word in self.civic_positive_intensifiers if word in text_lower)
        if positive_count > 0:
            adjustment += 0.1 * positive_count
        
        # Apply adjustment to compound score
        adjusted_compound = max(-1.0, min(1.0, vader_scores['compound'] + adjustment))
        
        # Recalculate other scores based on adjusted compound
        if adjusted_compound > 0.05:
            sentiment_label = 'positive'
        elif adjusted_compound < -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'compound': adjusted_compound,
            'positive': max(0, vader_scores['positive'] + (adjustment if adjustment > 0 else 0)),
            'negative': max(0, vader_scores['negative'] + (abs(adjustment) if adjustment < 0 else 0)),
            'neutral': max(0, 1.0 - abs(adjusted_compound)),
            'sentiment_label': sentiment_label,
            'civic_adjustment': adjustment
        }
    
    def analyze_text_sentiment(self, text: str, apply_civic_adjustment: bool = True) -> Dict:
        """
        Comprehensive sentiment analysis of text
        
        Args:
            text: Input text
            apply_civic_adjustment: Whether to apply civic-specific adjustments
            
        Returns:
            Dictionary with complete sentiment analysis
        """
        if not text or len(text.strip()) == 0:
            return {
                'text_length': 0,
                'vader': {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
                'transformer': {'label': 'NEUTRAL', 'score': 0.0},
                'final_sentiment': 'neutral',
                'confidence': 0.0
            }
        
        result = {
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        # VADER analysis
        vader_scores = self.analyze_vader_sentiment(text)
        result['vader'] = vader_scores
        
        # Apply civic adjustments if requested
        if apply_civic_adjustment:
            adjusted_scores = self.apply_civic_sentiment_adjustment(text, vader_scores)
            result['vader_adjusted'] = adjusted_scores
            primary_scores = adjusted_scores
        else:
            primary_scores = vader_scores
        
        # Transformer analysis (if available)
        if self.use_transformer:
            transformer_result = self.analyze_transformer_sentiment(text)
            result['transformer'] = transformer_result
        
        # Determine final sentiment
        compound_score = primary_scores['compound']
        
        if compound_score >= 0.05:
            final_sentiment = 'positive'
            confidence = compound_score
        elif compound_score <= -0.05:
            final_sentiment = 'negative'
            confidence = abs(compound_score)
        else:
            final_sentiment = 'neutral'
            confidence = 1.0 - abs(compound_score)
        
        result['final_sentiment'] = final_sentiment
        result['confidence'] = confidence
        
        return result
    
    def analyze_batch_sentiment(self, texts: List[str], apply_civic_adjustment: bool = True) -> List[Dict]:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of texts to analyze
            apply_civic_adjustment: Whether to apply civic-specific adjustments
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for i, text in enumerate(texts):
            sentiment_result = self.analyze_text_sentiment(text, apply_civic_adjustment)
            sentiment_result['text_index'] = i
            sentiment_result['text_preview'] = text[:100] + '...' if len(text) > 100 else text
            results.append(sentiment_result)
        
        return results
    
    def get_sentiment_summary(self, sentiment_results: List[Dict]) -> Dict:
        """
        Get summary statistics of sentiment analysis results
        
        Args:
            sentiment_results: List of sentiment analysis results
            
        Returns:
            Summary statistics
        """
        if not sentiment_results:
            return {}
        
        sentiments = [result['final_sentiment'] for result in sentiment_results]
        compound_scores = [result['vader']['compound'] for result in sentiment_results]
        confidences = [result['confidence'] for result in sentiment_results]
        
        sentiment_counts = Counter(sentiments)
        
        return {
            'total_texts': len(sentiment_results),
            'sentiment_distribution': dict(sentiment_counts),
            'sentiment_percentages': {
                sentiment: (count / len(sentiments)) * 100
                for sentiment, count in sentiment_counts.items()
            },
            'average_compound_score': np.mean(compound_scores),
            'compound_score_std': np.std(compound_scores),
            'average_confidence': np.mean(confidences),
            'most_positive': max(compound_scores),
            'most_negative': min(compound_scores)
        }
    
    def classify_urgency_by_sentiment(self, sentiment_result: Dict) -> str:
        """
        Classify urgency level based on sentiment analysis
        
        Args:
            sentiment_result: Result from sentiment analysis
            
        Returns:
            Urgency level: 'low', 'medium', or 'high'
        """
        compound_score = sentiment_result['vader']['compound']
        sentiment = sentiment_result['final_sentiment']
        confidence = sentiment_result['confidence']
        
        # High urgency for very negative sentiment with high confidence
        if sentiment == 'negative' and compound_score <= -0.6 and confidence >= 0.6:
            return 'high'
        
        # Medium urgency for moderately negative sentiment
        elif sentiment == 'negative' and compound_score <= -0.3:
            return 'medium'
        
        # Low urgency for neutral or positive sentiment
        else:
            return 'low'
    
    def extract_emotional_indicators(self, text: str) -> Dict:
        """
        Extract emotional indicators from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with emotional indicators
        """
        text_lower = text.lower()
        
        # Punctuation indicators
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        # Emotional words
        anger_words = ['angry', 'furious', 'mad', 'annoyed', 'frustrated', 'outraged']
        fear_words = ['afraid', 'scared', 'worried', 'concerned', 'anxious', 'panic']
        sadness_words = ['sad', 'disappointed', 'upset', 'depressed', 'hopeless']
        joy_words = ['happy', 'pleased', 'satisfied', 'glad', 'delighted', 'grateful']
        
        anger_count = sum(1 for word in anger_words if word in text_lower)
        fear_count = sum(1 for word in fear_words if word in text_lower)
        sadness_count = sum(1 for word in sadness_words if word in text_lower)
        joy_count = sum(1 for word in joy_words if word in text_lower)
        
        return {
            'exclamation_marks': exclamation_count,
            'question_marks': question_count,
            'caps_words': caps_words,
            'emotional_intensity': exclamation_count + caps_words * 0.5,
            'emotion_counts': {
                'anger': anger_count,
                'fear': fear_count,
                'sadness': sadness_count,
                'joy': joy_count
            },
            'dominant_emotion': max(
                [('anger', anger_count), ('fear', fear_count), 
                 ('sadness', sadness_count), ('joy', joy_count)],
                key=lambda x: x[1]
            )[0] if max(anger_count, fear_count, sadness_count, joy_count) > 0 else 'neutral'
        }

def main():
    """Test the sentiment analyzer"""
    # Sample civic texts with different sentiments
    sample_texts = [
        "Water supply is excellent in our area! Municipal corporation is doing great work.",
        "This is terrible! No electricity for 3 days. Complete failure of the power department!",
        "Road conditions are okay. Some potholes need to be fixed but overall manageable.",
        "URGENT! Sewage overflow causing health emergency in our locality. Immediate action needed!",
        "Happy to report that garbage collection has improved significantly in recent weeks.",
        "Traffic signal not working at main intersection. Creating chaos and safety concerns.",
        "Thank you to the municipal team for quick response to our water supply complaint.",
        "Worst road conditions ever! Potholes everywhere making driving impossible!!!"
    ]
    
    print("Testing Sentiment Analyzer:")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(use_transformer=False)  # Set to True if transformers available
    
    if not VADER_AVAILABLE:
        print("VADER not available. Please install: pip install vaderSentiment")
        return
    
    # Analyze individual texts
    print("\nIndividual Sentiment Analysis:")
    print("-" * 40)
    
    for i, text in enumerate(sample_texts[:4], 1):
        print(f"\n{i}. Text: {text}")
        
        result = analyzer.analyze_text_sentiment(text, apply_civic_adjustment=True)
        
        print(f"   Final Sentiment: {result['final_sentiment'].upper()}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   VADER Compound: {result['vader']['compound']:.3f}")
        
        if 'vader_adjusted' in result:
            print(f"   Adjusted Compound: {result['vader_adjusted']['compound']:.3f}")
            print(f"   Civic Adjustment: {result['vader_adjusted']['civic_adjustment']:.3f}")
        
        # Classify urgency
        urgency = analyzer.classify_urgency_by_sentiment(result)
        print(f"   Urgency Level: {urgency.upper()}")
        
        # Extract emotional indicators
        emotions = analyzer.extract_emotional_indicators(text)
        print(f"   Emotional Intensity: {emotions['emotional_intensity']:.1f}")
        print(f"   Dominant Emotion: {emotions['dominant_emotion']}")
    
    # Batch analysis
    print(f"\n\nBatch Sentiment Analysis:")
    print("-" * 40)
    
    batch_results = analyzer.analyze_batch_sentiment(sample_texts, apply_civic_adjustment=True)
    
    # Get summary
    summary = analyzer.get_sentiment_summary(batch_results)
    
    print(f"Total texts analyzed: {summary['total_texts']}")
    print(f"Sentiment distribution: {summary['sentiment_distribution']}")
    print(f"Sentiment percentages:")
    for sentiment, percentage in summary['sentiment_percentages'].items():
        print(f"  - {sentiment}: {percentage:.1f}%")
    
    print(f"Average compound score: {summary['average_compound_score']:.3f}")
    print(f"Most positive score: {summary['most_positive']:.3f}")
    print(f"Most negative score: {summary['most_negative']:.3f}")
    
    # Urgency classification summary
    urgency_levels = [analyzer.classify_urgency_by_sentiment(result) for result in batch_results]
    urgency_counts = Counter(urgency_levels)
    
    print(f"\nUrgency Level Distribution:")
    for level, count in urgency_counts.items():
        print(f"  - {level}: {count} texts")

if __name__ == "__main__":
    main()