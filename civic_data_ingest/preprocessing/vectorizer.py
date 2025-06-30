"""
Text Vectorizer
Converts text into numerical feature vectors using TF-IDF and other techniques
"""

import numpy as np
import pandas as pd
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import os

class TextVectorizer:
    def __init__(self, 
                 max_features: int = 5000,
                 min_df: int = 2,
                 max_df: float = 0.8,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_idf: bool = True):
        """
        Initialize text vectorizer
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            ngram_range: Range of n-grams to extract
            use_idf: Whether to use IDF weighting
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        
        # Initialize vectorizers
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.dimensionality_reducer = None
        self.scaler = None
        
        # Fitted status
        self.is_fitted = False
        
        # Civic-specific stop words (in addition to default)
        self.civic_stop_words = [
            'rt', 'via', 'http', 'https', 'www', 'com', 'org', 'gov',
            'twitter', 'tweet', 'retweet', 'follow', 'following',
            'please', 'just', 'like', 'get', 'go', 'got', 'going',
            'said', 'say', 'says', 'see', 'seen', 'know', 'new'
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_tfidf_vectorizer(self, stop_words: str = 'english'):
        """Setup TF-IDF vectorizer with configurations"""
        # Combine default stop words with civic-specific ones
        if stop_words == 'english':
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            combined_stop_words = list(ENGLISH_STOP_WORDS) + self.civic_stop_words
        else:
            combined_stop_words = self.civic_stop_words
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words=combined_stop_words,
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w\w+\b',  # Only words with 2+ characters
            use_idf=self.use_idf,
            smooth_idf=True,
            sublinear_tf=True  # Apply sublinear TF scaling
        )
    
    def setup_count_vectorizer(self, stop_words: str = 'english'):
        """Setup Count vectorizer for basic term frequency"""
        if stop_words == 'english':
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            combined_stop_words = list(ENGLISH_STOP_WORDS) + self.civic_stop_words
        else:
            combined_stop_words = self.civic_stop_words
        
        self.count_vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words=combined_stop_words,
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w\w+\b'
        )
    
    def fit_tfidf(self, texts: List[str]) -> 'TextVectorizer':
        """
        Fit TF-IDF vectorizer on texts
        
        Args:
            texts: List of texts to fit on
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list")
        
        # Setup vectorizer if not already done
        if self.tfidf_vectorizer is None:
            self.setup_tfidf_vectorizer()
        
        # Fit the vectorizer
        self.logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts")
        self.tfidf_vectorizer.fit(texts)
        
        # Get vocabulary info
        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        self.logger.info(f"TF-IDF vocabulary size: {vocab_size}")
        
        self.is_fitted = True
        return self
    
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors
        
        Args:
            texts: List of texts to transform
            
        Returns:
            TF-IDF matrix
        """
        if not self.is_fitted or self.tfidf_vectorizer is None:
            raise ValueError("Vectorizer must be fitted before transform")
        
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def fit_transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform texts
        
        Args:
            texts: List of texts
            
        Returns:
            TF-IDF matrix
        """
        self.fit_tfidf(texts)
        return self.transform_tfidf(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from the fitted vectorizer"""
        if not self.is_fitted or self.tfidf_vectorizer is None:
            raise ValueError("Vectorizer must be fitted first")
        
        return self.tfidf_vectorizer.get_feature_names_out().tolist()
    
    def get_top_features_per_document(self, texts: List[str], top_k: int = 10) -> List[Dict]:
        """
        Get top features for each document
        
        Args:
            texts: List of texts
            top_k: Number of top features to return per document
            
        Returns:
            List of dictionaries with top features and scores
        """
        if not texts:
            return []
        
        # Transform texts
        tfidf_matrix = self.transform_tfidf(texts)
        feature_names = self.get_feature_names()
        
        results = []
        for i, text in enumerate(texts):
            # Get TF-IDF scores for this document
            doc_scores = tfidf_matrix[i]
            
            # Get top k features
            top_indices = np.argsort(doc_scores)[-top_k:][::-1]
            
            top_features = []
            for idx in top_indices:
                if doc_scores[idx] > 0:  # Only include non-zero scores
                    top_features.append({
                        'feature': feature_names[idx],
                        'score': float(doc_scores[idx])
                    })
            
            results.append({
                'text_index': i,
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'top_features': top_features
            })
        
        return results
    
    def setup_dimensionality_reduction(self, n_components: int = 100):
        """Setup dimensionality reduction using TruncatedSVD"""
        self.dimensionality_reducer = TruncatedSVD(
            n_components=min(n_components, self.max_features),
            random_state=42
        )
    
    def reduce_dimensions(self, tfidf_matrix: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionality of TF-IDF matrix
        
        Args:
            tfidf_matrix: TF-IDF matrix
            
        Returns:
            Reduced dimensionality matrix
        """
        if self.dimensionality_reducer is None:
            self.setup_dimensionality_reduction()
        
        return self.dimensionality_reducer.fit_transform(tfidf_matrix)
    
    def get_vocabulary_stats(self) -> Dict:
        """Get statistics about the vocabulary"""
        if not self.is_fitted or self.tfidf_vectorizer is None:
            return {}
        
        vocab = self.tfidf_vectorizer.vocabulary_
        feature_names = self.get_feature_names()
        
        # Analyze n-grams
        unigrams = [f for f in feature_names if len(f.split()) == 1]
        bigrams = [f for f in feature_names if len(f.split()) == 2]
        
        return {
            'total_features': len(vocab),
            'unigrams': len(unigrams),
            'bigrams': len(bigrams),
            'max_features_used': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'ngram_range': self.ngram_range
        }
    
    def save_vectorizer(self, filepath: str):
        """Save the fitted vectorizer to file"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        
        vectorizer_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'dimensionality_reducer': self.dimensionality_reducer,
            'scaler': self.scaler,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'ngram_range': self.ngram_range,
            'use_idf': self.use_idf,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vectorizer_data, f)
        
        self.logger.info(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath: str):
        """Load a fitted vectorizer from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            vectorizer_data = pickle.load(f)
        
        # Restore vectorizer state
        self.tfidf_vectorizer = vectorizer_data['tfidf_vectorizer']
        self.count_vectorizer = vectorizer_data['count_vectorizer']
        self.dimensionality_reducer = vectorizer_data['dimensionality_reducer']
        self.scaler = vectorizer_data['scaler']
        self.max_features = vectorizer_data['max_features']
        self.min_df = vectorizer_data['min_df']
        self.max_df = vectorizer_data['max_df']
        self.ngram_range = vectorizer_data['ngram_range']
        self.use_idf = vectorizer_data['use_idf']
        self.is_fitted = vectorizer_data['is_fitted']
        
        self.logger.info(f"Vectorizer loaded from {filepath}")
    
    def create_civic_features(self, texts: List[str]) -> np.ndarray:
        """
        Create civic-specific features
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix with civic-specific features
        """
        # Define civic issue categories and keywords
        civic_categories = {
            'water': ['water', 'supply', 'shortage', 'crisis', 'tap', 'pipeline', 'leak'],
            'electricity': ['power', 'electricity', 'outage', 'blackout', 'supply', 'cut'],
            'roads': ['road', 'pothole', 'traffic', 'signal', 'jam', 'repair', 'maintenance'],
            'waste': ['garbage', 'waste', 'collection', 'dump', 'clean', 'sanitation'],
            'transport': ['bus', 'train', 'metro', 'transport', 'route', 'service'],
            'infrastructure': ['bridge', 'building', 'construction', 'development', 'park']
        }
        
        features = []
        
        for text in texts:
            text_lower = text.lower()
            text_features = []
            
            # Count keywords for each category
            for category, keywords in civic_categories.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                text_features.append(count)
            
            # Additional features
            text_features.extend([
                len(text),  # Text length
                len(text.split()),  # Word count
                text.count('!'),  # Exclamation marks (urgency indicator)
                text.count('?'),  # Question marks
                1 if any(urgent in text_lower for urgent in ['urgent', 'emergency', 'immediate']) else 0
            ])
            
            features.append(text_features)
        
        return np.array(features)

def main():
    """Test the text vectorizer"""
    # Sample civic texts
    sample_texts = [
        "Water crisis in Mumbai affecting thousands of residents daily supply issues",
        "Potholes on Bangalore roads causing major traffic problems need immediate repair",
        "Power outage in Delhi residential areas for third consecutive day",
        "Garbage collection stopped in our locality for past week sanitation issues",
        "Public transport bus service irregular causing problems for daily commuters",
        "Street lights not working in our area safety concerns at night",
        "Water supply pipe burst near the main road causing flooding",
        "Traffic signal malfunction at busy intersection creating chaos"
    ]
    
    print("Testing Text Vectorizer:")
    print("=" * 50)
    
    # Initialize vectorizer
    vectorizer = TextVectorizer(max_features=1000, ngram_range=(1, 2))
    
    # Fit and transform
    print(f"\nFitting vectorizer on {len(sample_texts)} texts...")
    tfidf_matrix = vectorizer.fit_transform_tfidf(sample_texts)
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Get vocabulary stats
    stats = vectorizer.get_vocabulary_stats()
    print(f"Vocabulary stats: {stats}")
    
    # Get top features for first few documents
    print(f"\nTop features per document:")
    top_features = vectorizer.get_top_features_per_document(sample_texts[:3], top_k=5)
    
    for doc_info in top_features:
        print(f"\nText: {doc_info['text_preview']}")
        print("Top features:")
        for feature in doc_info['top_features']:
            print(f"  - {feature['feature']}: {feature['score']:.3f}")
    
    # Test civic-specific features
    print(f"\nCivic-specific features:")
    civic_features = vectorizer.create_civic_features(sample_texts)
    print(f"Civic features shape: {civic_features.shape}")
    print(f"Sample civic features (first text): {civic_features[0]}")
    
    # Test saving and loading
    test_file = "test_vectorizer.pkl"
    try:
        vectorizer.save_vectorizer(test_file)
        print(f"\nVectorizer saved successfully")
        
        # Load and test
        new_vectorizer = TextVectorizer()
        new_vectorizer.load_vectorizer(test_file)
        
        # Test transform with loaded vectorizer
        test_transform = new_vectorizer.transform_tfidf(sample_texts[:2])
        print(f"Transform with loaded vectorizer successful: {test_transform.shape}")
        
        # Clean up
        os.remove(test_file)
        
    except Exception as e:
        print(f"Error testing save/load: {str(e)}")

if __name__ == "__main__":
    main()