"""
Keyword Extractor using KeyBERT
Extracts key terms and phrases from civic-related text data
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re

class KeywordExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize keyword extractor
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.keybert_model = None
        self.sentence_model = None
        self.tfidf_extractor = None
        
        # Civic-specific terms that should be prioritized
        self.civic_domain_terms = [
            'water', 'electricity', 'power', 'road', 'pothole', 'traffic',
            'garbage', 'waste', 'sanitation', 'sewage', 'drainage',
            'municipal', 'corporation', 'government', 'public', 'civic',
            'infrastructure', 'transport', 'bus', 'metro', 'train',
            'hospital', 'school', 'park', 'garden', 'bridge',
            'street', 'light', 'signal', 'maintenance', 'repair',
            'supply', 'shortage', 'outage', 'crisis', 'problem',
            'complaint', 'issue', 'concern', 'urgent', 'emergency'
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize KeyBERT and sentence transformer models"""
        try:
            if KEYBERT_AVAILABLE:
                self.logger.info(f"Initializing KeyBERT with model: {self.model_name}")
                self.sentence_model = SentenceTransformer(self.model_name)
                self.keybert_model = KeyBERT(model=self.sentence_model)
                self.logger.info("KeyBERT initialized successfully")
            else:
                self.logger.warning("KeyBERT not available. Install with: pip install keybert")
        except Exception as e:
            self.logger.error(f"Error initializing KeyBERT: {str(e)}")
    
    def extract_keywords_keybert(self, 
                                text: str,
                                top_k: int = 10,
                                keyphrase_ngram_range: Tuple[int, int] = (1, 2),
                                stop_words: str = 'english',
                                use_maxsum: bool = True,
                                use_mmr: bool = True,
                                diversity: float = 0.5) -> List[Tuple[str, float]]:
        """
        Extract keywords using KeyBERT
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            keyphrase_ngram_range: Range of n-grams for keyphrases
            stop_words: Stop words to ignore
            use_maxsum: Whether to use MaxSum for diversification
            use_mmr: Whether to use MMR for diversification
            diversity: Diversity parameter for MMR (0-1)
            
        Returns:
            List of (keyword, score) tuples
        """
        if not self.keybert_model:
            self.logger.error("KeyBERT model not initialized")
            return []
        
        if not text or len(text.strip()) < 10:
            return []
        
        try:
            # Extract keywords using different methods and combine
            keywords = []
            
            # Basic extraction
            basic_keywords = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words,
                top_k=top_k
            )
            keywords.extend(basic_keywords)
            
            # MaxSum diversification
            if use_maxsum:
                maxsum_keywords = self.keybert_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    use_maxsum=True,
                    nr_candidates=20,
                    top_k=top_k
                )
                keywords.extend(maxsum_keywords)
            
            # MMR diversification
            if use_mmr:
                mmr_keywords = self.keybert_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    use_mmr=True,
                    diversity=diversity,
                    top_k=top_k
                )
                keywords.extend(mmr_keywords)
            
            # Remove duplicates and sort by score
            keyword_dict = {}
            for keyword, score in keywords:
                if keyword in keyword_dict:
                    keyword_dict[keyword] = max(keyword_dict[keyword], score)
                else:
                    keyword_dict[keyword] = score
            
            # Sort by score and return top_k
            sorted_keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)
            return sorted_keywords[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords with KeyBERT: {str(e)}")
            return []
    
    def extract_keywords_tfidf(self, 
                              texts: List[str],
                              top_k: int = 10,
                              max_features: int = 1000) -> List[Dict]:
        """
        Extract keywords using TF-IDF for batch processing
        
        Args:
            texts: List of texts
            top_k: Number of top keywords per text
            max_features: Maximum features for TF-IDF
            
        Returns:
            List of dictionaries with keywords for each text
        """
        if not texts:
            return []
        
        try:
            # Initialize TF-IDF vectorizer
            if self.tfidf_extractor is None:
                self.tfidf_extractor = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    stop_words='english',
                    lowercase=True,
                    strip_accents='unicode'
                )
            
            # Fit and transform
            tfidf_matrix = self.tfidf_extractor.fit_transform(texts)
            feature_names = self.tfidf_extractor.get_feature_names_out()
            
            results = []
            for i, text in enumerate(texts):
                # Get TF-IDF scores for this document
                doc_scores = tfidf_matrix[i].toarray()[0]
                
                # Get top keywords
                top_indices = np.argsort(doc_scores)[-top_k:][::-1]
                
                keywords = []
                for idx in top_indices:
                    if doc_scores[idx] > 0:
                        keywords.append({
                            'keyword': feature_names[idx],
                            'score': float(doc_scores[idx])
                        })
                
                results.append({
                    'text_index': i,
                    'keywords': keywords
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords with TF-IDF: {str(e)}")
            return []
    
    def extract_civic_keywords(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Extract civic-specific keywords with domain knowledge
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text:
            return []
        
        text_lower = text.lower()
        keyword_scores = {}
        
        # Score civic domain terms
        for term in self.civic_domain_terms:
            if term in text_lower:
                # Calculate score based on frequency and position
                frequency = text_lower.count(term)
                position_bonus = 1.2 if text_lower.find(term) < len(text_lower) / 3 else 1.0
                keyword_scores[term] = frequency * position_bonus
        
        # Extract compound civic terms
        civic_patterns = [
            r'water (supply|shortage|crisis|problem)',
            r'power (outage|cut|failure|problem)',
            r'road (repair|maintenance|problem|condition)',
            r'traffic (jam|signal|problem|congestion)',
            r'garbage (collection|disposal|problem)',
            r'street (light|lighting|problem)',
            r'public (transport|service|utility)',
            r'municipal (corporation|service|office)'
        ]
        
        for pattern in civic_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                phrase = match.group()
                keyword_scores[phrase] = keyword_scores.get(phrase, 0) + 2.0
        
        # Sort by score and return top_k
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]
    
    def combine_keyword_methods(self, 
                               text: str,
                               top_k: int = 10,
                               weight_keybert: float = 0.6,
                               weight_civic: float = 0.4) -> List[Tuple[str, float]]:
        """
        Combine KeyBERT and civic-specific keyword extraction
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            weight_keybert: Weight for KeyBERT results
            weight_civic: Weight for civic-specific results
            
        Returns:
            Combined list of (keyword, score) tuples
        """
        all_keywords = {}
        
        # Get KeyBERT keywords
        if self.keybert_model:
            keybert_keywords = self.extract_keywords_keybert(text, top_k=top_k)
            for keyword, score in keybert_keywords:
                all_keywords[keyword] = all_keywords.get(keyword, 0) + (score * weight_keybert)
        
        # Get civic-specific keywords
        civic_keywords = self.extract_civic_keywords(text, top_k=top_k)
        for keyword, score in civic_keywords:
            # Normalize civic scores to 0-1 range
            norm_score = min(score / 5.0, 1.0)
            all_keywords[keyword] = all_keywords.get(keyword, 0) + (norm_score * weight_civic)
        
        # Sort and return top_k
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]
    
    def extract_keywords_batch(self, 
                              texts: List[str],
                              method: str = 'combined',
                              top_k: int = 10) -> List[Dict]:
        """
        Extract keywords from a batch of texts
        
        Args:
            texts: List of texts
            method: 'keybert', 'civic', 'tfidf', or 'combined'
            top_k: Number of top keywords per text
            
        Returns:
            List of dictionaries with keywords for each text
        """
        results = []
        
        for i, text in enumerate(texts):
            if method == 'keybert':
                keywords = self.extract_keywords_keybert(text, top_k=top_k)
            elif method == 'civic':
                keywords = self.extract_civic_keywords(text, top_k=top_k)
            elif method == 'combined':
                keywords = self.combine_keyword_methods(text, top_k=top_k)
            elif method == 'tfidf':
                # For TF-IDF, we need to process all texts together
                tfidf_results = self.extract_keywords_tfidf(texts, top_k=top_k)
                return tfidf_results
            else:
                keywords = []
            
            results.append({
                'text_index': i,
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'keywords': [{'keyword': k, 'score': s} for k, s in keywords],
                'method': method
            })
        
        return results
    
    def get_keyword_categories(self, keywords: List[str]) -> Dict[str, List[str]]:
        """
        Categorize keywords into civic issue types
        
        Args:
            keywords: List of keywords
            
        Returns:
            Dictionary with categories and their keywords
        """
        categories = {
            'water_issues': [],
            'power_issues': [],
            'road_transport': [],
            'waste_sanitation': [],
            'infrastructure': [],
            'general_civic': []
        }
        
        category_terms = {
            'water_issues': ['water', 'supply', 'shortage', 'leak', 'pipeline', 'tap'],
            'power_issues': ['power', 'electricity', 'outage', 'blackout', 'supply'],
            'road_transport': ['road', 'pothole', 'traffic', 'transport', 'bus', 'signal'],
            'waste_sanitation': ['garbage', 'waste', 'sanitation', 'sewage', 'clean'],
            'infrastructure': ['bridge', 'building', 'park', 'street', 'light'],
            'general_civic': ['municipal', 'government', 'public', 'service', 'complaint']
        }
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            categorized = False
            
            for category, terms in category_terms.items():
                if any(term in keyword_lower for term in terms):
                    categories[category].append(keyword)
                    categorized = True
                    break
            
            if not categorized:
                categories['general_civic'].append(keyword)
        
        return categories

def main():
    """Test the keyword extractor"""
    # Sample civic texts
    sample_texts = [
        "Water supply shortage in Mumbai affecting residential areas. Municipal corporation needs to address this crisis immediately.",
        "Potholes on Bangalore roads causing major traffic problems. Road repair work needed urgently on main streets.",
        "Power outage in Delhi for third consecutive day. Electricity supply disruption affecting thousands of households.",
        "Garbage collection stopped in our locality. Waste management and sanitation issues need immediate attention from municipal authorities.",
        "Public transport bus service irregular. Commuters facing problems due to poor service and lack of proper maintenance."
    ]
    
    print("Testing Keyword Extractor:")
    print("=" * 60)
    
    # Initialize extractor
    extractor = KeywordExtractor()
    
    if not KEYBERT_AVAILABLE:
        print("Note: KeyBERT not available. Testing with TF-IDF and civic methods only.")
    
    # Test different extraction methods
    methods = ['civic', 'tfidf']
    if KEYBERT_AVAILABLE:
        methods.extend(['keybert', 'combined'])
    
    for method in methods:
        print(f"\nTesting {method.upper()} method:")
        print("-" * 40)
        
        try:
            results = extractor.extract_keywords_batch(sample_texts[:3], method=method, top_k=5)
            
            for result in results:
                print(f"\nText: {result['text_preview']}")
                print("Keywords:")
                for kw in result['keywords']:
                    if isinstance(kw, dict):
                        print(f"  - {kw['keyword']}: {kw['score']:.3f}")
                    else:
                        print(f"  - {kw}")
        
        except Exception as e:
            print(f"Error testing {method}: {str(e)}")
    
    # Test keyword categorization
    print(f"\n\nKeyword Categorization Test:")
    print("-" * 40)
    
    sample_keywords = [
        'water supply', 'power outage', 'road repair', 'garbage collection',
        'traffic signal', 'municipal corporation', 'street light', 'public transport'
    ]
    
    categories = extractor.get_keyword_categories(sample_keywords)
    for category, keywords in categories.items():
        if keywords:
            print(f"{category}: {keywords}")

if __name__ == "__main__":
    main()