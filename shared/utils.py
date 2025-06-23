"""
Shared utilities for the civic data ingest system
Common helpers for logging, configuration, and data processing
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

def load_config(config_file: str = ".env") -> Dict[str, str]:
    """
    Load configuration from environment file
    
    Args:
        config_file: Path to environment file
        
    Returns:
        Dictionary with configuration
    """
    config = {}
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    
    # Override with environment variables
    for key in config:
        config[key] = os.getenv(key, config[key])
    
    return config

def save_json(data: Any, filepath: str, indent: int = 2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

def load_json(filepath: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_timestamp() -> str:
    """Create ISO timestamp string"""
    return datetime.now(timezone.utc).isoformat()

def create_unique_id() -> str:
    """Create unique identifier"""
    return str(uuid.uuid4())

def ensure_directory(directory: str):
    """
    Ensure directory exists, create if not
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def calculate_text_stats(text: str) -> Dict[str, Any]:
    """
    Calculate basic text statistics
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0
        }
    
    words = text.split()
    sentences = text.split('.') + text.split('!') + text.split('?')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
    }

def validate_data_record(record: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate data record has required fields
    
    Args:
        record: Data record to validate
        required_fields: List of required field names
        
    Returns:
        True if valid, False otherwise
    """
    return all(field in record and record[field] is not None for field in required_fields)

def normalize_text_for_comparison(text: str) -> str:
    """
    Normalize text for comparison/deduplication
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()

def detect_duplicates(texts: List[str], similarity_threshold: float = 0.8) -> List[List[int]]:
    """
    Detect duplicate or highly similar texts
    
    Args:
        texts: List of texts to compare
        similarity_threshold: Similarity threshold (0-1)
        
    Returns:
        List of groups containing indices of similar texts
    """
    from difflib import SequenceMatcher
    
    duplicate_groups = []
    processed = set()
    
    for i, text1 in enumerate(texts):
        if i in processed:
            continue
        
        current_group = [i]
        processed.add(i)
        
        for j, text2 in enumerate(texts[i+1:], i+1):
            if j in processed:
                continue
            
            # Calculate similarity
            similarity = SequenceMatcher(None, 
                                       normalize_text_for_comparison(text1),
                                       normalize_text_for_comparison(text2)).ratio()
            
            if similarity >= similarity_threshold:
                current_group.append(j)
                processed.add(j)
        
        if len(current_group) > 1:
            duplicate_groups.append(current_group)
    
    return duplicate_groups

def create_data_pipeline_status() -> Dict[str, Any]:
    """
    Create initial data pipeline status tracking
    
    Returns:
        Status dictionary
    """
    return {
        'pipeline_id': create_unique_id(),
        'created_at': create_timestamp(),
        'status': 'initialized',
        'stages': {
            'data_collection': {'status': 'pending', 'records': 0},
            'preprocessing': {'status': 'pending', 'records': 0},
            'sentiment_analysis': {'status': 'pending', 'records': 0},
            'keyword_extraction': {'status': 'pending', 'records': 0},
            'classification': {'status': 'pending', 'records': 0},
            'geo_extraction': {'status': 'pending', 'records': 0}
        },
        'errors': [],
        'metrics': {}
    }

def update_pipeline_status(status: Dict[str, Any], 
                          stage: str, 
                          stage_status: str, 
                          records: int = 0,
                          error: str = None) -> Dict[str, Any]:
    """
    Update pipeline status
    
    Args:
        status: Current status dictionary
        stage: Stage name
        stage_status: New stage status
        records: Number of records processed
        error: Error message if any
        
    Returns:
        Updated status dictionary
    """
    status['stages'][stage]['status'] = stage_status
    status['stages'][stage]['records'] = records
    status['last_updated'] = create_timestamp()
    
    if error:
        status['errors'].append({
            'stage': stage,
            'error': error,
            'timestamp': create_timestamp()
        })
    
    # Update overall status
    all_completed = all(
        s['status'] in ['completed', 'skipped'] 
        for s in status['stages'].values()
    )
    
    any_failed = any(
        s['status'] == 'failed' 
        for s in status['stages'].values()
    )
    
    if any_failed:
        status['status'] = 'failed'
    elif all_completed:
        status['status'] = 'completed'
    else:
        status['status'] = 'running'
    
    return status

class DataProcessor:
    """Base class for data processing components"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.config = load_config()
    
    def process(self, data: Any) -> Any:
        """Override this method in subclasses"""
        raise NotImplementedError
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(f"[{self.name}] {message}")
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"[{self.name}] {message}")

def main():
    """Test utility functions"""
    print("Testing Civic Data Ingest Utilities:")
    print("=" * 50)
    
    # Test logging setup
    setup_logging("INFO")
    logger = logging.getLogger("test")
    logger.info("Logging setup successful")
    
    # Test timestamp and ID generation
    print(f"Current timestamp: {create_timestamp()}")
    print(f"Unique ID: {create_unique_id()}")
    
    # Test text statistics
    sample_text = "Water crisis in Mumbai! This is a serious problem affecting thousands of residents."
    stats = calculate_text_stats(sample_text)
    print(f"Text stats: {stats}")
    
    # Test data validation
    test_record = {'text': 'sample text', 'source': 'twitter', 'timestamp': create_timestamp()}
    required_fields = ['text', 'source', 'timestamp']
    is_valid = validate_data_record(test_record, required_fields)
    print(f"Data validation result: {is_valid}")
    
    # Test duplicate detection
    sample_texts = [
        "Water supply problem in Mumbai",
        "Water supply issue in Mumbai", 
        "Power outage in Delhi",
        "Electricity cut in Delhi"
    ]
    
    duplicates = detect_duplicates(sample_texts, similarity_threshold=0.7)
    print(f"Duplicate groups: {duplicates}")
    
    # Test pipeline status
    status = create_data_pipeline_status()
    print(f"Initial pipeline status: {status['status']}")
    
    # Update status
    status = update_pipeline_status(status, 'data_collection', 'completed', records=100)
    print(f"Updated pipeline status: {status['status']}")

if __name__ == "__main__":
    main()