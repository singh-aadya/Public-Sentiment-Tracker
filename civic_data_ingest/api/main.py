"""
FastAPI Backend for Civic Data Ingest System
Real-time API connecting processing pipeline with frontend dashboard
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os
import json
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import threading
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.cleaner import TextCleaner
from nlp_modeling.keyword_extractor import KeywordExtractor
from nlp_modeling.sentiment_analyzer import SentimentAnalyzer
from data_ingestion.geo_extractor import GeoExtractor
from shared.utils import create_timestamp

# Initialize FastAPI app
app = FastAPI(
    title="Civic Data Ingest API",
    description="Real-time backend API for civic data processing and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processing components
text_cleaner = TextCleaner()
keyword_extractor = KeywordExtractor()
sentiment_analyzer = SentimentAnalyzer()
geo_extractor = GeoExtractor()

# In-memory data store (use Redis/Database in production)
live_data_store = []
processing_status = {"status": "idle", "message": "System ready", "timestamp": create_timestamp()}

# Pydantic models
class ProcessTextRequest(BaseModel):
    text: str
    source: str = "manual"

class BulkProcessRequest(BaseModel):
    texts: List[str]
    source: str = "bulk"

class SimulateDataRequest(BaseModel):
    count: int = 10
    interval_seconds: int = 5

# Sample civic data templates for simulation
CIVIC_TEMPLATES = [
    {
        'text': 'Water supply disrupted in {location} for {days} days. Residents facing severe shortage! #WaterCrisis',
        'location': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'],
        'sentiment_bias': 'negative',
        'urgency_bias': 'high'
    },
    {
        'text': 'Potholes everywhere in {location}! Road conditions terrible, causing traffic problems. #RoadRepair',
        'location': ['Pune', 'Hyderabad', 'Ahmedabad', 'Jaipur'],
        'sentiment_bias': 'negative',
        'urgency_bias': 'medium'
    },
    {
        'text': 'Power outage in {location} for {hours} hours. When will electricity board take action? #PowerCut',
        'location': ['Mumbai', 'Delhi', 'Bangalore'],
        'sentiment_bias': 'negative',
        'urgency_bias': 'medium'
    },
    {
        'text': 'Great improvement in garbage collection in {location}! Thanks to municipal team. #CleanCity',
        'location': ['Mumbai', 'Pune', 'Bangalore'],
        'sentiment_bias': 'positive',
        'urgency_bias': 'low'
    },
    {
        'text': 'Traffic signals not working at main junction in {location}. Safety concerns! #TrafficIssue',
        'location': ['Delhi', 'Bangalore', 'Chennai'],
        'sentiment_bias': 'negative',
        'urgency_bias': 'medium'
    }
]

def process_single_text(text: str, source: str = "api") -> Dict:
    """Process a single text through the complete pipeline"""
    
    try:
        # Clean text
        if source == 'twitter':
            cleaned_text = text_cleaner.clean_twitter_text(text, remove_hashtags=True)
        else:
            cleaned_text = text_cleaner.clean_news_text(text)
        
        # Extract keywords
        keywords = keyword_extractor.extract_civic_keywords(cleaned_text, top_k=5)
        
        # Analyze sentiment
        sentiment_result = sentiment_analyzer.analyze_text_sentiment(
            cleaned_text, apply_civic_adjustment=True
        )
        
        # Extract locations
        locations = geo_extractor.extract_locations_from_text(text)
        
        # Classify urgency
        urgency = sentiment_analyzer.classify_urgency_by_sentiment(sentiment_result)
        
        # Create processed record
        processed_record = {
            "id": len(live_data_store) + 1,
            "original_text": text,
            "cleaned_text": cleaned_text,
            "source": source,
            "keywords": [{"keyword": k, "score": float(s)} for k, s in keywords],
            "sentiment": sentiment_result['final_sentiment'],
            "sentiment_confidence": float(sentiment_result['confidence']),
            "vader_compound": float(sentiment_result['vader']['compound']),
            "urgency": urgency,
            "extracted_locations": locations if locations else ['Unknown'],
            "timestamp": create_timestamp(),
            "processed_at": datetime.now().isoformat()
        }
        
        return processed_record
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Civic Data Ingest API - Live System",
        "version": "1.0.0",
        "status": "running",
        "total_records": len(live_data_store),
        "timestamp": create_timestamp()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components_ready": True,
        "data_store_size": len(live_data_store),
        "timestamp": create_timestamp()
    }

@app.post("/process-text")
async def process_text(request: ProcessTextRequest):
    """Process a single text and add to live data"""
    
    try:
        processed = process_single_text(request.text, request.source)
        
        if processed:
            live_data_store.append(processed)
            return {
                "message": "Text processed successfully",
                "record_id": processed["id"],
                "result": processed
            }
        else:
            raise HTTPException(status_code=500, detail="Processing failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-bulk")
async def process_bulk_text(request: BulkProcessRequest):
    """Process multiple texts in bulk"""
    
    try:
        results = []
        for text in request.texts:
            processed = process_single_text(text, request.source)
            if processed:
                live_data_store.append(processed)
                results.append(processed)
        
        return {
            "message": f"Processed {len(results)} texts successfully",
            "records_added": len(results),
            "total_records": len(live_data_store)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/live")
async def get_live_data(limit: int = 50, skip: int = 0):
    """Get live processed data with pagination"""
    
    # Sort by timestamp (most recent first)
    sorted_data = sorted(live_data_store, key=lambda x: x['timestamp'], reverse=True)
    
    total_records = len(sorted_data)
    data_slice = sorted_data[skip:skip + limit]
    
    return {
        "data": data_slice,
        "total_records": total_records,
        "returned_records": len(data_slice),
        "skip": skip,
        "limit": limit,
        "last_updated": create_timestamp()
    }

@app.get("/data/stats")
async def get_live_statistics():
    """Get real-time statistics of processed data"""
    
    if not live_data_store:
        return {
            "message": "No data available",
            "total_records": 0
        }
    
    # Calculate statistics
    total_records = len(live_data_store)
    
    # Sentiment distribution
    sentiment_counts = {}
    urgency_counts = {}
    location_counts = {}
    source_counts = {}
    
    # Recent data (last 24 hours)
    recent_threshold = datetime.now() - timedelta(hours=24)
    recent_records = 0
    
    for record in live_data_store:
        # Sentiment
        sentiment = record.get('sentiment', 'neutral')
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Urgency
        urgency = record.get('urgency', 'low')
        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
        
        # Locations
        locations = record.get('extracted_locations', ['Unknown'])
        for location in locations:
            location_counts[location] = location_counts.get(location, 0) + 1
        
        # Source
        source = record.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
        
        # Recent records
        try:
            record_time = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
            if record_time > recent_threshold:
                recent_records += 1
        except:
            pass
    
    # Calculate averages
    avg_sentiment_score = sum(record.get('vader_compound', 0) for record in live_data_store) / total_records
    
    return {
        "total_records": total_records,
        "recent_records_24h": recent_records,
        "sentiment_distribution": sentiment_counts,
        "urgency_distribution": urgency_counts,
        "top_locations": dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "source_distribution": source_counts,
        "average_sentiment_score": round(avg_sentiment_score, 3),
        "last_updated": create_timestamp()
    }

@app.post("/simulate-data")
async def start_data_simulation(request: SimulateDataRequest, background_tasks: BackgroundTasks):
    """Start simulating live civic data"""
    
    background_tasks.add_task(simulate_civic_data, request.count, request.interval_seconds)
    
    return {
        "message": f"Started data simulation: {request.count} records every {request.interval_seconds} seconds",
        "simulation_active": True,
        "timestamp": create_timestamp()
    }

@app.delete("/data/clear")
async def clear_all_data():
    """Clear all stored data"""
    global live_data_store
    count = len(live_data_store)
    live_data_store.clear()
    
    return {
        "message": f"Cleared {count} records",
        "total_records": 0,
        "timestamp": create_timestamp()
    }

@app.get("/data/recent/{minutes}")
async def get_recent_data(minutes: int):
    """Get data from last N minutes"""
    
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    
    recent_data = []
    for record in live_data_store:
        try:
            record_time = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
            if record_time > cutoff_time:
                recent_data.append(record)
        except:
            continue
    
    # Sort by timestamp (most recent first)
    recent_data.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {
        "data": recent_data,
        "records_found": len(recent_data),
        "time_window_minutes": minutes,
        "timestamp": create_timestamp()
    }

# Background simulation task
async def simulate_civic_data(count: int, interval: int):
    """Background task to simulate live civic data"""
    
    import random
    import numpy as np
    
    global processing_status
    processing_status = {
        "status": "simulating",
        "message": f"Generating {count} records every {interval} seconds",
        "timestamp": create_timestamp()
    }
    
    for i in range(count):
        try:
            # Select random template
            template = random.choice(CIVIC_TEMPLATES)
            
            # Generate realistic text
            location = random.choice(template['location'])
            days = random.randint(1, 7)
            hours = random.randint(2, 12)
            
            text = template['text'].format(
                location=location,
                days=days,
                hours=hours
            )
            
            # Add some variation
            if random.random() < 0.3:  # 30% chance of adding urgency words
                urgency_words = ['URGENT!', 'EMERGENCY!', 'IMMEDIATE ACTION NEEDED!']
                text = f"{random.choice(urgency_words)} {text}"
            
            # Process the simulated text
            processed = process_single_text(text, "simulation")
            
            if processed:
                live_data_store.append(processed)
                print(f"Simulated record {i+1}: {text[:50]}...")
            
            # Sleep before next iteration
            if i < count - 1:  # Don't sleep after last iteration
                await asyncio.sleep(interval)
                
        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            continue
    
    processing_status = {
        "status": "idle",
        "message": f"Simulation completed. Generated {count} records.",
        "timestamp": create_timestamp()
    }

@app.get("/status")
async def get_processing_status():
    """Get current processing status"""
    return processing_status

# Initialize with some sample data
@app.on_event("startup")
async def startup_event():
    """Initialize system with sample data"""
    
    print("🚀 Starting Civic Data Ingest API...")
    
    # Add initial sample data
    initial_texts = [
        "Water supply disrupted in Mumbai for 3 days. Residents facing severe shortage! #WaterCrisis",
        "Great improvement in garbage collection in Pune! Thanks to municipal team. #CleanCity",
        "Potholes everywhere in Bangalore! Road conditions terrible. #RoadRepair",
        "Power outage in Delhi for 6 hours. When will electricity board take action? #PowerCut",
        "Traffic signals not working at main junction in Chennai. Safety concerns! #TrafficIssue"
    ]
    
    for text in initial_texts:
        processed = process_single_text(text, "initial")
        if processed:
            live_data_store.append(processed)
    
    print(f"✅ API started with {len(live_data_store)} initial records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)