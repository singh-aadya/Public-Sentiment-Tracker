"""
Database Connector
Handles connections to MongoDB and PostgreSQL for storing civic data
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# PostgreSQL imports
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

@dataclass
class CivicDataRecord:
    """Data class for civic data records"""
    text: str
    source: str  # 'twitter' or 'news'
    timestamp: str
    location: Optional[Dict] = None
    raw_data: Optional[Dict] = None
    processed_data: Optional[Dict] = None

class MongoDBConnector:
    """MongoDB connector for storing civic data"""
    
    def __init__(self, connection_string: str = None):
        """Initialize MongoDB connection"""
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo not installed. Install with: pip install pymongo")
        
        self.connection_string = connection_string or os.getenv('MONGODB_URL', 'mongodb://localhost:27017/')
        self.client = None
        self.db = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self, database_name: str = 'civic_data'):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            self.logger.info(f"Connected to MongoDB database: {database_name}")
            return True
        except ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    def create_collections(self):
        """Create necessary collections with indexes"""
        try:
            # Create collections
            collections = ['raw_data', 'processed_data', 'classifications', 'locations']
            
            for collection_name in collections:
                if collection_name not in self.db.list_collection_names():
                    self.db.create_collection(collection_name)
            
            # Create indexes
            self.db.raw_data.create_index([('source', 1), ('timestamp', -1)])
            self.db.raw_data.create_index([('text', 'text')])  # Text search index
            self.db.processed_data.create_index([('sentiment.compound', 1)])
            self.db.classifications.create_index([('category', 1), ('urgency', 1)])
            self.db.locations.create_index([('coordinates', '2dsphere')])  # Geospatial index
            
            self.logger.info("Collections and indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating collections: {str(e)}")
    
    def insert_raw_data(self, data: List[Dict]) -> List[str]:
        """Insert raw data records"""
        try:
            # Add insertion timestamp
            for record in data:
                record['inserted_at'] = datetime.utcnow()
            
            result = self.db.raw_data.insert_many(data)
            self.logger.info(f"Inserted {len(result.inserted_ids)} raw data records")
            return [str(id) for id in result.inserted_ids]
            
        except Exception as e:
            self.logger.error(f"Error inserting raw data: {str(e)}")
            return []
    
    def insert_processed_data(self, data: List[Dict]) -> List[str]:
        """Insert processed data records"""
        try:
            for record in data:
                record['processed_at'] = datetime.utcnow()
            
            result = self.db.processed_data.insert_many(data)
            self.logger.info(f"Inserted {len(result.inserted_ids)} processed data records")
            return [str(id) for id in result.inserted_ids]
            
        except Exception as e:
            self.logger.error(f"Error inserting processed data: {str(e)}")
            return []
    
    def get_unprocessed_data(self, limit: int = 100) -> List[Dict]:
        """Get unprocessed raw data"""
        try:
            # Find raw data that doesn't have corresponding processed data
            pipeline = [
                {
                    "$lookup": {
                        "from": "processed_data",
                        "localField": "_id",
                        "foreignField": "raw_id",
                        "as": "processed"
                    }
                },
                {
                    "$match": {
                        "processed": {"$size": 0}
                    }
                },
                {
                    "$limit": limit
                }
            ]
            
            return list(self.db.raw_data.aggregate(pipeline))
            
        except Exception as e:
            self.logger.error(f"Error getting unprocessed data: {str(e)}")
            return []
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.logger.info("MongoDB connection closed")

class PostgreSQLConnector:
    """PostgreSQL connector for storing civic data"""
    
    def __init__(self, connection_string: str = None):
        """Initialize PostgreSQL connection"""
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2")
        
        self.connection_string = connection_string or os.getenv('DATABASE_URL')
        self.connection = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.connection = psycopg2.connect(self.connection_string)
            self.connection.autocommit = True
            self.logger.info("Connected to PostgreSQL database")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            return False
    
    def create_tables(self):
        """Create necessary tables"""
        try:
            cursor = self.connection.cursor()
            
            # Raw data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_data (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP,
                    location JSONB,
                    raw_json JSONB,
                    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Processed data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_data (
                    id SERIAL PRIMARY KEY,
                    raw_id INTEGER REFERENCES raw_data(id),
                    cleaned_text TEXT,
                    keywords TEXT[],
                    sentiment_compound FLOAT,
                    sentiment_positive FLOAT,
                    sentiment_negative FLOAT,
                    sentiment_neutral FLOAT,
                    tfidf_vector FLOAT[],
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Classifications table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    id SERIAL PRIMARY KEY,
                    raw_id INTEGER REFERENCES raw_data(id),
                    category VARCHAR(100),
                    urgency VARCHAR(20),
                    confidence FLOAT,
                    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Locations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS locations (
                    id SERIAL PRIMARY KEY,
                    raw_id INTEGER REFERENCES raw_data(id),
                    location_name VARCHAR(255),
                    latitude FLOAT,
                    longitude FLOAT,
                    address TEXT,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_source_timestamp ON raw_data(source, timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_data_sentiment ON processed_data(sentiment_compound);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_classifications_category ON classifications(category, urgency);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_locations_coords ON locations(latitude, longitude);")
            
            cursor.close()
            self.logger.info("Tables and indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {str(e)}")
    
    def insert_raw_data(self, data: List[Dict]) -> List[int]:
        """Insert raw data records"""
        try:
            cursor = self.connection.cursor()
            
            # Prepare data for insertion
            insert_data = []
            for record in data:
                insert_data.append((
                    record.get('text', ''),
                    record.get('source', ''),
                    record.get('timestamp'),
                    json.dumps(record.get('location')) if record.get('location') else None,
                    json.dumps(record)
                ))
            
            # Insert data
            execute_values(
                cursor,
                """INSERT INTO raw_data (text, source, timestamp, location, raw_json) 
                   VALUES %s RETURNING id""",
                insert_data
            )
            
            inserted_ids = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            self.logger.info(f"Inserted {len(inserted_ids)} raw data records")
            return inserted_ids
            
        except Exception as e:
            self.logger.error(f"Error inserting raw data: {str(e)}")
            return []
    
    def get_unprocessed_data(self, limit: int = 100) -> List[Dict]:
        """Get unprocessed raw data"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT r.* FROM raw_data r
                LEFT JOIN processed_data p ON r.id = p.raw_id
                WHERE p.raw_id IS NULL
                ORDER BY r.timestamp DESC
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"Error getting unprocessed data: {str(e)}")
            return []
    
    def close(self):
        """Close PostgreSQL connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("PostgreSQL connection closed")

class DatabaseManager:
    """Unified database manager supporting both MongoDB and PostgreSQL"""
    
    def __init__(self, db_type: str = 'mongodb'):
        """
        Initialize database manager
        
        Args:
            db_type: 'mongodb' or 'postgresql'
        """
        self.db_type = db_type.lower()
        self.connector = None
        
        if self.db_type == 'mongodb':
            self.connector = MongoDBConnector()
        elif self.db_type == 'postgresql':
            self.connector = PostgreSQLConnector()
        else:
            raise ValueError("db_type must be 'mongodb' or 'postgresql'")
    
    def setup_database(self, database_name: str = None):
        """Setup database connection and create necessary structures"""
        if self.db_type == 'mongodb':
            success = self.connector.connect(database_name or 'civic_data')
            if success:
                self.connector.create_collections()
        else:
            success = self.connector.connect()
            if success:
                self.connector.create_tables()
        
        return success
    
    def store_raw_data(self, data: List[Dict]) -> List:
        """Store raw data"""
        return self.connector.insert_raw_data(data)
    
    def get_unprocessed_data(self, limit: int = 100) -> List[Dict]:
        """Get unprocessed data"""
        return self.connector.get_unprocessed_data(limit)
    
    def close(self):
        """Close database connection"""
        self.connector.close()

def main():
    """Test database connections"""
    print("Testing database connections...")
    
    # Test MongoDB (if available)
    if MONGODB_AVAILABLE:
        print("\nTesting MongoDB:")
        try:
            db_manager = DatabaseManager('mongodb')
            success = db_manager.setup_database()
            if success:
                print("✓ MongoDB connection successful")
                
                # Test data insertion
                test_data = [{
                    'text': 'Test civic issue',
                    'source': 'test',
                    'timestamp': datetime.utcnow().isoformat()
                }]
                
                ids = db_manager.store_raw_data(test_data)
                if ids:
                    print(f"✓ Test data inserted with IDs: {ids}")
                
                db_manager.close()
            else:
                print("✗ MongoDB connection failed")
        except Exception as e:
            print(f"✗ MongoDB error: {str(e)}")
    else:
        print("✗ MongoDB not available (pymongo not installed)")
    
    # Test PostgreSQL (if available)
    if POSTGRESQL_AVAILABLE and os.getenv('DATABASE_URL'):
        print("\nTesting PostgreSQL:")
        try:
            db_manager = DatabaseManager('postgresql')
            success = db_manager.setup_database()
            if success:
                print("✓ PostgreSQL connection successful")
                db_manager.close()
            else:
                print("✗ PostgreSQL connection failed")
        except Exception as e:
            print(f"✗ PostgreSQL error: {str(e)}")
    else:
        print("✗ PostgreSQL not available (psycopg2 not installed or DATABASE_URL not set)")

if __name__ == "__main__":
    main()