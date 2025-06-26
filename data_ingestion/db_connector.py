from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))

db = client["civic_sentiment"]
collection = db["raw_data"]

def store_record(record):
    collection.insert_one(record)
