import tweepy
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load API keys and secrets from .env
load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER")
MONGO_URI = os.getenv("MONGO_URI")

# Initialize Twitter client
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Initialize MongoDB client
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["public_sentiment"]
collection = db["tweets"]

def fetch_tweets():
    query = "(pothole OR electricity outage OR water cut OR #powercut) lang:en -is:retweet"
    
    tweets = client.search_recent_tweets(
        query=query,
        max_results=20,
        tweet_fields=['created_at', 'geo'],
        expansions='geo.place_id',
        place_fields='full_name,geo'
    )

    results = []
    places = {p["id"]: p for p in tweets.includes["places"]} if "places" in tweets.includes else {}

    if tweets.data:
        for tweet in tweets.data:
            place = None
            if tweet.geo and tweet.geo.get("place_id"):
                place_id = tweet.geo["place_id"]
                place = places.get(place_id, {}).get("full_name")

            tweet_data = {
                "text": tweet.text,
                "timestamp": tweet.created_at.isoformat(),
                "source": "Twitter",
                "location": place,
                "raw_json": tweet.data if isinstance(tweet.data, dict) else tweet.__dict__

            }

            results.append(tweet_data)
            collection.insert_one(tweet_data)
            print(f"Inserted: {tweet.text[:50]}...")

    return results

# Run the script
if __name__ == "__main__":
    fetched = fetch_tweets()
    print(f"\n✅ Fetched & Stored {len(fetched)} tweets in MongoDB.")
