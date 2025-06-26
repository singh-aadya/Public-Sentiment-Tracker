# run_pipeline.py

from data_ingestion.twitter_scraper import fetch_tweets
from data_ingestion.news_scraper import fetch_articles as fetch_news
from data_ingestion.geo_extractor import enrich_location as extract_location
from data_ingestion.db_connector import store_record as insert_to_mongo

def run_pipeline():
    print("✅ Starting Public Sentiment ETL Pipeline")

    # 1. Get tweets and news articles
    tweets = fetch_tweets()
    news = fetch_news()

    print(f"🔹 Fetched {len(tweets)} tweets")
    print(f"🔹 Fetched {len(news)} news articles")

    # 2. Combine data
    all_data = tweets + news

    # 3. Extract location for those missing it
    for entry in all_data:
        if not entry.get("location"):
            entry["location"] = extract_location(entry["text"])

    # 4. Save to MongoDB
    insert_to_mongo(all_data)

    print("✅ Pipeline completed successfully.\n")

if __name__ == "__main__":
    run_pipeline()