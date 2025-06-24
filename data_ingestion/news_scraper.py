from newspaper import Article

def fetch_articles():
    urls = [
        "https://timesofindia.indiatimes.com/city/delhi",
        "https://www.thehindu.com/news/cities/chennai/"
    ]
    results = []
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            results.append({
                "text": article.title + ". " + article.text,
                "timestamp": article.publish_date.isoformat() if article.publish_date else None,
                "source": "News",
                "location": None,
                "url": url,
                "raw_html": article.html
            })
        except:
            continue
    return results
