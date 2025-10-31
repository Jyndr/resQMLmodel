# prompt: now i want to extract the real time news and ready to send predicted processed reduklt

import requests
import feedparser
import pandas as pd
import nltk
import json
import os
import time
import ssl
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from fake_useragent import UserAgent
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Assuming these are global variables that will be populated by build_fake_news_detection_model
ml_model = None
vectorizer = None

DISASTER_KEYWORDS = [
    "disaster", "earthquake", "flood", "cyclone", "hurricane", "tsunami",
    "wildfire", "forest fire","landslide","tornado", "storm",
    "drought","extreme weather"
]


class NewsArticle:
    def __init__(self, source, title, content, url, published_date=None):
        self.source = source
        self.title = title
        self.content = content if content else ""
        self.url = url
        self.published_date = published_date # Corrected assignment
        self.is_disaster_related = self._check_disaster_relevance()

    def _check_disaster_relevance(self):
        """Check if content is related to disasters using keyword matching"""
        text = f"{self.title} {self.content}".lower()
        return any(keyword.lower() in text for keyword in DISASTER_KEYWORDS)

    def to_dict(self):
        return {
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_date": self.published_date,
            "is_disaster_related": self.is_disaster_related
        }

def scrape_google_news(location=None):
    """Scrape Google News for disaster-related articles for a given location."""
    articles = []

    # Multiple queries for better coverage
    queries = [
        "disaster",
        "earthquake+OR+tsunami",
        "hurricane+OR+tornado+OR+storm",
        "wildfire+OR+forest+fire",
        "flood+OR+landslide"
    ]

    for query in queries:
        search_query = f"{location}+{query}" if location else query
        url = f"https://news.google.com/rss/search?q={search_query}"
        try:
            feed = feedparser.parse(url)


            for entry in feed.entries:
                article = NewsArticle(
                    source="Google News",
                    title=entry.title,
                    content=entry.summary if hasattr(entry, 'summary') else "",
                    url=entry.link,
                    published_date=entry.published if hasattr(entry, 'published') else None
                )

                if article.is_disaster_related:
                    articles.append(article.to_dict())

        except Exception as e:
            print(f"Error scraping Google News for query '{query}': {str(e)}")


    return articles

def scrape_news_websites(location=None):
    """Scrape Indian news websites using RSS feeds for a given location."""
    articles = []
    RSS_FEEDS = {
        "NDTV": "https://feeds.feedburner.com/ndtvnews-top-stories",
        "The Times of India": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
        "Hindustan Times": "https://www.hindustantimes.com/rss/topnews/rssfeed.xml",
        "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
        "India Today": "https://www.indiatoday.in/rss/1"
    }
    for source, url in RSS_FEEDS.items():
        try:
            print(f"Checking {source}...")
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.title if hasattr(entry, 'title') else ""
                content = entry.summary if hasattr(entry, 'summary') else ""
                link = entry.link if hasattr(entry, 'link') else ""
                published = entry.published if hasattr(entry, 'published') else None

                # Check if location is in title or content (simple approach)
                if location and location.lower() not in title.lower() and location.lower() not in content.lower():
                    continue # Skip article if location not found


                article = NewsArticle(
                    source=source,
                    title=title,
                    content=content,
                    url=link,
                    published_date=published
                )
                if article.is_disaster_related:
                    articles.append(article.to_dict())
        except Exception as e:
            print(f"Error scraping {source}: {str(e)}")
    return articles

def scrape_twitter(location=None):
    """Scrape Twitter for disaster-related tweets for a given location."""
    articles = []

    search_url = "https://api.twitter.com/2/tweets/search/recent"

    # Sample query (adjust as needed)
    query = '("earthquake" OR "flood" OR "cyclone" OR "hurricane" OR "wildfire" OR "landslide" OR "tsunami" OR "tornado" OR "disaster" OR "volcanic eruption") ("breaking" OR "alert" OR "update" OR "news" OR "evacuation" OR "damage" OR "rescue" OR "emergency") lang:en'
    if location:
        query = f"{location} {query}"

    query_params = {
        'query': query,
        'max_results': '50',  # Max results per request
        'tweet.fields': 'created_at,author_id',
    }

    # Twitter API credentials (replace with your actual credentials)
    API_KEY = "YOUR_API_KEY"
    API_SECRET = "YOUR_API_SECRET"
    BEARER_TOKEN = "YOUR_BEARER_TOKEN"

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }

    try:
        response = requests.get(search_url, headers=headers, params=query_params)
        if response.status_code != 200:
            raise Exception(
                f"Request returned an error: {response.status_code} {response.text}"
            )

        data = response.json()
        tweets = data.get("data", [])


        for tweet in tweets:
            article = NewsArticle(
                source="Twitter",
                title=tweet.get("text", ""),
                content="",
                url=f"https://twitter.com/i/web/status/{tweet.get('id')}",
                published_date=tweet.get("created_at")
            )

            if article.is_disaster_related:
                articles.append(article.to_dict())

    except Exception as e:
        print(f"Error scraping Twitter: {str(e)}")


    return articles

def scrape_newsapi(location=None, api_key="7911c7cdb7024fa6aa98108541d35a3b"):
    """Fetch disaster news from NewsAPI for a given location."""
    print("Fetching from NewsAPI...")
    articles = []

    if not api_key:

        return articles

    url = "https://newsapi.org/v2/everything"
    query = " OR ".join(DISASTER_KEYWORDS[:10])  # Use first 10 keywords
    if location:
        query = f"{location} AND ({query})"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code != 200:
            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return articles

        if "articles" not in data:

            return articles


        for article_data in data["articles"]:
            article = NewsArticle(
                source=article_data["source"]["name"],
                title=article_data["title"],
                content=article_data["description"] if article_data.get("description") else "",
                url=article_data["url"],
                published_date=article_data["publishedAt"]
            )

            if article.is_disaster_related:
                articles.append(article.to_dict())

    except Exception as e:
        print(f"Error fetching from NewsAPI: {str(e)}")


    return articles


def clean_text(text):
    """Clean and normalize text data"""
    if not text:
        return ""

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = ' '.join(word for word in text.split() if not word.startswith('http'))

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def classify_disaster(title, content):
    """Classify if an article is related to a disaster."""
    text = f"{title} {content}".lower()
    return any(keyword.lower() in text for keyword in DISASTER_KEYWORDS)


def clean_combined_dataset(df):
    """Clean and preprocess the combined dataset"""

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()  # Create an explicit copy

    if df.empty:
        return df

    # Remove duplicates based on URL
    initial_count = len(df)
    df = df.drop_duplicates(subset=["url"])
    # Ensure all required columns exist
    for col in ["source", "title", "content", "url", "published_date"]:
        if col not in df.columns:
            df[col] = None

    # Clean text data
    df.loc[:, 'title'] = df['title'].fillna('').astype(str).apply(clean_text)
    df.loc[:, 'content'] = df['content'].fillna('').astype(str).apply(clean_text)

    # Add timestamp for real-time tracking
    df.loc[:, 'timestamp'] = datetime.now().isoformat()


    # Double-verify disaster relevance using NLP
    df.loc[:, "is_disaster_related"] = df.apply(
    lambda row: classify_disaster(row['title'], row['content']), axis=1
)

    # Filter to keep only disaster-related content
    df_filtered = df[df["is_disaster_related"]]
    print(f"{len(df_filtered)}")
    return df_filtered

# Assuming ml_model and vectorizer are initialized globally or passed as arguments
ml_model = None
vectorizer = None

# Fallback trust score calculation using heuristics
def fallback_calculate_trust_score(article):
    """Fallback method to calculate trust score when ML model is unavailable"""
    # Base score between 0.6 and 0.9
    base_score = 0.6

    # List of trusted mainstream sources
    mainstream_sources = ['NDTV','The Times Of India', 'Hindustan Times', 'The Hindu', 'India Today']
    # Bonus for mainstream sources
    if any(source in article['source'] for source in mainstream_sources):
        base_score += 0.2

    # Bonus for having a valid URL
    if article['url'] and ('http' in article['url']):
        base_score += 0.1

    # Check content length - higher quality articles tend to be longer
    content_length = len(article['content']) if article['content'] else 0
    if content_length > 500:
        base_score += 0.05

    # Cap at 0.95
    return min(base_score, 0.95)

def calculate_trust_score(article):
    """Calculate a trust score for an article using ML model"""
    global ml_model, vectorizer

    # If model not loaded, use fallback method
    if ml_model is None or vectorizer is None:
        # Fallback to a simpler heuristic if model is not available
        return fallback_calculate_trust_score(article)

    # Prepare the text for prediction
    text = f"{article['title']} {article['content']}" if article['content'] else article['title']

    try:
        # Transform text using vectorizer
        text_tfidf = vectorizer.transform([text])

        # Get probability of being 'real'
        probabilities = ml_model.predict_proba(text_tfidf)[0]
        real_idx = ml_model.classes_.tolist().index('real')
        trust_score = probabilities[real_idx]

        # Ensure score is between 0 and 1
        return max(0.1, min(0.95, trust_score))

    except Exception as e:

        return fallback_calculate_trust_score(article)

def calculate_trust_scores_for_dataset(df):
    """Calculate trust scores for the entire dataset using ML model"""


    if df.empty:
        return df

    # Calculate trust scores
    df["trust_score"] = df.apply(
        lambda row: calculate_trust_score(row),
        axis=1
    )

    return df


def build_fake_news_detection_model():
    """Build and train a model to detect fake news"""


    # Prepare training data
    # For demo purposes, we'll create a simple dataset with synthetic labels
    # In a real implementation, you would use an actual labeled dataset

    # Collect some news articles first
    articles = []
    articles.extend(scrape_google_news())
    articles.extend(scrape_news_websites())

    if not articles:

        return None, None

    df = pd.DataFrame(articles)

    # Create labeled data (in a real scenario, you would have actual labeled data)
    # For now, we'll use heuristics to create synthetic labels

    # Mainstream sources are more likely to be real (but not guaranteed)
    mainstream_sources = ['NDTV', 'The Times of India', 'Hindustan Times', 'The Hindu', 'India Today']

    # Assign 'real' label to mainstream sources and longer content articles
    df['label'] = 'fake'  # Default

    # Mark as real if from mainstream source
    df.loc[df['source'].apply(lambda x: any(s in x for s in mainstream_sources)), 'label'] = 'real'

    # Mark as real if content is substantial
    df.loc[df['content'].apply(lambda x: len(str(x)) > 500), 'label'] = 'real'

    # Random label assignment to create more balanced dataset
    # Assign 'fake' label to 20% of the 'real' labeled articles randomly
    real_indices = df[df['label'] == 'real'].index.tolist()
    import random
    fake_indices = random.sample(real_indices, k=int(0.2 * len(real_indices)))
    df.loc[fake_indices, 'label'] = 'fake'

    # Combine title and content for better features
    df['text'] = df['title'] + ' ' + df['content'].fillna('')

    # Split into training and test sets
    from sklearn.model_selection import train_test_split
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature extraction using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train the model (Logistic Regression for simplicity and interpretability)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate model
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    return model, tfidf_vectorizer

def collect_all_disaster_news(location=None):
    """Collect disaster news from all sources into a single combined dataset"""

    all_articles = []

    # Collect from various sources
    all_articles.extend(scrape_google_news(location))
    all_articles.extend(scrape_news_websites(location))
    all_articles.extend(scrape_twitter(location))
    all_articles.extend(scrape_newsapi(location))

    # Convert to DataFrame
    df = pd.DataFrame(all_articles)

    if df.empty:

        return df


    # Clean and process the combined dataset
    df = clean_combined_dataset(df)

    # Calculate trust scores using ML model
    df = calculate_trust_scores_for_dataset(df)

    return df

def process_and_display_results(df):
    """Process the collected data and prepare for display"""
    if df.empty:

        return pd.DataFrame()

    # Convert trust_score to percentage
    df['trust_score_percent'] = (df['trust_score'] * 100).round(1)

    # Prepare final output
    output_df = df[['source', 'title', 'url', 'trust_score_percent']].copy()
    output_df.columns = ['Source', 'Title', 'URL', 'Trust Score (%)']


    print(output_df.head(10))
    return output_df

def send_prediction_results(results):
    """Sends the predicted processed results to a designated endpoint."""

    # In a real implementation, you'd have a specific API or messaging system
    # Here, we'll just print the results to the console as an example
    print(json.dumps(results, indent=2))

# Build ML model for fake news detection before calling main
ml_model, vectorizer = build_fake_news_detection_model()

def main():
    # Get location input from the user
    location = input("Enter the location for disaster news: ")

    # Collect and process all disaster news for the specified location
    df = collect_all_disaster_news(location)

    # Process and display results
    processed_df = process_and_display_results(df)

    # Send predicted processed results
    if not processed_df.empty:
        results = []
        for _, row in processed_df.iterrows():
            results.append({
                'source': row['Source'],
                'title': row['Title'],
                'url': row['URL'],
                'trust_score': row['Trust Score (%)']
            })
        send_prediction_results(results)

if __name__ == "__main__":
    main()
