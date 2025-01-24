import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import uuid

stocks_of_interest = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT"]

# ----------------------------- Scraping Functions -----------------------------

def get_news_data():
    """Scrape Google News for stock market-related articles."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
    }
    response = requests.get(
        "https://www.google.com/search?q=us+stock+markets&gl=us&tbm=nws&num=5", headers=headers
    )
    soup = BeautifulSoup(response.content, "html.parser")
    news_results = []

    for el in soup.select("div.SoaBEf"):
        try:
            news_results.append(
                {
                    "link": el.find("a")["href"] if el.find("a") else "No link available",
                    "title": el.select_one("div.MBeuO").get_text() if el.select_one("div.MBeuO") else "No title available",
                    "snippet": el.select_one(".GI74Re").get_text() if el.select_one(".GI74Re") else "No snippet available",
                    "date": el.select_one(".LfVVr").get_text() if el.select_one(".LfVVr") else "No date available",
                    "source": el.select_one(".NUnG9d span").get_text() if el.select_one(".NUnG9d span") else "No source available"
                }
            )
        except AttributeError:
            continue

    if not news_results:
        print("No articles found. Check the structure of Google News.")
        return pd.DataFrame()  # Return an empty DataFrame

    df = pd.DataFrame(news_results)
    df.to_csv("news_data.csv", index=False)
    print("Data saved to news_data.csv")
    return df

# --------------------------- Historical Data Handling -------------------------

def load_historical_data(file_path="historical_data.csv"):
    """Load historical sentiment data."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["title", "snippet", "link", "date", "source", "sentiment"])

def save_historical_data(new_data, file_path="historical_data.csv"):
    """Save sentiment data to historical storage with a date separator."""
    # Load existing data
    historical_data = load_historical_data(file_path)
    
    # Create a date separator row
    date_separator_row = {
        "title": f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n",
        "snippet": "",
        "link": "",
        "date": "",
        "source": "",
        "sentiment": ""
    }
    date_separator_df = pd.DataFrame([date_separator_row])
    
    # Combine the date separator, new data, and historical data
    updated_data = pd.concat([historical_data, date_separator_df, new_data], ignore_index=True)
    
    # Save to file
    updated_data.to_csv(file_path, index=False)
    print(f"Historical data updated and saved to {file_path}")

# ------------------------- Sentiment Analysis Functions -----------------------

def process_ollama_response(response_text):
    """Process and parse response text from the Ollama API."""
    result = []
    for line in response_text.splitlines():
        try:
            json_line = json.loads(line)
            if "response" in json_line:
                result.append(json_line["response"])
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {line} - {e}")
    return "".join(result).strip()

def analyze_sentiment_for_article(article, stocks, model="deepseek-r1:7b"): # "llama3.2"):
    """Analyze sentiment for a single news article."""
    try:
        prompt = (
            f"Analyze the sentiment of the following news article:\n\n"
            f"Title: {article['title']}\nSnippet: {article['snippet']}\n\n"
            f"Provide a recommendation for each stock: buy, sell, or hold."
        )
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt},
        )
        parsed_response = process_ollama_response(response.text)
        return article, parsed_response
    except Exception as e:
        print(f"Error analyzing sentiment for article: {e}")
        return article, "Unable to analyze sentiment."

def get_stock_recommendations(articles, stocks, model="llama3.2"):
    """Run sentiment analysis for all articles in parallel."""
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(lambda article: analyze_sentiment_for_article(article, stocks, model), articles)
        )
    return results

# -------------------------- Recommendation Aggregation ------------------------

def aggregate_recommendations_with_history(sentiment_results, stocks, historical_data, recency_weight=2, historical_decay=0.5):
    """Aggregate recommendations using current and historical data with recency weighting."""
    stock_sentiments = {stock: [] for stock in stocks}

    # Add current sentiment results with full weight
    for _, sentiment in sentiment_results:
        for stock in stocks:
            if stock in sentiment:
                if "buy" in sentiment.lower():
                    stock_sentiments[stock].extend(["buy"] * recency_weight)
                elif "sell" in sentiment.lower():
                    stock_sentiments[stock].extend(["sell"] * recency_weight)
                elif "hold" in sentiment.lower():
                    stock_sentiments[stock].extend(["hold"] * recency_weight)

    # Add historical data with decay factor for older entries
    for stock in stocks:
        historical_sentiments = historical_data[historical_data["sentiment"].str.contains(stock, na=False)]
        for index, row in historical_sentiments.iterrows():
            weight = max(1, recency_weight * (historical_decay ** index))  # Apply decay
            if "buy" in row["sentiment"].lower():
                stock_sentiments[stock].extend(["buy"] * int(weight))
            elif "sell" in row["sentiment"].lower():
                stock_sentiments[stock].extend(["sell"] * int(weight))
            elif "hold" in row["sentiment"].lower():
                stock_sentiments[stock].extend(["hold"] * int(weight))

    # Calculate final recommendation and confidence
    final_recommendations = {}
    for stock, sentiments in stock_sentiments.items():
        if sentiments:
            sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
            final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            confidence = (sentiment_counts[final_sentiment] / len(sentiments)) * 100
            final_recommendations[stock] = {"recommendation": final_sentiment, "confidence": round(confidence, 2)}
        else:
            final_recommendations[stock] = {"recommendation": "No recommendation", "confidence": 0.0}

    return final_recommendations

# ----------------------------- Streamlit UI -----------------------------------

def display_ui(stock_recommendations):
    """Display stock recommendations in Streamlit."""

    # Display stock recommendations
    st.subheader("Stock Recommendations")
    for stock, recommendation_data in stock_recommendations.items():
        recommendation = recommendation_data["recommendation"]
        confidence = recommendation_data["confidence"]
        st.write(f"**{stock}**: {recommendation} (Confidence: {confidence}%)")

# ------------------------------- Main Script ----------------------------------

if __name__ == "__main__":
    print("Scraping Google News...")
    news_data = get_news_data()

    if news_data.empty:
        print("No data to process. Exiting.")
    else:
        print("Loading historical data...")
        historical_data = load_historical_data()

        print("Analyzing sentiment for articles...")
        articles = news_data.to_dict(orient="records")
        sentiment_results = get_stock_recommendations(articles, stocks_of_interest)

        print("Aggregating stock recommendations with historical data...")
        stock_recommendations = aggregate_recommendations_with_history(sentiment_results, stocks_of_interest, historical_data)

        print("Saving current sentiment results to historical data...")
        current_data = pd.DataFrame([
            {
                "title": article["title"],
                "snippet": article["snippet"],
                "link": article["link"],
                "date": article["date"],
                "source": article["source"],
                "sentiment": sentiment
            }
            for article, sentiment in sentiment_results
        ])
        save_historical_data(current_data)

        print("Launching Streamlit UI...")
        display_ui(stock_recommendations)