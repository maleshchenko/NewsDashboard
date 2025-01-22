import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import json

def get_news_data():
    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"
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
                    "link": el.find("a")["href"],
                    "title": el.select_one("div.MBeuO").get_text(),
                    "snippet": el.select_one(".GI74Re").get_text(),
                    "date": el.select_one(".LfVVr").get_text(),
                    "source": el.select_one(".NUnG9d span").get_text()
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

def process_ollama_response(response_text):
    result = []
    for line in response_text.splitlines():
        try:
            json_line = json.loads(line)
            if "response" in json_line:
                result.append(json_line["response"])
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {line} - {e}")
    return "".join(result).strip()  # Concatenate all partial responses

def extract_keywords_and_cluster_ollama(articles, model="llama3.2"):
    clusters = []
    for article in articles:
        try:
            prompt = f"Extract keywords and suggest a topic cluster for the following news headline:\n\n'{article['title']}'"
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt},
            )
            parsed_response = process_ollama_response(response.text)

            print("Raw Response:", response.text)  # Log the raw streaming response
            print("Parsed Response:", parsed_response)  # Log the final processed response

            cluster = parsed_response or "Uncategorized"
            clusters.append({"title": article["title"], "cluster": cluster})
        except Exception as e:
            print(f"Error in Ollama processing: {e}")
            clusters.append({"title": article["title"], "cluster": "Uncategorized"})
    return clusters

def summarize_clusters_ollama(articles, model="llama3.2"):
    summaries = {}
    clusters = set(article["cluster"] for article in articles)
    for cluster in clusters:
        articles_in_cluster = [article["title"] for article in articles if article["cluster"] == cluster]
        if not articles_in_cluster:
            summaries[cluster] = "No articles available for summarization."
            continue

        try:
            prompt = f"Summarize these news articles about {cluster}:\n\n{articles_in_cluster}. Determine if each articls is positive or negative in regards to stok market"
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt},
            )
            parsed_response = process_ollama_response(response.text)
            summary = parsed_response or "Unable to generate summary."
            summaries[cluster] = summary
        except Exception as e:
            print(f"Error summarizing cluster {cluster} with Ollama: {e}")
            summaries[cluster] = "Unable to generate summary."
    return summaries

def display_ui():
    df = pd.read_csv("categorized_news.csv")
    st.title("Google News Topics & Summaries")

    topics = df["cluster"].unique()
    selected_topic = st.selectbox("Select a Topic", topics)

    st.subheader(f"Articles about {selected_topic}")
    filtered_df = df[df["cluster"] == selected_topic]

    st.write(filtered_df[["title", "link", "source"]].to_dict(orient="records"))

    st.subheader("Summary")
    st.write(filtered_df["summary"].iloc[0])

if __name__ == "__main__":
    print("Scraping Google News...")
    news_data = get_news_data()

    if news_data.empty:
        print("No data to process. Exiting.")
    else:
        print("Extracting keywords and clustering articles using Ollama...")
        articles_with_clusters = [
            {**article, "cluster": extract_keywords_and_cluster_ollama([article])[0]["cluster"]}
            for article in news_data.to_dict(orient="records")
        ]

        print("Summarizing clusters using Ollama...")
        summaries = summarize_clusters_ollama(articles_with_clusters)

        df = pd.DataFrame(articles_with_clusters)
        df["summary"] = df["cluster"].map(summaries)
        df.to_csv("categorized_news.csv", index=False)
        print("Categorized data saved to categorized_news.csv")

        print("Launching Streamlit UI...")
        display_ui()