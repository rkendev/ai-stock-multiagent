# src/agents/sentiment.py
import argparse
import json
import os
from datetime import datetime

from transformers import pipeline
from ingest.news import fetch_rss_news

# Initialize the FinBERT sentiment pipeline on GPU (device=0)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert", 
    device=0
)

def run_sentiment(ticker: str, limit: int = 20):
    """
    Fetch news with the Researcher logic, then classify each item.
    Returns list of dicts with added 'sentiment' and 'score'.
    """
    items = fetch_rss_news(ticker, limit)
    for itm in items:
        result = sentiment_pipe(itm["title"], truncation=True)[0]
        # Standardize to uppercase so tests pass
        itm["sentiment"] = result["label"].upper()
        itm["score"]     = float(result["score"])
    return items

def main():
    p = argparse.ArgumentParser(description="Sentiment agent: label news for a ticker")
    p.add_argument("--ticker", required=True, help="Stock ticker symbol")
    p.add_argument("--limit",  type=int, default=20, help="Max number of articles")
    args = p.parse_args()

    labeled = run_sentiment(args.ticker, args.limit)

    out_dir = os.path.join("output", "sentiment", args.ticker)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"sentiment_{ts}.json")

    with open(out_path, "w", encoding="utf8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    print(f"[Sentiment] Wrote {len(labeled)} items to {out_path}")

if __name__ == "__main__":
    main()
