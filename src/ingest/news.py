from datetime import datetime, timezone
# src/ingest/news.py
import argparse
import json
import os
from datetime import datetime
import feedparser

def fetch_rss_news(ticker: str, limit: int = 20):
    import feedparser
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:limit]:
        items.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "published": getattr(e, "published", ""),
            "summary": getattr(e, "summary", "")
        })
    return items


def main():
    p = argparse.ArgumentParser(description="Download RSS news for a ticker")
    p.add_argument("--ticker", required=True, help="Stock ticker symbol")
    p.add_argument("--limit",  type=int, default=20, help="Max number of articles")
    args = p.parse_args()

    items = fetch_rss_news(args.ticker, args.limit)
    out_dir = os.path.join("data", args.ticker)
    json_dir = os.path.join(out_dir, "news")
    os.makedirs(json_dir, exist_ok=True)

    # Timestamped file for repeatability
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(json_dir, f"rss_{ts}.json")
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} news items to {out_path}")

if __name__ == "__main__":
    main()
