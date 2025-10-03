# src/agents/researcher.py
import argparse
import json
import os
from datetime import datetime, timezone

from ingest.news import fetch_rss_news

def run_researcher(ticker: str, limit: int = 20):
    """
    Fetch recent RSS items for the ticker.
    Returns a list of dicts.
    """
    return fetch_rss_news(ticker, limit)

def main():
    p = argparse.ArgumentParser(description="Researcher agent: gather news for a ticker")
    p.add_argument("--ticker", required=True, help="Stock ticker symbol")
    p.add_argument("--limit",  type=int, default=20, help="Max number of articles")
    args = p.parse_args()

    items = run_researcher(args.ticker, args.limit)

    # Write to output folder
    out_dir = os.path.join("output", "researcher", args.ticker)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"news_{ts}.json")

    with open(out_path, "w", encoding="utf8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[Researcher] Wrote {len(items)} items to {out_path}")

if __name__ == "__main__":
    main()
