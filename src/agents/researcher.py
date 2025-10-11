# src/agents/researcher.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

from ingest.news import fetch_rss_news


def _write_jsonl(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_researcher(
    ticker: str,
    limit: int = 10,
    out_root: str = "output/news",
    rss_urls: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Return a list of news items (dicts with title/link/published/summary)
    and write them to output/news/<ticker>/news.jsonl.
    """
    rows = fetch_rss_news(ticker, limit=limit, rss_urls=rss_urls)
    out_dir = Path(out_root) / ticker
    out_path = out_dir / "news.jsonl"
    _write_jsonl(rows, out_path)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Researcher agent: gather headlines and persist JSONL.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--out-root", default="output/news")
    p.add_argument("--rss-url", action="append", help="Optional RSS URL(s) to parse (else uses stub).")
    args = p.parse_args()

    rows = run_researcher(args.ticker, limit=args.limit, out_root=args.out_root, rss_urls=args.rss_url)
    print(f"Wrote {len(rows)} headlines to {Path(args.out_root) / args.ticker}")


if __name__ == "__main__":
    main()
