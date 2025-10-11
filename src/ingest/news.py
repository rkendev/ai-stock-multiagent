# src/ingest/news.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

try:
    import feedparser  # type: ignore
except Exception:  # pragma: no cover
    feedparser = None


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _stub_headlines(ticker: str, limit: int) -> List[Dict]:
    now = datetime.now(timezone.utc)
    rows: List[Dict] = []
    for i in range(limit):
        rows.append({
            "published": _iso(now),
            "title": f"{ticker} sample headline #{i+1}",
            "link": f"https://example.com/{ticker}/{i+1}",
            "summary": f"Stub summary for {ticker} #{i+1}",
        })
    return rows


def fetch_rss_news(
    ticker: str,
    limit: int = 10,
    rss_urls: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Returns list of dicts with keys: title, link, published, summary.
    Falls back to an offline stub if RSS is unavailable.
    """
    if not rss_urls or feedparser is None:
        return _stub_headlines(ticker, limit)

    items: List[Dict] = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)  # type: ignore
        except Exception:
            continue
        for e in feed.get("entries", []):
            title = e.get("title") or ""
            link = e.get("link") or ""
            summary = e.get("summary") or ""
            pub = None
            for k in ("published_parsed", "updated_parsed"):
                tm = e.get(k)
                if tm:
                    try:
                        pub = datetime(*tm[:6], tzinfo=timezone.utc)
                        break
                    except Exception:
                        pass
            if pub is None:
                pub = datetime.now(timezone.utc)
            items.append({
                "published": _iso(pub),
                "title": str(title),
                "link": str(link),
                "summary": str(summary),
            })
    items.sort(key=lambda r: r["published"], reverse=True)
    return items[:limit] if items else _stub_headlines(ticker, limit)


# Back-compat alias some tests import
def fetch_headlines(ticker: str, limit: int = 10) -> List[Dict]:
    return fetch_rss_news(ticker, limit=limit, rss_urls=None)


def write_news_jsonl(
    ticker: str,
    out_root: str = "output",
    limit: int = 10,
    rss_urls: Optional[List[str]] = None,
) -> str:
    rows = fetch_rss_news(ticker, limit=limit, rss_urls=rss_urls)
    out_dir = Path(out_root) / "news" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "news.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest headlines for a ticker (RSS or offline stub).")
    p.add_argument("--ticker", required=True)
    p.add_argument("--out-root", default="output")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--rss-url", action="append", help="Optional RSS URL(s).")
    args = p.parse_args()

    path = write_news_jsonl(
        args.ticker,
        out_root=args.out_root,
        limit=args.limit,
        rss_urls=args.rss_url,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
