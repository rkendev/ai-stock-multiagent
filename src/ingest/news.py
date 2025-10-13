from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional


# Optional feed reader; if absent, we write a stub entry so the pipeline keeps working
try:
    import feedparser  # type: ignore
except Exception:
    feedparser = None


DEFAULT_RSS = "https://news.google.com/rss/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_since(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d")


def _fetch_rss(ticker: str, rss_url: Optional[str]) -> List[Dict[str, Any]]:
    if feedparser is None:
        # Fallback stub
        return [{
            "title": f"{ticker} placeholder headline (feedparser not installed)",
            "link": "",
            "published": _iso(datetime.utcnow()),
            "summary": "",
        }]

    url = (rss_url or DEFAULT_RSS).format(ticker=ticker)
    feed = feedparser.parse(url)
    items: List[Dict[str, Any]] = []
    for e in getattr(feed, "entries", []):
        items.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "published": getattr(e, "published", datetime.now(timezone.utc).isoformat()),
            "summary": getattr(e, "summary", ""),
        })
    return items


def _write_jsonl(items: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch RSS news and write JSONL")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out-root", default="output")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--since", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--rss-url", default=None, help="Override RSS URL; {ticker} will be substituted")
    args = ap.parse_args()

    since_dt = _parse_since(args.since)
    items = _fetch_rss(args.ticker, args.rss_url)

    # Optional since filter (by published field if it parses, otherwise keep)
    if since_dt:
        kept: List[Dict[str, Any]] = []
        for it in items:
            ts = it.get("published", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", ""))
            except Exception:
                dt = since_dt  # keep if we can't parse
            if dt >= since_dt:
                kept.append(it)
        items = kept

    items = items[: args.limit]

    out_dir = Path(args.out_root) / "news" / args.ticker
    out_path = out_dir / "news.jsonl"
    _write_jsonl(items, out_path)
    print(f"Wrote {out_path} ({len(items)} items)")


if __name__ == "__main__":
    main()
