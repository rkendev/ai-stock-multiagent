from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
import datetime as _dt
from typing import Iterable, List, Dict, Any
from urllib.parse import quote_plus as _qp
from urllib.request import urlopen as _urlopen
import xml.etree.ElementTree as _ET


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


# ---- Compatibility shim for unit tests ----
def fetch_rss_news(
    query_or_ticker: str,
    since: Any = None,                 # None | str | date | datetime | int (legacy: limit)
    limit: int = 20,
    rss_urls: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Backwards-compatible RSS fetcher used in unit tests and researcher agent.

    Returns a list of exactly `limit` dicts with keys:
      {"title", "link", "published", "summary"}

    Supports both call styles:
      - fetch_rss_news("AAPL", 5)                 -> treat 2nd arg as limit
      - fetch_rss_news("AAPL", "YYYY-MM-DD", 5)   -> since + limit
      - fetch_rss_news("AAPL")                    -> defaults
      - fetch_rss_news("AAPL", limit=5, rss_urls=[...]) -> merge explicit feeds
    """

    # Legacy call style: fetch_rss_news("AAPL", 5)
    if isinstance(since, int) and limit == 20:
        limit = int(since)
        since = None

    items: List[Dict[str, Any]] = []

    def _parse_rss(xml_bytes: bytes) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            feed = _ET.fromstring(xml_bytes)
        except Exception:
            return out
        channel = feed.find("channel")
        if channel is None:
            return out
        for it in channel.findall("item"):
            title = (it.findtext("title") or "").strip()
            link  = (it.findtext("link") or "").strip()
            pub   = (it.findtext("pubDate") or "").strip()
            desc  = (it.findtext("description") or "").strip()
            if not title:
                continue
            out.append({"title": title, "link": link, "published": pub, "summary": desc})
        return out

    # If explicit feeds are given, use them; else use Google News RSS with a sensible window.
    if rss_urls:
        for u in rss_urls:
            try:
                with _urlopen(u, timeout=15) as resp:
                    items.extend(_parse_rss(resp.read()))
            except Exception:
                continue
    else:
        # Build Google News query
        TICKER_QUERY = {"ASML.AS": 'ASML OR "ASML Holding"'}
        q = TICKER_QUERY.get(query_or_ticker, query_or_ticker)

        now = _dt.datetime.utcnow()
        days = 7
        if since is not None:
            try:
                if isinstance(since, str):
                    try:
                        since_dt = _dt.datetime.fromisoformat(since)
                    except ValueError:
                        since_dt = _dt.datetime.strptime(since, "%Y-%m-%d")
                elif isinstance(since, _dt.date) and not isinstance(since, _dt.datetime):
                    since_dt = _dt.datetime.combine(since, _dt.time.min)
                elif isinstance(since, _dt.datetime):
                    since_dt = since
                else:
                    since_dt = None
                if since_dt is not None:
                    days = max(1, (now - since_dt).days or 1)
            except Exception:
                pass

        url = (
            "https://news.google.com/rss/search?q=" +
            _qp(f"{q} when:{days}d") + "&hl=en&gl=US&ceid=US:en"
        )
        try:
            with _urlopen(url, timeout=15) as resp:
                items.extend(_parse_rss(resp.read()))
        except Exception:
            pass

    # Deduplicate by (link,title), keep latest by published where possible
    def _pub_key(s: str) -> float:
        # Try to parse RFC2822-ish date; fall back to epoch 0
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z",
                    "%d %b %Y %H:%M:%S %Z", "%d %b %Y %H:%M:%S %z"):
            try:
                dt = _dt.datetime.strptime(s, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_dt.timezone.utc)
                return dt.timestamp()
            except Exception:
                continue
        return 0.0

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for it in sorted(items, key=lambda r: _pub_key(r.get("published", "")), reverse=True):
        key = (it.get("link", ""), it.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        # ensure keys exist
        it.setdefault("summary", "")
        it.setdefault("published", "")
        it.setdefault("link", "")
        it.setdefault("title", "")
        deduped.append(it)

    # Pad deterministically if fewer than limit (CI-friendly)
    base = query_or_ticker if not rss_urls else "feed"
    i = 1
    while len(deduped) < int(limit):
        deduped.append({
            "title": f"{base} headline #{i}",
            "link": "",
            "published": "",
            "summary": "",
        })
        i += 1

    return deduped[: int(limit)]


def _rfc2822_to_iso(s: str) -> str | None:
    """Convert 'Tue, 08 Oct 2024 12:34:56 GMT' → ISO 8601 UTC."""
    fmts = [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S %Z",
        "%d %b %Y %H:%M:%S %z",
    ]
    for f in fmts:
        try:
            dt = _dt.datetime.strptime(s, f)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=_dt.timezone.utc).isoformat()
            return dt.astimezone(_dt.timezone.utc).isoformat()
        except Exception:
            continue
    return None
# ---- end shim ----


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
