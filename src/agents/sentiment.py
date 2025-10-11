# src/agents/sentiment.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ingest.news import fetch_rss_news

POS = {"beat", "beats", "growth", "surge", "upgrade", "bullish", "record", "strong", "win", "wins", "profit"}
NEG = {"miss", "cuts", "downgrade", "bearish", "slump", "weak", "loss", "lawsuit", "recall", "delay"}


def _score_text(text: str) -> float:
    t = text.lower()
    pos = sum(1 for w in POS if w in t)
    neg = sum(1 for w in NEG if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(pos + neg, 1)


# src/agents/sentiment.py
def _label(score: float) -> str:
    # tests expect uppercase labels
    if score > 0:
        return "POSITIVE"
    if score < 0:
        return "NEGATIVE"
    return "NEUTRAL"


def _read_news_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def run_sentiment(
    ticker: str,
    limit: int = 10,
    news_root: Optional[str] = None,
    out_root: Optional[str] = None,
) -> Union[List[Dict], Tuple[str, int]]:
    """
    If news_root/out_root are None: return a LIST of scored items (no files).
    If news_root/out_root are provided: write files and return (out_dir, n).
    Each item includes: title, link, published, summary, score, sentiment.
    """
    items: List[Dict]
    if news_root:
        news_path = Path(news_root) / ticker / "news.jsonl"
        items = _read_news_jsonl(news_path)
        if not items:
            items = fetch_rss_news(ticker, limit=limit)
    else:
        items = fetch_rss_news(ticker, limit=limit)

    scored: List[Dict] = []
    for r in items[:limit]:
        title = r.get("title", "")
        score = float(_score_text(title))
        scored.append({
            "published": r.get("published"),
            "title": title,
            # accept either 'link' or 'url' from inputs; output 'link'
            "link": r.get("link") or r.get("url"),
            "summary": r.get("summary", ""),
            "score": score,
            "sentiment": _label(score),
        })

    # If no out_root given -> return list (used by test_run_sentiment)
    if not out_root:
        return scored

    # Otherwise, write files and return (out_dir, n) (used by test_sentiment_agent)
    out_dir = Path(out_root) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, payload in enumerate(scored, start=1):
        (out_dir / f"{i:04d}.json").write_text(json.dumps(payload, indent=2))
    return (str(out_dir), len(scored))


def main() -> None:
    p = argparse.ArgumentParser(description="Score headlines and write sentiment JSON files.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--news-root", default=None)
    p.add_argument("--out-root", default=None)
    args = p.parse_args()

    res = run_sentiment(args.ticker, limit=args.limit, news_root=args.news_root, out_root=args.out_root)
    if isinstance(res, tuple):
        out_dir, n = res
        print(f"Wrote {n} sentiment files to {out_dir}")
    else:
        print(f"Scored {len(res)} headlines")
