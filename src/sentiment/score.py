from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional


@dataclass
class Article:
    title: str
    link: Optional[str]
    published: Optional[str]
    text: str  # title + description/content


# Tiny keyword-based scorer (no external deps).
POS_WORDS = {
    "beats", "beat", "record", "surge", "rally", "upgraded",
    "strong", "growth", "bullish", "outperform"
}
NEG_WORDS = {
    "miss", "misses", "downgraded", "plunge", "fall", "weak",
    "bearish", "lawsuit", "recall", "layoff"
}


def score_text(text: str) -> Tuple[float, str]:
    t = text.lower()
    pos = sum(w in t for w in POS_WORDS)
    neg = sum(w in t for w in NEG_WORDS)
    raw = pos - neg
    if raw > 0:
        return (min(1.0, 0.1 * raw), "POSITIVE")
    if raw < 0:
        return (max(-1.0, -0.1 * abs(raw)), "NEGATIVE")
    return (0.0, "NEUTRAL")


def _read_news_jsonl(path: Path) -> Iterable[Article]:
    """
    Accepts heterogeneous inputs. It tries common keys:
      title:       "title" | "headline"
      link/url:    "link"  | "url"
      published:   "published" | "published_at" | "date"
      body fields: "description" | "content"
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        title = str(obj.get("title") or obj.get("headline") or "").strip()
        if not title:
            continue

        link = obj.get("link") or obj.get("url")
        published = obj.get("published") or obj.get("published_at") or obj.get("date")
        text = " ".join(
            s for s in [
                title,
                str(obj.get("description") or ""),
                str(obj.get("content") or ""),
            ] if s
        )

        yield Article(title=title, link=link, published=published, text=text)


def _write_one(out_dir: Path, idx: int, a: Article, score: float, label: str) -> None:
    """
    Write a single scored article. Keys intentionally match tests:
      - title (str)
      - link (str or "")
      - published (str or "")    <-- NOT 'published_at'
      - score (float)
      - sentiment (str)
    """
    out = {
        "title": a.title,
        "link": a.link or "",
        "published": a.published or "",
        "score": float(score),
        "sentiment": label,  # "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    }
    (out_dir / f"{idx:04d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def score_articles_for_ticker(
    ticker: str,
    in_root: str = "output",
    out_root: str = "output",
    news_file: str = "news.jsonl",
) -> int:
    """
    Read output/news/<TICKER>/<news_file> and write one json per article to
    output/sentiment/<TICKER>/*.json. Returns count written.
    """
    in_path = Path(in_root) / "news" / ticker / news_file
    if not in_path.exists():
        return 0

    out_dir = Path(out_root) / "sentiment" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, art in enumerate(_read_news_jsonl(in_path), start=1):
        sc, label = score_text(art.text)
        _write_one(out_dir, i, art, sc, label)
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--in-root", default="output")
    ap.add_argument("--out-root", default="output")
    ap.add_argument("--news-file", default="news.jsonl")
    args = ap.parse_args()

    n = score_articles_for_ticker(
        args.ticker,
        in_root=args.in_root,
        out_root=args.out_root,
        news_file=args.news_file,
    )
    print(f"Wrote {n} sentiment files to {Path(args.out_root) / 'sentiment' / args.ticker}")


if __name__ == "__main__":
    main()
