from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Article:
    title: str
    url: str | None
    published_at: str | None
    text: str  # title + description or content


# Tiny keyword-based scorer (no external deps).
POS_WORDS = {"beats", "beat", "record", "surge", "rally", "upgraded", "strong", "growth", "bullish", "outperform"}
NEG_WORDS = {"miss", "misses", "downgraded", "plunge", "fall", "weak", "bearish", "lawsuit", "recall", "layoff"}

def score_text(text: str) -> tuple[float, str]:
    t = text.lower()
    pos = sum(w in t for w in POS_WORDS)
    neg = sum(w in t for w in NEG_WORDS)
    raw = pos - neg
    if raw > 0:
        return (min(1.0, 0.1 * raw), "POSITIVE")
    if raw < 0:
        return (max(-1.0, -0.1 * abs(raw)), "NEGATIVE")
    return (0.0, "NEUTRAL")


def read_news_jsonl(path: Path) -> Iterable[Article]:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        title = str(obj.get("title") or obj.get("headline") or "").strip()
        url = obj.get("url") or obj.get("link")
        published_at = obj.get("published_at") or obj.get("published") or obj.get("date")
        text = " ".join(
            s for s in [
                title,
                str(obj.get("description") or ""),
                str(obj.get("content") or ""),
            ] if s
        )
        if title:
            yield Article(title=title, url=url, published_at=published_at, text=text)


def write_one(out_dir: Path, idx: int, a: Article, score: float, label: str) -> None:
    out = {
        "title": a.title,
        "url": a.url,
        "published_at": a.published_at,
        "score": float(score),
        "sentiment": label,   # "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    }
    (out_dir / f"{idx:04d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--in-root", default="output")
    ap.add_argument("--out-root", default="output")
    ap.add_argument("--news-file", default="news.jsonl")   # for flexibility
    args = ap.parse_args()

    in_path = Path(args.in_root) / "news" / args.ticker / args.news_file
    if not in_path.exists():
        raise SystemExit(f"News file not found: {in_path}")

    out_dir = Path(args.out_root) / "sentiment" / args.ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, art in enumerate(read_news_jsonl(in_path), start=1):
        sc, label = score_text(art.text)
        write_one(out_dir, i, art, sc, label)
        count += 1

    print(f"Wrote {count} sentiment files to {out_dir}")


if __name__ == "__main__":
    main()
