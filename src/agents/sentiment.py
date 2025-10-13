from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not p.exists():
        return items
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            pass
    return items


def _score_summary(text: str) -> float:
    """
    Tiny heuristic scorer to keep pipeline simple. Replace in future branch with your real NLP.
    """
    txt = text.lower()
    pos = sum(w in txt for w in ["beat", "surge", "up", "growth", "record", "strong"])
    neg = sum(w in txt for w in ["miss", "fall", "down", "slow", "weak", "probe"])
    return float(pos - neg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score headlines -> JSON items")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--news-root", default="output/news")
    ap.add_argument("--out-root", default="output/sentiment")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    news_path = Path(args.news_root) / args.ticker / "news.jsonl"
    items = _load_jsonl(news_path)[: args.limit]

    out_dir = Path(args.out_root) / args.ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, it in enumerate(items):
        title = str(it.get("title", "")).strip()
        score = _score_summary(title or str(it.get("summary", "")))
        sentiment = "NEUTRAL"
        if score > 0:
            sentiment = "POSITIVE"
        elif score < 0:
            sentiment = "NEGATIVE"

        out = {
            "title": title,
            "link": it.get("link") or it.get("url"),
            "published": it.get("published"),
            "score": score,
            "sentiment": sentiment,
        }
        (out_dir / f"{i:03d}.json").write_text(json.dumps(out, ensure_ascii=False))
        written += 1

    print(f"Wrote {written} sentiment files to {out_dir}")


if __name__ == "__main__":
    main()
