from __future__ import annotations
from pathlib import Path
import json

from agents.sentiment import run_sentiment

def test_run_sentiment_writes_jsons(tmp_path: Path):
    ticker = "TEST"
    news_dir = tmp_path / "output" / "news" / ticker
    news_dir.mkdir(parents=True, exist_ok=True)
    sample = [
        {"published": "2025-10-08T10:00:00Z", "title": "TEST posts record growth and strong profit", "url": "u1"},
        {"published": "2025-10-08T12:00:00Z", "title": "TEST faces lawsuit and delay", "url": "u2"},
        {"published": "2025-10-08T15:00:00Z", "title": "Neutral headline no keywords", "url": "u3"},
    ]
    news_path = news_dir / "news.jsonl"
    with news_path.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")

    out_dir, n = run_sentiment(ticker, news_root=str(tmp_path / "output" / "news"),
                               out_root=str(tmp_path / "output" / "sentiment"))
    assert n == 3
    # confirm files and fields
    files = sorted(Path(out_dir).glob("*.json"))
    assert len(files) == 3
    j = json.loads(files[0].read_text())
    assert "published" in j and "score" in j
