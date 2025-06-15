# src/tests/test_news.py
import json
from pathlib import Path

import pytest

from ingest.news import fetch_rss_news

@pytest.mark.parametrize("ticker,limit", [
    ("AAPL", 5),
])
def test_fetch_rss_news_and_write(tmp_path, ticker, limit):
    items = fetch_rss_news(ticker, limit)
    assert isinstance(items, list)
    assert len(items) == limit
    for item in items:
        # each item should have these keys
        assert set(item.keys()) == {"title", "link", "published", "summary"}

    # write to temp file and read back
    out = tmp_path / "news.json"
    out.write_text(json.dumps(items))
    items2 = json.loads(out.read_text())
    assert items2 == items
