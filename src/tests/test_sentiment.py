# src/tests/test_sentiment.py
import json
from pathlib import Path

import pytest

from agents.sentiment import run_sentiment

@pytest.mark.parametrize("ticker,limit", [
    ("AAPL", 3),
])
def test_run_sentiment(tmp_path, ticker, limit):
    items = run_sentiment(ticker, limit)
    assert isinstance(items, list)
    assert len(items) == limit
    for it in items:
        assert {"title","link","published","summary","sentiment","score"}.issubset(it.keys())
        assert it["sentiment"] in {"POSITIVE","NEGATIVE","NEUTRAL"}
        assert isinstance(it["score"], float)

    # write & read back
    out = tmp_path / "sent.json"
    out.write_text(json.dumps(items))
    assert json.loads(out.read_text()) == items
