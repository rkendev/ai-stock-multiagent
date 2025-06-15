# src/tests/test_researcher.py
import json
from pathlib import Path

import pytest

from agents.researcher import run_researcher

@pytest.mark.parametrize("ticker,limit", [
    ("AAPL", 3),
])
def test_researcher_fetch(tmp_path, ticker, limit):
    items = run_researcher(ticker, limit)
    assert isinstance(items, list)
    assert len(items) == limit
    for it in items:
        assert set(it.keys()) == {"title", "link", "published", "summary"}

    # write & read back
    out = tmp_path / "res.json"
    out.write_text(json.dumps(items))
    assert json.loads(out.read_text()) == items
