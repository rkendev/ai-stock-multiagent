from __future__ import annotations
from pathlib import Path
import json

from reporter.report_generator import _mk_sentiment_summary, _load_sentiment_items

def test_sentiment_section_renders(tmp_path: Path):
    t = "TEST"
    sdir = tmp_path / "output" / "sentiment" / t
    sdir.mkdir(parents=True, exist_ok=True)

    # 3 items: pos, neg, neutral
    (sdir / "0001.json").write_text(json.dumps({"score": 0.5, "sentiment": "POSITIVE"}))
    (sdir / "0002.json").write_text(json.dumps({"score": -0.5, "sentiment": "NEGATIVE"}))
    (sdir / "0003.json").write_text(json.dumps({"score": 0.0, "sentiment": "NEUTRAL"}))

    items = _load_sentiment_items(sdir)
    md = _mk_sentiment_summary(items)
    assert md is not None
    assert "Sentiment Snapshot" in md
    assert "Headlines analyzed: 3" in md
