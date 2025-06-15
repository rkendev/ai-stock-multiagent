# src/tests/test_reporter_agent.py

import json
import os
from pathlib import Path

import pytest

from agents.reporter import ReporterAgent, BaseReporter

class DummyReporter(BaseReporter):
    def __init__(self):
        # Skip OpenAI key requirement
        self.api_key = "DUMMY"
    def generate_report(self, context: dict) -> str:
        # ensure all keys present
        assert set(context.keys()) == {"fundamental","technical","sentiment","researcher"}
        # return a simple confirmation string
        return "## Dummy Report\nAll agents present."

@pytest.fixture(autouse=True)
def isolate_output(tmp_path, monkeypatch):
    # Redirect all output/ reads to a fresh tmp_path / data structure
    out = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_ROOT", str(out))
    return out

def write_json(dirpath: Path, filename: str, data):
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / filename, "w", encoding="utf8") as f:
        json.dump(data, f)

def test_reporter_end_to_end(isolate_output):
    out_root = isolate_output
    ticker = "T"

    # Prepare fake outputs for each agent
    write_json(out_root / "fundamental" / ticker, "f.json",   {"ticker": ticker})
    write_json(out_root / "technical" / ticker,   "t.json",   {"ticker": ticker})
    write_json(out_root / "sentiment" / ticker,   "s.json",   [{"title":"X"}])
    write_json(out_root / "researcher" / ticker,  "r.json",   [{"title":"Y"}])

    # Run reporter with dummy
    agent = ReporterAgent(DummyReporter())
    report_path = agent.run(ticker)

    # Verify output file exists and contents match dummy
    txt = Path(report_path).read_text()
    assert txt.startswith("## Dummy Report")

    # Also ensure it's under OUTPUT_ROOT/reporter/<ticker>
    assert Path(report_path).parents[1].name == ticker
