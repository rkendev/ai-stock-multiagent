# src/tests/test_fundamental_agent.py
import json
from pathlib import Path

import pytest

from agents.fundamental import run_fundamental

@pytest.mark.parametrize("ticker", ["AAPL"])
def test_run_fundamental(tmp_path, ticker):
    # Write a small CSV to data/<ticker>/fundamentals.csv
    sample = tmp_path / "data" / ticker
    sample.mkdir(parents=True)
    csv = sample / "fundamentals.csv"
    csv.write_text(
        "Date,Total Revenue\n"
        "2023-03-31,100\n"
        "2023-06-30,110\n"
        "2023-09-30,120\n"
        "2023-12-31,130\n"
        "2024-03-31,140\n"
    )
    # Point the code at tmp_path
    import os
    os.environ["DATA_ROOT"] = str(tmp_path / "data")
    # Monkey-patch inside the agent to respect DATA_ROOT
    from agents import fundamental
    fundamental.DATA_ROOT = os.environ["DATA_ROOT"]
    
    res = run_fundamental(ticker)
    assert res["revenue_growth"] == pytest.approx((140 - 100) / 100)
    assert res["rating"] == "Strong"
