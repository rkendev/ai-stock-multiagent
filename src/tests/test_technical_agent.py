# src/tests/test_technical_agent.py
import json
from pathlib import Path

import pytest

from agents.technical import run_technical

@pytest.mark.parametrize("ticker", ["TEST"])
def test_run_technical(tmp_path, ticker):
    # Create a fake prices.parquet
    data_dir = tmp_path / "data" / ticker
    data_dir.mkdir(parents=True)
    import pandas as pd
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "Close": list(range(100, 160))
    }, index=dates)
    df.to_parquet(data_dir / "prices.parquet")

    # Monkey-patch DATA_ROOT if needed, else assume cwd/data
    from agents import technical
    technical.DATA_DIR = str(tmp_path / "data")

    res = run_technical(ticker)
    assert "rsi" in res and "ma50" in res and isinstance(res["overbought"], bool)
