# src/tests/test_fundamentals.py
import pandas as pd
import pytest

from ingest.fundamentals import fetch_fundamentals

@pytest.mark.parametrize("ticker", ["AAPL"])
def test_fetch_fundamentals(tmp_path, ticker):
    df = fetch_fundamentals(ticker)
    assert not df.empty
    # Write and read back
    out = tmp_path / "fund.csv"
    df.to_csv(out)
    df2 = pd.read_csv(out, index_col=0)
    # Check that every column in df appears in df2
    assert set(df.columns) == set(df2.columns)
