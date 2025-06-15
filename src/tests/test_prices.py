# src/tests/test_prices.py
import os
import pandas as pd
import pytest

from ingest.prices import fetch_prices

@pytest.mark.parametrize("ticker,since", [
    ("AAPL", "2024-01-01"),
])
def test_fetch_prices_and_write(tmp_path, ticker, since):
    # Fetch a small slice of data
    df = fetch_prices(ticker, since)
    assert not df.empty
    # Write out and read back
    out = tmp_path / "prices.parquet"
    df.to_parquet(out)
    df2 = pd.read_parquet(out)
    assert list(df2.columns) == list(df.columns)
