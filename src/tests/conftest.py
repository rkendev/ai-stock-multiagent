import os
import pandas as pd
import pytest

from ingest.prices import fetch_prices
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.mark.unit
@pytest.mark.parametrize("ticker,since", [
    ("AAPL", "2024-01-01"),
])
def test_fetch_prices_and_write(tmp_path, monkeypatch, ticker, since):
    # Mock yfinance to avoid network
    import types

    def fake_download(sym, start=None, progress=False):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "Open":  [10, 11, 12, 13, 14],
                "High":  [11, 12, 13, 14, 15],
                "Low":   [ 9, 10, 11, 12, 13],
                "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
                "Volume":[100, 110, 120, 130, 140],
            },
            index=idx,
        )
        df.index.name = "Date"
        return df

    import ingest.prices as prices_mod
    yf = types.SimpleNamespace(download=fake_download)
    monkeypatch.setattr(prices_mod, "yf", yf)

    # Ensure data is written under tmp_path
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))

    df = fetch_prices(ticker, since)
    assert not df.empty

    out = tmp_path / "prices.parquet"
    df.to_parquet(out)
    df2 = pd.read_parquet(out)
    assert list(df2.columns) == list(df.columns)
