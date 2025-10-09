# src/tests/test_reporter_agent.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reporter import report_generator as rg


def _write_minimal_prices(parquet_path: Path) -> None:
    """Create a tiny Parquet file with a DatetimeIndex and a Close column."""
    dates = pd.date_range("2025-09-15", periods=10, freq="B")
    close = pd.Series([100 + i for i in range(len(dates))], index=dates, name="Close")
    df = pd.DataFrame({"Close": close})
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path)


def _write_minimal_technical(json_path: Path, asof: str) -> None:
    """Create a minimal technical.json with signals expected by the reporter."""
    payload = {
        "ticker": "TEST",
        "asof": asof,
        "indicators": {
            "MA20": [],
            "MA50": [],
            "MA200": [],
            "RSI14": [],
            "volatility_20d": [],
        },
        "signals": {
            "above_MA50": True,
            "above_MA200": True,
            "overbought": False,
            "oversold": False,
            "rally_volatility_high": False,
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))


def test_report_generator_end_to_end(tmp_path: Path) -> None:
    ticker = "ASML.AS"  # any symbol is fine; paths are synthetic here

    # Arrange: create data/<ticker>/{prices.parquet, technical.json}
    data_dir = tmp_path / "data" / ticker
    prices_path = data_dir / "prices.parquet"
    _write_minimal_prices(prices_path)

    # Use the last date in prices as the asof
    asof = pd.read_parquet(prices_path).index[-1].date().isoformat()
    tech_path = data_dir / "technical.json"
    _write_minimal_technical(tech_path, asof)

    # Act: load inputs, build technical section, write markdown report
    prices = rg._load_prices(prices_path)
    tech = rg._load_technical(tech_path)
    technical_md = rg._mk_technical_summary(ticker, tech.get("asof", asof), tech, prices)

    out_dir = tmp_path / "output" / ticker
    report_path = rg._write_report_md(ticker, out_dir, {"technical": technical_md})

    # Assert: report exists and contains expected sections/phrases
    assert report_path.exists(), "report.md was not created"
    text = report_path.read_text()
    assert text.startswith(f"# {ticker} — MVP Technical Report")
    assert "**ASML Technical Snapshot**" in text or "**ASML Technical Snapshot**".replace("ASML", ticker.split(".")[0]) in text
    assert "Overall bias" in text or "Overall bias" in text.replace("**", "")
