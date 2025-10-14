from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

# Optional dependency (works offline if you already have the parquet)
import yfinance as yf


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    # Handle epoch-like or object index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=False)
        except Exception:
            pass
    # Sort and drop tz if present
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    return df


def _rename_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Normalize casing
    lower_map = {c.lower(): c for c in df.columns}
    # Prefer Yahoo-style names if present
    if "close" in lower_map and "Close" not in df.columns:
        df = df.rename(columns={lower_map["close"]: "Close"})
    return df


def _compute_tech(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("prices DataFrame must contain a 'Close' column")
    out = df.copy()
    out["MA50"] = out["Close"].rolling(50, min_periods=1).mean()
    out["MA200"] = out["Close"].rolling(200, min_periods=1).mean()
    return out


def fetch_prices(ticker: str, since: Optional[str]) -> pd.DataFrame:
    """
    Fetch prices with yfinance (if available) or raise a clear error.
    Returns a DataFrame with index as DatetimeIndex and columns at least ['Close','MA50','MA200'].
    """
    if yf is None:
        raise RuntimeError(
            "yfinance not installed. Either install it or make sure data/<TICKER>/prices.parquet exists."
        )
    df = yf.download(ticker, start=since or None, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    df = _ensure_datetime_index(df)
    df = _rename_common_columns(df)
    df = _compute_tech(df)
    return df


def write_parquet(df: pd.DataFrame, out_root: Path, ticker: str) -> Path:
    out_dir = out_root / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prices.parquet"
    df.to_parquet(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch prices and write data/<TICKER>/prices.parquet")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--since", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--out-root", default="data")
    args = ap.parse_args()

    df = fetch_prices(args.ticker, args.since)
    out_path = write_parquet(df, Path(args.out_root), args.ticker)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
