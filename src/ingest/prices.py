# src/ingest/prices.py
import argparse
import os

import yfinance as yf
import pandas as pd

def fetch_prices(ticker: str, since: str) -> pd.DataFrame:
    df = yf.download(ticker, start=since, progress=False, auto_adjust=True)
    return df

def main():
    p = argparse.ArgumentParser(description="Download OHLCV data for a ticker")
    p.add_argument("--ticker", required=True, help="Stock ticker symbol")
    p.add_argument("--since",  required=True, help="Start date (YYYY-MM-DD)")
    args = p.parse_args()

    out_dir = os.path.join("data", args.ticker)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prices.parquet")
    df = fetch_prices(args.ticker, args.since)
    df.to_parquet(out_path)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
