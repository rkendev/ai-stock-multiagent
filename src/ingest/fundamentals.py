# src/ingest/fundamentals.py
import argparse
import os

import yfinance as yf
import pandas as pd

def fetch_fundamentals(ticker: str) -> pd.DataFrame:
    # Pull quarterly financials (income statement, balance sheet, cashflow)
    tk = yf.Ticker(ticker)
    df = tk.quarterly_financials.T  # transpose for date-based index
    return df

def main():
    p = argparse.ArgumentParser(description="Download quarterly fundamentals for a ticker")
    p.add_argument("--ticker", required=True, help="Stock ticker symbol")
    args = p.parse_args()

    out_dir = os.path.join("data", args.ticker)
    os.makedirs(out_dir, exist_ok=True)

    df = fetch_fundamentals(args.ticker)
    out_path = os.path.join(out_dir, "fundamentals.csv")
    df.to_csv(out_path)
    print(f"Wrote {df.shape[0]} quarters × {df.shape[1]} metrics to {out_path}")

if __name__ == "__main__":
    main()
