# src/ingest/fundamentals.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf


def fetch_fundamentals(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        stmt = t.income_stmt
        if stmt is None or stmt.empty:
            return pd.DataFrame(columns=["Date", "Revenue", "Earnings"])
        df = stmt.T.reset_index().rename(columns={"index": "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[["Date", "Total Revenue", "Net Income"]].rename(
            columns={"Total Revenue": "Revenue", "Net Income": "Earnings"}
        )
        return df.dropna(subset=["Date"])
    except Exception:
        return pd.DataFrame(columns=["Date", "Revenue", "Earnings"])


def write_parquet_fundamentals(ticker: str, out_root: str = "data") -> str:
    df = fetch_fundamentals(ticker)
    out_dir = Path(out_root) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fundamentals.parquet"
    df.to_parquet(out_path)
    return str(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch & write quarterly fundamentals parquet")
    p.add_argument("--ticker", required=True)
    p.add_argument("--out-root", default="data")
    args = p.parse_args()
    out = write_parquet_fundamentals(args.ticker, out_root=args.out_root)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
