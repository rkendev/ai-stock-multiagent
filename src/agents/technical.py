# src/agents/technical.py

import argparse
import json
import os
from datetime import datetime

import pandas as pd

# Allow tests or environment to override the data directory
DATA_DIR = os.environ.get("DATA_DIR", "data")

def run_technical(ticker: str):
    """
    Read prices.parquet from DATA_DIR, compute 50-day MA and 14-day RSI,
    then flag overbought. Returns a dict of metrics.
    """
    path = os.path.join(DATA_DIR, ticker, "prices.parquet")
    df = pd.read_parquet(path).sort_index()

    # If multi-indexed columns, flatten to first level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1) 50-day moving average
    df["ma50"] = df["Close"].rolling(window=50).mean()

    # 2) RSI(14) via Wilder’s smoothing
    delta     = df["Close"].diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss  = loss.ewm(alpha=1/14, adjust=False).mean()
    rs        = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 3) Drop rows missing either indicator
    valid = df.dropna(subset=["ma50", "rsi"])
    if valid.empty:
        raise ValueError(f"No valid data to compute technical indicators for {ticker}")

    latest = valid.iloc[-1]
    return {
        "ticker":     ticker,
        "date":       str(latest.name.date()),
        "close":      float(latest["Close"]),
        "rsi":        float(latest["rsi"]),
        "ma50":       float(latest["ma50"]),
        "overbought": bool(latest["rsi"] > 70)
    }

def main():
    p = argparse.ArgumentParser(description="Technical Analyst agent")
    p.add_argument("--ticker", required=True, help="Stock ticker symbol")
    args = p.parse_args()

    out = run_technical(args.ticker)
    out_dir = os.path.join("output", "technical", args.ticker)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"technical_{ts}.json")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Technical] Wrote metrics to {out_path}")

if __name__ == "__main__":
    main()
