# src/agents/fundamental.py

import argparse
import json
import os
from datetime import datetime, timezone

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output")
import pandas as pd

# Allow tests or environment to override the data root
DATA_ROOT = os.environ.get("DATA_ROOT", "data")

def run_fundamental(ticker: str):
    """
    Read fundamentals.csv from DATA_ROOT, compute YoY revenue growth and a simple rating.
    Returns a dict with the latest metrics + growth + rating.
    """
    path = os.path.join(DATA_ROOT, ticker, "fundamentals.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()

    latest = df.iloc[-1]
    prior  = df.iloc[-5] if len(df) >= 5 else df.iloc[0]

    # Use "Total Revenue" if present, otherwise first column
    rev_col = "Total Revenue" if "Total Revenue" in df.columns else df.columns[0]
    growth = (latest[rev_col] - prior[rev_col]) / abs(prior[rev_col])
    rating = "Strong" if growth > 0.1 else "Weak" if growth < 0 else "Neutral"

    return {
        "ticker":        ticker,
        "latest_quarter": latest.name.strftime("%Y-%m-%d"),
        "revenue":       float(latest[rev_col]),
        "revenue_growth": float(growth),
        "rating":        rating
    }

def main():
    p = argparse.ArgumentParser(description="Fundamental Analyst agent")
    p.add_argument("--ticker", required=True, help="Stock ticker symbol")
    args = p.parse_args()

    out = run_fundamental(args.ticker)
    out_dir = os.path.join(OUTPUT_ROOT, "fundamental", args.ticker)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(out_dir, f"fundamental_{ts}.json")

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Fundamental] Wrote metrics to {out_path}")

if __name__ == "__main__":
    main()
