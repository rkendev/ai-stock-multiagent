# src/agents/visualizer.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, List
import json

# Headless backend for CI/servers
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_prices(prices_path: Path) -> pd.DataFrame:
    """
    Load price data from Parquet (index is date, must have 'Close').
    Compute a 50-day moving average (MA50).
    """
    df = pd.read_parquet(prices_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Flatten MultiIndex columns if present (keep first level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError("prices.parquet must contain a 'Close' column")

    # 50-DMA; allow shorter series in tests via min_periods=1
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    return df


def _plot_price_ma50(df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df.index, df["Close"], label="Close")   # default styles/colors
    ax.plot(df.index, df["MA50"], label="MA50")
    ax.set_title("Price & 50-DMA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _load_sentiment(sent_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load minimal sentiment time series from the latest JSON file in sent_dir.
    Items format: [{ "published": "YYYY-MM-DD", "score": float }, ...]
    Returns DataFrame indexed by date with 'score' column, or None if missing.
    """
    if not sent_dir.exists():
        return None

    json_files = sorted(sent_dir.glob("sentiment_*.json"))
    if not json_files:
        return None

    latest = json_files[-1]
    with latest.open("r", encoding="utf-8") as f:
        try:
            items = json.load(f)
        except json.JSONDecodeError:
            return None

    if not items:
        return None

    rows: List[tuple[pd.Timestamp, float]] = []
    for it in items:
        pub = it.get("published")
        score = it.get("score")
        if pub is None or score is None:
            continue
        try:
            rows.append((pd.to_datetime(pub), float(score)))
        except Exception:
            continue

    if not rows:
        return None

    s_df = pd.DataFrame(rows, columns=["date", "score"]).set_index("date").sort_index()
    return s_df


def _plot_sentiment(s_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(s_df.index, s_df["score"], label="sentiment")  # default color
    ax.set_title("Sentiment Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _auto_detect_ticker() -> Optional[str]:
    """
    If exactly one ticker directory under data/ contains a prices.parquet,
    return that ticker (dir name). Otherwise, return None.
    """
    data_root = Path("data")
    if not data_root.exists():
        return None
    candidates = []
    for sub in data_root.iterdir():
        if sub.is_dir() and (sub / "prices.parquet").exists():
            candidates.append(sub.name)
    if len(candidates) == 1:
        return candidates[0]
    return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Visualizer agent (charts -> output/visuals/<TICKER>)"
    )
    ap.add_argument("--ticker", required=False, help="Ticker symbol (e.g., AAPL)")
    return ap


def main(argv: Optional[list[str]] = None) -> None:
    """
    Entry point.

    NOTE: default argv to [] so pytest flags (e.g. -q) aren't parsed here
    when called as a function in tests.
    """
    if argv is None:
        argv = []  # avoid inheriting pytest's sys.argv

    ap = build_parser()
    args = ap.parse_args(argv)

    # Resolve ticker in this order:
    # 1) --ticker argument
    # 2) auto-detect single ticker under data/*/prices.parquet
    # 3) TICKER env var
    # 4) fallback "AAPL"
    ticker = (args.ticker or _auto_detect_ticker() or os.environ.get("TICKER") or "AAPL").upper()

    # Inputs match repository layout
    prices_path = Path("data") / ticker / "prices.parquet"
    sentiment_dir = Path("output") / "sentiment" / ticker

    # Outputs
    out_dir = Path("output") / "visuals" / ticker
    _mkdirp(out_dir)

    # Price / MA50 chart
    if prices_path.exists():
        df = _load_prices(prices_path)
        _plot_price_ma50(df, out_dir / "price_ma50.png")

    # Sentiment chart (if present)
    s_df = _load_sentiment(sentiment_dir)
    if s_df is not None and not s_df.empty:
        _plot_sentiment(s_df, out_dir / "sentiment.png")


if __name__ == "__main__":
    # When running as a script, accept real CLI args.
    main(None)
