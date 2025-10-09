# src/agents/visualizer.py
from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Optional

# Headless plotting backend (CI/server safe)
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = ["viz_main"]


def _mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_prices(prices_path: Path) -> pd.DataFrame:
    """Load price data and compute 50-day moving average (MA50)."""
    df = pd.read_parquet(prices_path)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError("Parquet must include a 'Close' column")

    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    return df


def _load_sentiment(ticker: str, sentiment_root: Path) -> Optional[pd.DataFrame]:
    """
    Load sentiment points from JSON files:
      sentiment_root/<ticker>/*.json
    Each file contains a list of { "published": "YYYY-MM-DD", "score": float } items.
    Returns a DataFrame with columns ['published', 'score'] sorted by date, or None if no files.
    """
    sdir = sentiment_root / ticker
    files = sorted(glob(str(sdir / "*.json")))
    if not files:
        return None

    rows = []
    for fp in files:
        try:
            data = json.loads(Path(fp).read_text())
            if isinstance(data, dict):  # tolerate dict payloads
                data = [data]
            for item in data:
                pub = item.get("published")
                sc = item.get("score")
                if pub is not None and sc is not None:
                    rows.append((pub, float(sc)))
        except Exception:
            # Be robust: skip unreadable files
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["published", "score"])
    # Parse dates; coerce errors to NaT and drop them
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"]).sort_values("published").reset_index(drop=True)
    return df if not df.empty else None


def _plot_price_ma50(df: pd.DataFrame, ticker: str, out_png: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["MA50"], label="MA50")
    ax.set_title(f"{ticker} — Close & 50-DMA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def _plot_sentiment(df: Optional[pd.DataFrame], ticker: str, out_png: Path) -> None:
    """
    Plot sentiment score over time. If df is None (no sentiment available),
    still emit an empty placeholder chart so the file exists (tests expect it).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if df is not None and not df.empty:
        ax.plot(df["published"], df["score"], marker="o", linestyle="-", label="Sentiment")
    else:
        # Placeholder baseline if no data
        ax.plot([], [], label="Sentiment")

    ax.set_title(f"{ticker} — Sentiment Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.legend()
    fig.autofmt_xdate()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def viz_main(
    ticker: str = "TEST",
    data_dir: str = "data",
    out_dir: str = "output/visuals",
) -> Path:
    """
    Reads:
      - prices:   data/<ticker>/prices.parquet
      - sentiment: output/sentiment/<ticker>/*.json  (fixed path expected by tests)
    Writes (under out_dir/<ticker>/):
      - price_ma50.png
      - sentiment.png
    Returns: path to price_ma50.png (primary artifact)
    """
    prices_path = Path(data_dir) / ticker / "prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"Could not find prices Parquet at: {prices_path}")

    target_dir = Path(out_dir) / ticker
    _mkdirp(target_dir)

    # Price chart
    price_df = _load_prices(prices_path)
    price_png = target_dir / "price_ma50.png"
    _plot_price_ma50(price_df, ticker, price_png)

    # Sentiment chart (input fixed at output/sentiment/<ticker>/*.json)
    sentiment_root = Path("output") / "sentiment"
    sent_df = _load_sentiment(ticker, sentiment_root)
    sent_png = target_dir / "sentiment.png"
    _plot_sentiment(sent_df, ticker, sent_png)

    return price_png


def main() -> None:
    # parse_known_args so pytest flags don't crash the CLI if invoked during tests
    parser = argparse.ArgumentParser(description="Visualize price (Close & MA50) and sentiment.")
    parser.add_argument("--ticker", default="TEST", help="Ticker symbol directory under data/, e.g. ASML.AS or AAPL")
    parser.add_argument("--data-dir", default="data", help="Root directory containing <ticker>/prices.parquet")
    parser.add_argument("--out-dir", default="output/visuals", help="Output root (default: output/visuals)")
    args, _ = parser.parse_known_args()

    viz_main(ticker=args.ticker, data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
