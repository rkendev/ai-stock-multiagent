# src/agents/visualizer.py
from __future__ import annotations

import argparse
from pathlib import Path

# headless plotting for CI/servers
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_prices(prices_path: Path) -> pd.DataFrame:
    """
    Load price data from Parquet and compute a 50-day moving average (MA50).
    Requires a 'Close' column.
    """
    df = pd.read_parquet(prices_path)

    # Keep sorted index if datetime
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError("Parquet must include a 'Close' column")

    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    return df


def viz_main(
    ticker: str = "TEST",
    data_dir: str = "data",
    out_dir: str = "output",
) -> Path:
    """
    Programmatic entrypoint (used by tests).
    Looks for: data/<ticker>/prices.parquet
    Writes:    output/<ticker>/charts/price_ma50.png
    Returns path to the PNG.
    """
    prices_path = Path(data_dir) / ticker / "prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"Could not find prices Parquet at: {prices_path}")

    charts_dir = Path(out_dir) / ticker / "charts"
    _mkdirp(charts_dir)

    df = _load_prices(prices_path)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["MA50"], label="MA50")
    ax.set_title(f"{ticker} — Close & 50-DMA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()

    png = charts_dir / "price_ma50.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {png}")
    return png


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize Close & 50-DMA from prices.parquet")
    p.add_argument("--ticker", required=True, help="Ticker symbol directory under data/, e.g. ASML.AS or AAPL")
    p.add_argument("--data-dir", default="data", help="Root directory containing <ticker>/prices.parquet")
    p.add_argument("--out-dir", default="output", help="Root output directory")
    args = p.parse_args()

    viz_main(ticker=args.ticker, data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
