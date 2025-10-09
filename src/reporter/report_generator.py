# src/reporter/report_generator.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def _load_prices(prices_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(prices_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise ValueError("Parquet must include a 'Close' column")
    return df


def _load_technical(tech_path: Path) -> Dict[str, Any]:
    return json.loads(tech_path.read_text())


def _mk_technical_summary(ticker: str, asof: str, tech: Dict[str, Any], prices: pd.DataFrame) -> str:
    sig = tech.get("signals", {})
    close = float(prices["Close"].iloc[-1])

    lookback = min(30, len(prices) - 1) if len(prices) > 1 else 1
    pct_30d = (close / float(prices["Close"].iloc[-lookback]) - 1.0) * 100.0

    trend_bits = []
    if sig.get("above_MA200"):
        trend_bits.append("above its 200-DMA (long-term uptrend intact)")
    elif sig.get("above_MA50"):
        trend_bits.append("above its 50-DMA (short-term strength)")
    else:
        trend_bits.append("below key moving averages (trend weak)")

    rsi_state = "neutral"
    if sig.get("overbought"):
        rsi_state = "overbought (>70)"
    elif sig.get("oversold"):
        rsi_state = "oversold (<30)"

    vol_note = "elevated" if sig.get("rally_volatility_high") else "normal"

    bias = "neutral"
    if sig.get("above_MA200") and not sig.get("overbought"):
        bias = "constructive"
    if not sig.get("above_MA50") and not sig.get("above_MA200"):
        bias = "cautious"

    lines = [
        f"**ASML Technical Snapshot** *(as of {asof})*",
        f"- Price: ~{close:,.2f} | 30-day change: {pct_30d:+.1f}%",
        f"- Trend: {', '.join(trend_bits)}",
        f"- Momentum (RSI14): {rsi_state}",
        f"- Volatility: {vol_note}",
        f"- **Overall bias:** {bias}",
    ]
    return "\n".join(lines)


def _write_report_md(ticker: str, out_dir: Path, sections: Dict[str, str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.md"
    content = [
        f"# {ticker} — MVP Technical Report",
        "",
        sections["technical"],
        "",
        "_Note: MVP report generated from price/technical data only._",
    ]
    out_path.write_text("\n".join(content))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MVP report from technical.json and prices.parquet")
    ap.add_argument("--ticker", required=True, help="Ticker (directory under data/)")
    ap.add_argument("--data-dir", default="data", help="Input root")
    ap.add_argument("--out-dir", default="output", help="Output root (writes output/<ticker>/report.md)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) / args.ticker
    prices_path = data_dir / "prices.parquet"
    tech_path = data_dir / "technical.json"

    if not prices_path.exists():
        raise FileNotFoundError(f"Missing prices: {prices_path}")
    if not tech_path.exists():
        raise FileNotFoundError(f"Missing technical: {tech_path}")

    prices = _load_prices(prices_path)
    tech = _load_technical(tech_path)

    asof = tech.get("asof", str(prices.index[-1].date()) if len(prices) else "N/A")
    technical_md = _mk_technical_summary(args.ticker, asof, tech, prices)

    out = Path(args.out_dir) / args.ticker
    out_path = _write_report_md(args.ticker, out, {"technical": technical_md})
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
