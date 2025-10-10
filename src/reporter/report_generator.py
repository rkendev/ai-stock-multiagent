# src/reporter/report_generator.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

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


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


# Back-compat for tests that import this symbol
def _load_technical(path: Path) -> Dict[str, Any]:
    return _load_json(path)


# in src/reporter/report_generator.py
def _mk_technical_summary(ticker: str, asof: str, tech: Dict[str, Any], prices: pd.DataFrame) -> str:
    sig = tech.get("signals", {})
    close = float(prices["Close"].iloc[-1])

    # base symbol (e.g., ASML from ASML.AS) to match test expectations
    base = ticker.split(".")[0]

    # 30-trading-day change (or as far back as available)
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
        f"**{base} Technical Snapshot** *(as of {asof})*",
        f"- Price: ~{close:,.2f} | 30-day change: {pct_30d:+.1f}%",
        f"- Trend: {', '.join(trend_bits)}",
        f"- Momentum (RSI14): {rsi_state}",
        f"- Volatility: {vol_note}",
        f"- **Overall bias:** {bias}",
    ]
    return "\n".join(lines)



def _load_fundamental_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.exists() else None


def _mk_fundamental_summary(ticker: str, fund: dict) -> str:
    sig = (fund or {}).get("signals", {})
    rev_pos = sig.get("rev_yoy_positive")
    eps_pos = sig.get("eps_yoy_positive")
    margin_imp = sig.get("margin_improving")
    bits = []
    bits.append("YoY revenue growth positive" if rev_pos else "YoY revenue growth not confirmed")
    bits.append("YoY earnings growth positive" if eps_pos else "YoY earnings growth not confirmed")
    bits.append("margin improving" if margin_imp else "margin not improving")
    return "**Fundamental Snapshot**\n- " + "\n- ".join(bits)


def _write_report_md(
    ticker: str,
    out_dir: Path,
    sections: Dict[str, str],
    has_fundamentals: bool = False,  # keep default for back-compat
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.md"

    # CHANGE THIS LINE ↓
    content = [f"# {ticker} — MVP Technical Report", ""]

    content.append(sections["technical"])
    if "fundamental" in sections:
        content += ["", sections["fundamental"]]

    note_tail = "price/technical" + (" + fundamental" if has_fundamentals else "")
    content += ["", f"_Note: MVP report generated from {note_tail} data._"]

    out_path.write_text("\n".join(content))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MVP report from technical.json (+ optional fundamental.json) and prices.parquet")
    ap.add_argument("--ticker", required=True, help="Ticker (directory under data/)")
    ap.add_argument("--data-dir", default="data", help="Input root")
    ap.add_argument("--out-dir", default="output", help="Output root (writes output/<ticker>/report.md)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) / args.ticker
    prices_path = data_dir / "prices.parquet"
    tech_path = data_dir / "technical.json"
    fund_path = data_dir / "fundamental.json"

    if not prices_path.exists():
        raise FileNotFoundError(f"Missing prices: {prices_path}")
    if not tech_path.exists():
        raise FileNotFoundError(f"Missing technical: {tech_path}")

    prices = _load_prices(prices_path)
    tech = _load_technical(tech_path)

    asof = tech.get("asof", str(prices.index[-1].date()) if len(prices) else "N/A")
    technical_md = _mk_technical_summary(args.ticker, asof, tech, prices)

    fund = _load_fundamental_json(fund_path)
    sections: Dict[str, str] = {"technical": technical_md}
    if fund:
        sections["fundamental"] = _mk_fundamental_summary(args.ticker, fund)

    out = Path(args.out_dir) / args.ticker
    out_path = _write_report_md(args.ticker, out, sections, has_fundamentals=bool(fund))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
