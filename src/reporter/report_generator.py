# src/reporter/report_generator.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

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


def _mk_technical_summary(ticker: str, asof: str, tech: Dict[str, Any], prices: pd.DataFrame) -> str:
    sig = tech.get("signals", {})
    close = float(prices["Close"].iloc[-1])

    base = ticker.split(".")[0]
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


# -------- Sentiment (new) --------

def _load_sentiment_items(sent_dir: Path) -> List[Dict[str, Any]]:
    """Load sentiment items from output/sentiment/<ticker>/*.json."""
    if not sent_dir.exists():
        return []
    items: List[Dict[str, Any]] = []
    for p in sorted(sent_dir.glob("*.json")):
        try:
            items.append(json.loads(p.read_text()))
        except Exception:
            continue
    return items


def _split_top_headlines(
    items: List[Dict[str, Any]], k: int = 3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return (top_positive, top_negative) by score.
    Each item may have: title, link/url, score, sentiment.
    """
    def _score(it: Dict[str, Any]) -> float:
        try:
            return float(it.get("score", 0.0))
        except Exception:
            return 0.0

    pos = [it for it in items if str(it.get("sentiment", "")).upper() == "POSITIVE"]
    neg = [it for it in items if str(it.get("sentiment", "")).upper() == "NEGATIVE"]
    pos.sort(key=_score, reverse=True)
    neg.sort(key=_score)  # most negative first (lowest score)
    return pos[:k], neg[:k]


def _mk_sentiment_summary(items: List[Dict[str, Any]]) -> Optional[str]:
    if not items:
        return None

    scores = []
    for i in items:
        try:
            scores.append(float(i.get("score", 0.0)))
        except Exception:
            pass
    n = len(items)
    avg = sum(scores) / len(scores) if scores else 0.0

    def pct(label: str) -> float:
        return 100.0 * sum(1 for it in items if str(it.get("sentiment", "")).upper() == label) / n

    pos = pct("POSITIVE"); neu = pct("NEUTRAL"); neg = pct("NEGATIVE")

    # top headlines
    top_pos, top_neg = _split_top_headlines(items, k=3)

    def _fmt(it: Dict[str, Any]) -> str:
        title = str(it.get("title", "")).strip()
        link = (it.get("link") or it.get("url") or "").strip()
        score = 0.0
        try:
            score = float(it.get("score", 0.0))
        except Exception:
            pass
        base = f'  - {title} *(score {score:+.2f})*'
        return f"{base} — {link}" if link else base

    lines = [
        "**Sentiment Snapshot**",
        f"- Headlines analyzed: {n}",
        f"- Average score: {avg:+.2f}",
        f"- Mix: {pos:.0f}% POSITIVE, {neu:.0f}% NEUTRAL, {neg:.0f}% NEGATIVE",
    ]

    if top_pos:
        lines.append("")
        lines.append("Top positive headlines:")
        lines += [_fmt(it) for it in top_pos]

    if top_neg:
        lines.append("")
        lines.append("Top negative headlines:")
        lines += [_fmt(it) for it in top_neg]

    return "\n".join(lines)


# ---------------------------------


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
    has_fundamentals: bool = False,
    has_sentiment: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.md"

    content = [f"# {ticker} — MVP Technical Report", ""]
    content.append(sections["technical"])

    if "fundamental" in sections:
        content += ["", sections["fundamental"]]

    if "sentiment" in sections:
        content += ["", sections["sentiment"]]

    note_tail = "price/technical"
    if has_fundamentals:
        note_tail += " + fundamental"
    if has_sentiment:
        note_tail += " + sentiment"

    content += ["", f"_Note: MVP report generated from {note_tail} data._"]

    out_path.write_text("\n".join(content))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MVP report from technical.json (+ optional fundamental.json & sentiment files) and prices.parquet")
    ap.add_argument("--ticker", required=True, help="Ticker (directory under data/)")
    ap.add_argument("--data-dir", default="data", help="Input root")
    ap.add_argument("--out-dir", default="output", help="Output root (writes output/<ticker>/report.md)")
    ap.add_argument("--sent-root", default="output/sentiment", help="Sentiment root (default: output/sentiment)")
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

    # Optional fundamentals
    fund = _load_fundamental_json(fund_path)
    sections: Dict[str, str] = {"technical": technical_md}
    has_fund = bool(fund)
    if fund:
        sections["fundamental"] = _mk_fundamental_summary(args.ticker, fund)

    # Optional sentiment (includes top headlines)
    sent_items = _load_sentiment_items(Path(args.sent_root) / args.ticker)
    sent_md = _mk_sentiment_summary(sent_items)
    has_sent = bool(sent_md)
    if sent_md:
        sections["sentiment"] = sent_md

    out = Path(args.out_dir) / args.ticker
    out_path = _write_report_md(args.ticker, out, sections, has_fundamentals=has_fund, has_sentiment=has_sent)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
