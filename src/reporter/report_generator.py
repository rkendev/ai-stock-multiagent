from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

import pandas as pd


# ----------------------------
# Loaders (tolerant of layouts)
# ----------------------------
def _load_prices_any(ticker: str, data_dir: Path, out_dir: Path) -> pd.DataFrame:
    """
    Try data/<TICKER>/prices.parquet, then output/<TICKER>/prices.parquet.
    Normalize to have a 'Close' column and tz-naive DatetimeIndex.
    """
    candidates = [
        data_dir / ticker / "prices.parquet",
        out_dir / ticker / "prices.parquet",
    ]
    for p in candidates:
        if p.exists():
            return _load_prices(p)
    raise FileNotFoundError(f"Missing prices parquet; tried: {', '.join(str(c) for c in candidates)}")


def _load_prices(prices_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(prices_path)
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Normalize a 'Close' column
    if "Close" not in df.columns:
        for c in df.columns:
            if str(c).lower() == "close":
                df = df.rename(columns={c: "Close"})
                break
    # Datetime index, tz-naive, sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=False)
        except Exception:
            pass
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
    if "Close" not in df.columns:
        raise ValueError("Parquet must include a 'Close' column (case-insensitive)")
    return df


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_technical(path: Path) -> Dict[str, Any]:
    return _load_json(path)


def _load_sentiment_items(sent_dir: Path) -> List[Dict[str, Any]]:
    """
    Load sentiment items from per-article JSON files. Be tolerant:
    - If a file contains a list, extend with its dict items.
    - If a file contains a dict, append it.
    - Skip everything else.
    """
    if not sent_dir.exists():
        return []
    items: List[Dict[str, Any]] = []
    for p in sorted(sent_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict):
            items.append(obj)
        elif isinstance(obj, list):
            items.extend([o for o in obj if isinstance(o, dict)])
        # else: skip
    return items


def _load_fundamentals_any(ticker: str, data_dir: Path, out_dir: Path) -> Optional[dict]:
    """
    Prefer output/fundamentals/<TICKER>.json (FMP adapter),
    fall back to data/<TICKER>/fundamentals.json or fundamental.json.
    """
    candidates = [
        out_dir.parent / "fundamentals" / f"{ticker}.json",  # output/fundamentals/<T>.json
        data_dir / ticker / "fundamentals.json",
        data_dir / ticker / "fundamental.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return _load_json(p)
            except Exception:
                continue
    return None


# ----------------------------
# Render helpers
# ----------------------------
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


def _split_top_headlines(items: List[Dict[str, Any]], k: int = 3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def _score(it: Dict[str, Any]) -> float:
        try:
            return float(it.get("score", 0.0))
        except Exception:
            return 0.0
    pos = [it for it in items if str(it.get("sentiment", "")).upper() == "POSITIVE"]
    neg = [it for it in items if str(it.get("sentiment", "")).upper() == "NEGATIVE"]
    pos.sort(key=_score, reverse=True)
    neg.sort(key=_score)
    return pos[:k], neg[:k]


def _mk_sentiment_summary(items: List[Dict[str, Any]]) -> Optional[str]:
    # guard: only dicts
    items = [it for it in items if isinstance(it, dict)]
    if not items:
        return None

    scores: List[float] = []
    for i in items:
        try:
            scores.append(float(i.get("score", 0.0)))
        except Exception:
            pass
    n = max(1, len(items))
    avg = (sum(scores) / len(scores)) if scores else 0.0

    def pct(label: str) -> float:
        # label is POSITIVE | NEUTRAL | NEGATIVE
        return 100.0 * sum(
            1 for it in items if str(it.get("sentiment", "")).upper() == label
        ) / n

    pos = pct("POSITIVE"); neu = pct("NEUTRAL"); neg = pct("NEGATIVE")
    top_pos, top_neg = _split_top_headlines(items, 3)

    def _fmt(it: Dict[str, Any]) -> str:
        title = str(it.get("title", "")).strip()
        link = (it.get("link") or it.get("url") or "").strip()
        try:
            score = float(it.get("score", 0.0))
        except Exception:
            score = 0.0
        base = f"  - {title} *(score {score:+.2f})*"
        return f"{base} — {link}" if link else base

    lines = [
        "**Sentiment Snapshot**",
        f"- Headlines analyzed: {n}",
        f"- Average score: {avg:+.2f}",
        f"- Mix: {pos:.0f}% POSITIVE, {neu:.0f}% NEUTRAL, {neg:.0f}% NEGATIVE",
    ]
    if top_pos:
        lines += ["", "Top positive headlines:"] + [_fmt(it) for it in top_pos]
    if top_neg:
        lines += ["", "Top negative headlines:"] + [_fmt(it) for it in top_neg]
    return "\n".join(lines)


def _mk_fundamental_summary(ticker: str, fund: dict) -> str:
    """
    Render fundamentals from either:
      - fund['signals'] (legacy)  OR
      - fund['metrics'] (FMP adapter: revenue_yoy, net_income_yoy, margins...)
    """
    sig = (fund or {}).get("signals")
    if isinstance(sig, dict) and sig:
        bits = []
        bits.append("YoY revenue growth positive" if sig.get("rev_yoy_positive") else "YoY revenue growth not confirmed")
        bits.append("YoY earnings growth positive" if sig.get("eps_yoy_positive") else "YoY earnings growth not confirmed")
        bits.append("margin improving" if sig.get("margin_improving") else "margin not improving")
        return "**Fundamental Snapshot**\n- " + "\n- ".join(bits)

    # Fallback: infer quick signals from FMP metrics
    m = (fund or {}).get("metrics", {}) or {}
    rev_yoy = m.get("revenue_yoy")
    ni_yoy = m.get("net_income_yoy")
    gm = m.get("gross_margin")
    om = m.get("operating_margin")

    def _pct(v):
        try:
            return f"{float(v)*100:+.1f}%"
        except Exception:
            return "N/A"

    bits: List[str] = ["**Fundamental Snapshot**"]
    if rev_yoy is not None:
        bits.append(f"- Revenue YoY: {_pct(rev_yoy)}")
    if ni_yoy is not None:
        bits.append(f"- Net income YoY: {_pct(ni_yoy)}")
    if gm is not None:
        bits.append(f"- Gross margin (TTM): {float(gm)*100:.1f}%")
    if om is not None:
        bits.append(f"- Operating margin (TTM): {float(om)*100:.1f}%")

    if len(bits) == 1:
        bits.append("- (No fundamentals available yet)")
    return "\n".join(bits)


# ----------------------------
# Markdown writer
# ----------------------------
def _write_report_md(
    ticker: str,
    out_dir: Path,
    sections: Dict[str, str],
    has_fundamentals: Optional[bool] = None,
    has_sentiment: Optional[bool] = None,
) -> Path:
    """
    Write the Markdown report. Backwards compatible with the older
    3-argument test signature.

    If has_fundamentals/has_sentiment are None, infer from `sections`
    (presence of "fundamentals"/"sentiment" keys with truthy content).
    """
    if has_fundamentals is None:
        has_fundamentals = bool(sections.get("fundamentals"))
    if has_sentiment is None:
        has_sentiment = bool(sections.get("sentiment"))

    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {ticker} — MVP Technical Report",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
    ]

    tech = sections.get("technical")
    if tech:
        lines += ["## Technical Summary", tech, ""]

    fund = sections.get("fundamentals")
    if fund:
        lines += ["## Fundamentals", fund, ""]

    senti = sections.get("sentiment")
    if senti:
        lines += ["## Sentiment", senti, ""]

    lines += [
        "**Included sections:** "
        f"technical {'✅' if bool(tech) else '—'}, "
        f"fundamentals {'✅' if has_fundamentals else '—'}, "
        f"sentiment {'✅' if has_sentiment else '—'}"
    ]

    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate report.md")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--sent-root", default="output/sentiment")
    # accept legacy alias names too
    ap.add_argument("--out-root", dest="out_dir_alias", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir if args.out_dir else (args.out_dir_alias or "output"))
    sent_root = Path(args.sent_root)

    # Load prices (data/ first, then output/)
    prices = _load_prices_any(args.ticker, data_dir, out_dir)

    # Technical: tolerate missing; render with neutral bias if absent
    tech_path = data_dir / args.ticker / "technical.json"
    if tech_path.exists():
        tech = _load_technical(tech_path)
    else:
        tech = {"signals": {}}

    # as-of date: prefer technical.json -> last price date
    asof = tech.get("asof")
    if not asof and isinstance(prices.index, pd.DatetimeIndex) and len(prices):
        asof = str(prices.index[-1].date())
    asof = asof or "N/A"
    technical_md = _mk_technical_summary(args.ticker, asof, tech, prices)

    sections: Dict[str, str] = {"technical": technical_md}

    # Fundamentals: prefer output/fundamentals/<T>.json; fallback to data/
    fund = _load_fundamentals_any(args.ticker, data_dir, out_dir)
    has_fund = bool(fund)
    if fund:
        sections["fundamentals"] = _mk_fundamental_summary(args.ticker, fund)

    # Sentiment
    sent_items = _load_sentiment_items(sent_root / args.ticker)
    sent_md = _mk_sentiment_summary(sent_items)
    has_sent = bool(sent_md)
    if sent_md:
        sections["sentiment"] = sent_md

    # Write
    out_dir_ticker = out_dir / args.ticker
    out = _write_report_md(args.ticker, out_dir_ticker, sections, has_fund, has_sent)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
