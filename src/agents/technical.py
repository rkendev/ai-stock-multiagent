# src/agents/technical.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, List
import numpy as np
import pandas as pd

# Default data directory (override in tests or via env)
DATA_DIR = os.getenv("DATA_ROOT", "data")


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)  # neutral when undefined


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # use high/low/close if available; otherwise fall back to close-range
    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # lowercase + snake-ish
    rename_map = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=rename_map)

    # promote aliases → canonical
    alias_map = {
        "close": ["close", "adj_close", "adjusted_close", "close_price", "price"],
        "open": ["open"],
        "high": ["high"],
        "low": ["low"],
        "ma50": ["ma50", "sma50", "moving_average_50"],
        "ma200": ["ma200", "sma200", "moving_average_200"],
    }

    def first_existing(cands: List[str]) -> Optional[str]:
        for n in cands:
            if n in df.columns:
                return n
        return None

    col_promote = {}
    for target, candidates in alias_map.items():
        src = first_existing(candidates)
        if src and src != target:
            col_promote[src] = target
    if col_promote:
        df = df.rename(columns=col_promote)

    # index cleanup
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def compute_signals(ticker: str, data_root: Optional[str] = None) -> Path:
    """
    Compute RSI(14), ATR(14) percentile (1y), and 52w proximity from prices.parquet.
    Writes data/<TICKER>/technical.json and returns that Path.

    Also tolerates prices at output/<TICKER>/prices.parquet if data/ is missing.
    """
    # resolve data_root default to module-level DATA_DIR
    if data_root is None:
        data_root = DATA_DIR

    # Look for prices in data/ first, then output/
    p_data = Path(data_root) / ticker / "prices.parquet"
    p_out = Path("output") / ticker / "prices.parquet"
    if p_data.exists():
        p = p_data
    elif p_out.exists():
        p = p_out
    else:
        raise FileNotFoundError(f"Prices parquet not found: {p_data} or {p_out}")

    df = pd.read_parquet(p)
    df = _normalize_price_df(df)

    if "close" not in df.columns:
        raise ValueError(
            "prices.parquet must include a 'close' column (aliases supported: adj_close/adjusted_close)"
        )

    # Compute MAs if missing
    if "ma50" not in df.columns:
        df["ma50"] = df["close"].rolling(window=50, min_periods=1).mean()
    if "ma200" not in df.columns:
        df["ma200"] = df["close"].rolling(window=200, min_periods=1).mean()

    # RSI(14)
    rsi14 = _rsi(df["close"], 14)
    rsi_latest = float(rsi14.iloc[-1])

    # ATR(14) + 1y percentile (robust percentile calc)
    atr14 = _atr(df, 14)
    lookback = min(len(atr14), 252)
    if lookback >= 20:
        ref = atr14.iloc[-lookback:]
        current = float(ref.iloc[-1])
        pct = float((ref <= current).mean())  # percentile of current vs 1y window
    else:
        pct = 0.0

    # 52-week proximity
    window = min(len(df), 252)
    close = df["close"]
    if window >= 2:
        last = float(close.iloc[-1])
        high_52w = float(close.iloc[-window:].max())
        low_52w = float(close.iloc[-window:].min())
        dist_high = (high_52w - last) / high_52w if high_52w else 0.0
        dist_low = (last - low_52w) / low_52w if low_52w else 0.0
    else:
        dist_high = dist_low = 0.0

    # Flags
    above_ma50 = bool(close.iloc[-1] > df["ma50"].iloc[-1])
    above_ma200 = bool(close.iloc[-1] > df["ma200"].iloc[-1])
    overbought = bool(rsi_latest >= 70)
    oversold = bool(rsi_latest <= 30)
    volatility_high = bool(pct >= 0.8)  # ≥80th percentile

    payload = {
        "ticker": ticker,
        "signals": {
            "above_MA50": above_ma50,
            "above_MA200": above_ma200,
            "overbought": overbought,
            "oversold": oversold,
            "rally_volatility_high": volatility_high,
        },
        "indicators": {
            "RSI14": rsi_latest,
            "MA50": float(df["ma50"].iloc[-1]),
            "MA200": float(df["ma200"].iloc[-1]),
            "ATR14": float(atr14.iloc[-1]) if len(atr14) else 0.0,
            "ATR14_percentile_1y": pct,
            "dist_to_52w_high_pct": dist_high,
            "dist_from_52w_low_pct": dist_low,
        },
    }

    out_path = Path(data_root) / ticker / "technical.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ---- Backwards-compatible wrapper expected by tests ----
def run_technical(ticker: str, *args: Any, data_root: Optional[str] = None, **kwargs: Any) -> dict:
    """
    Compatibility entrypoint used by tests.

    Supports:
      - run_technical(ticker)                         -> uses DATA_DIR (monkey-patchable), returns dict
      - run_technical(ticker, "<custom_data>")        -> legacy positional data_root
      - run_technical(ticker, data_root="...")        -> keyword data_root

    Returns dict with at least:
      {"rsi": float, "ma50": float, "overbought": bool}
    """
    # Legacy positional form: run_technical(ticker, "<data_root>")
    if args and isinstance(args[0], str):
        data_root = args[0]
    if data_root is None:
        data_root = DATA_DIR

    # Write JSON to disk via the canonical path
    path = compute_signals(ticker, data_root=data_root)

    # Read it back and build a flat, test-friendly summary
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        data = {}

    ind = (data or {}).get("indicators", {}) or {}
    sig = (data or {}).get("signals", {}) or {}

    # Map to legacy flat keys the test checks
    result = {
        "rsi": float(ind.get("RSI14", 0.0)),
        "ma50": float(ind.get("MA50", 0.0)),
        "overbought": bool(sig.get("overbought", False)),
        # extras (harmless)
        "ma200": float(ind.get("MA200", 0.0)),
        "volatility_high": bool(sig.get("rally_volatility_high", False)),
    }
    return result


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    p = compute_signals(args.ticker, data_root=args.data_root)
    print(f"Wrote technical signals → {p}")


__all__ = ["run_technical", "compute_signals", "DATA_DIR"]


if __name__ == "__main__":
    main()
