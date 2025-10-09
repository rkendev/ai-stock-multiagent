# src/agents/technical.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Module-level defaults so tests can monkeypatch these
DATA_DIR = "data"
OUT_DIR = "data"


def _load_prices(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise ValueError("Parquet must include a 'Close' column")
    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=series.index).rolling(period, min_periods=period).mean()
    loss = pd.Series(loss, index=series.index).rolling(period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50.0)


def _volatility(series: pd.Series, window: int = 20) -> pd.Series:
    returns = series.pct_change()
    return returns.rolling(window, min_periods=window).std().bfill()


@dataclass
class TechnicalConfig:
    ma_windows: tuple = (20, 50, 200)
    rsi_period: int = 14
    vol_window: int = 20


def compute_indicators(df: pd.DataFrame, cfg: TechnicalConfig) -> pd.DataFrame:
    out = df.copy()
    for w in cfg.ma_windows:
        out[f"MA{w}"] = out["Close"].rolling(w, min_periods=1).mean()
    out[f"RSI{cfg.rsi_period}"] = _rsi(out["Close"], cfg.rsi_period)
    out[f"VOL{cfg.vol_window}"] = _volatility(out["Close"], cfg.vol_window)
    return out


def make_signals(df: pd.DataFrame, cfg: TechnicalConfig) -> Dict[str, bool]:
    latest = df.iloc[-1]
    signals = {
        "above_MA50": bool(latest.get("Close", np.nan) > latest.get("MA50", np.nan))
        if "MA50" in df.columns else False,
        "above_MA200": bool(latest.get("Close", np.nan) > latest.get("MA200", np.nan))
        if "MA200" in df.columns else False,
        "overbought": bool(latest.get(f"RSI{cfg.rsi_period}", 50) > 70),
        "oversold": bool(latest.get(f"RSI{cfg.rsi_period}", 50) < 30),
        "rally_volatility_high": bool(
            latest.get(f"VOL{cfg.vol_window}", 0)
            > df[f"VOL{cfg.vol_window}"].median()
        )
        if f"VOL{cfg.vol_window}" in df.columns else False,
    }
    return signals


def dump_json(ticker: str, df_ind: pd.DataFrame, cfg: TechnicalConfig, out_path: Path) -> None:
    last_date = df_ind.index[-1]
    payload = {
        "ticker": ticker,
        "asof": str(last_date.date() if hasattr(last_date, "date") else last_date),
        "indicators": {
            "MA20": df_ind["MA20"].tolist() if "MA20" in df_ind else [],
            "MA50": df_ind["MA50"].tolist() if "MA50" in df_ind else [],
            "MA200": df_ind["MA200"].tolist() if "MA200" in df_ind else [],
            "RSI14": df_ind["RSI14"].tolist() if "RSI14" in df_ind else [],
            "volatility_20d": df_ind["VOL20"].tolist() if "VOL20" in df_ind else [],
        },
        "signals": make_signals(df_ind, cfg),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def run_technical(
    ticker: str,
    data_dir: Optional[str] = None,
    out_dir: Optional[str] = None
) -> Dict[str, object]:
    """
    Loads <data_dir>/<ticker>/prices.parquet, computes indicators & signals,
    writes <out_dir>/<ticker>/technical.json, and returns a small summary dict
    the tests expect: {'rsi': float, 'ma50': float, 'overbought': bool, 'path': str}
    """
    data_dir = data_dir or DATA_DIR
    out_dir = out_dir or OUT_DIR

    parquet_path = Path(data_dir) / ticker / "prices.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing: {parquet_path}")

    df = _load_prices(parquet_path)
    cfg = TechnicalConfig()
    df_ind = compute_indicators(df, cfg)

    out_json = Path(out_dir) / ticker / "technical.json"
    dump_json(ticker, df_ind, cfg, out_json)

    latest = df_ind.iloc[-1]
    rsi_val = float(latest.get("RSI14", 50.0))
    ma50_val = float(latest.get("MA50", latest["Close"]))
    overbought = bool(rsi_val > 70.0)
    return {"rsi": rsi_val, "ma50": ma50_val, "overbought": overbought, "path": str(out_json)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute technical indicators & signals from prices.parquet")
    ap.add_argument("--ticker", required=True, help="Ticker (directory under data/)")
    ap.add_argument("--data-dir", default=None, help="Input root (defaults to module DATA_DIR)")
    ap.add_argument("--out-dir", default=None, help="Output root (defaults to module OUT_DIR)")
    args = ap.parse_args()

    res = run_technical(args.ticker, data_dir=args.data_dir, out_dir=args.out_dir)
    print(f"Wrote {res['path']}")


if __name__ == "__main__":
    main()
