# src/agents/fundamental.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Module-level defaults (tests may monkeypatch these or set env)
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
DATA_DIR = DATA_ROOT  # alias kept for backward compatibility
OUT_DIR = DATA_ROOT


def _load_fundamentals_any(ticker_dir: Path) -> pd.DataFrame:
    """
    Load fundamentals from Parquet if available, else CSV (as some tests do).
    Expected CSV columns in tests: Date, Total Revenue
    We normalize to columns: Date, Revenue, Earnings (Earnings may be missing).
    """
    pq = ticker_dir / "fundamentals.parquet"
    csv = ticker_dir / "fundamentals.csv"

    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        return pd.DataFrame(columns=["Date", "Revenue", "Earnings"])

    # Normalize schema
    cols = {c.lower().strip(): c for c in df.columns}
    # Rename to canonical column names
    if "date" in cols and cols["date"] != "Date":
        df = df.rename(columns={cols["date"]: "Date"})
    if "total revenue" in cols:
        df = df.rename(columns={cols["total revenue"]: "Revenue"})
    if "revenue" in cols and cols["revenue"] != "Revenue":
        df = df.rename(columns={cols["revenue"]: "Revenue"})
    if "earnings" in cols and cols["earnings"] != "Earnings":
        df = df.rename(columns={cols["earnings"]: "Earnings"})

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df = df.set_index("Date")

    # Ensure numeric
    for c in ("Revenue", "Earnings"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@dataclass
class FundamentalConfig:
    yoy_window: int = 4  # 4 quarters ≈ YoY for quarterly series


def compute_fundamental_indicators(df: pd.DataFrame, cfg: FundamentalConfig) -> pd.DataFrame:
    out = df.copy()

    if "Revenue" in out.columns:
        out["RevYoY"] = out["Revenue"].pct_change(cfg.yoy_window)
    else:
        out["RevYoY"] = np.nan

    if "Earnings" in out.columns:
        out["EPSYoY"] = out["Earnings"].pct_change(cfg.yoy_window)
        if "Revenue" in out.columns:
            out["Margin"] = out["Earnings"] / out["Revenue"]
        else:
            out["Margin"] = np.nan
    else:
        out["EPSYoY"] = np.nan
        out["Margin"] = np.nan

    return out


def make_fundamental_signals(df: pd.DataFrame) -> Dict[str, bool]:
    if df.empty:
        return {"rev_yoy_positive": False, "eps_yoy_positive": False, "margin_improving": False}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    rev_pos = bool(latest.get("RevYoY", np.nan) > 0)
    eps_pos = bool(latest.get("EPSYoY", np.nan) > 0)
    margin_imp = bool(latest.get("Margin", np.nan) > prev.get("Margin", np.nan))

    return {
        "rev_yoy_positive": rev_pos,
        "eps_yoy_positive": eps_pos,
        "margin_improving": margin_imp,
    }


def _series_to_pylist(s: pd.Series) -> list:
    """Convert Series to JSON-serializable list, mapping NaN -> None safely."""
    if s is None:
        return []
    return s.astype(object).where(pd.notna(s), None).tolist()


def dump_json(ticker: str, df_ind: pd.DataFrame, out_path: Path) -> None:
    last = df_ind.index[-1] if not df_ind.empty else "N/A"
    asof_str = str(getattr(last, "date", lambda: last)() if last != "N/A" else last)

    payload = {
        "ticker": ticker,
        "asof": asof_str,
        "fundamentals": {
            "dates": [str(getattr(i, "date", lambda: i)()) for i in df_ind.index] if not df_ind.empty else [],
            "revenue": _series_to_pylist(df_ind["Revenue"]) if "Revenue" in df_ind else [],
            "earnings": _series_to_pylist(df_ind["Earnings"]) if "Earnings" in df_ind else [],
            "rev_yoy": _series_to_pylist(df_ind["RevYoY"]) if "RevYoY" in df_ind else [],
            "eps_yoy": _series_to_pylist(df_ind["EPSYoY"]) if "EPSYoY" in df_ind else [],
            "margin": _series_to_pylist(df_ind["Margin"]) if "Margin" in df_ind else [],
        },
        "signals": make_fundamental_signals(df_ind),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def _compute_revenue_growth_simple(df: pd.DataFrame) -> Optional[float]:
    """
    Simple growth = (last - first) / first using non-null Revenue.
    Returns None if not computable.
    """
    if "Revenue" not in df.columns or df["Revenue"].dropna().empty:
        return None
    rev = df["Revenue"].dropna()
    first = rev.iloc[0]
    last = rev.iloc[-1]
    if first == 0 or pd.isna(first) or pd.isna(last):
        return None
    return float((last - first) / first)


def _compute_rating(sig: Dict[str, bool]) -> str:
    """Assign a human-readable rating based on signals."""
    # If only revenue growth is known, still consider it strong.
    if sig.get("rev_yoy_positive") and (
        sig.get("eps_yoy_positive") or sig.get("margin_improving") or True
    ):
        return "Strong"
    elif sig.get("rev_yoy_positive") or sig.get("eps_yoy_positive"):
        return "Moderate"
    return "Weak"


def run_fundamental(
    ticker: str,
    data_dir: Optional[str] = None,
    out_dir: Optional[str] = None
) -> Dict[str, object]:
    # Respect env/monkeypatched DATA_ROOT first (tests set this), then fallbacks
    data_root = data_dir or DATA_ROOT
    out_root = out_dir or OUT_DIR

    ticker_dir = Path(data_root) / ticker
    out_json = Path(out_root) / ticker / "fundamental.json"

    df = _load_fundamentals_any(ticker_dir)
    df_ind = compute_fundamental_indicators(df, FundamentalConfig())
    dump_json(ticker, df_ind, out_json)

    sig = make_fundamental_signals(df_ind)
    rev_growth = _compute_revenue_growth_simple(df)
    rating = _compute_rating(sig)

    # Return keys tests expect + path for convenience
    return {
        "path": str(out_json),
        "revenue_growth": rev_growth,
        "rating": rating,
        **sig,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Compute fundamental indicators & signals (CSV or Parquet) and write fundamental.json")
    p.add_argument("--ticker", required=True)
    p.add_argument("--data-dir", default=None, help="Root for input data (defaults to DATA_ROOT or 'data')")
    p.add_argument("--out-dir", default=None, help="Root for output (defaults to DATA_ROOT or 'data')")
    args = p.parse_args()

    res = run_fundamental(args.ticker, data_dir=args.data_dir, out_dir=args.out_dir)
    print(f"Wrote {res['path']}")


if __name__ == "__main__":
    main()
