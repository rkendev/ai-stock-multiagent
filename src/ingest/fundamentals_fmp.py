# src/ingest/fundamentals_fmp.py
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import time
import urllib.request
import urllib.error

FMP_KEY_ENV = "FMP_API_KEY"

# Minimal in-module ticker mapping for non-US tickers (adjust as needed)
FMP_TICKER_MAP: Dict[str, str] = {
    "ASML.AS": "ASML"  # FMP uses ASML (NASDAQ) for most endpoints
}

@dataclass
class FMPClient:
    api_key: str
    base: str = "https://financialmodelingprep.com/api/v3"

    def _get(self, path: str) -> Optional[list[dict]]:
        url = f"{self.base}{path}&apikey={self.api_key}" if "?" in path else f"{self.base}{path}?apikey={self.api_key}"
        for attempt in range(2):
            try:
                with urllib.request.urlopen(url, timeout=6) as r:
                    return json.loads(r.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                time.sleep(0.6 * (attempt + 1))
            except Exception:
                break
        return None


def _map_symbol(ticker: str) -> str:
    return FMP_TICKER_MAP.get(ticker, ticker)


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default


def _normalize(
    profile: Optional[list[dict]],
    key_metrics_ttm: Optional[list[dict]],
    income_annual: Optional[list[dict]],
) -> dict:
    # profile[0] contains company metadata
    company = {}
    if profile and isinstance(profile, list) and profile:
        p0 = profile[0]
        company = {
            "name": p0.get("companyName") or p0.get("companyName") or "",
            "sector": p0.get("sector") or "",
            "industry": p0.get("industry") or "",
        }

    # TTM metrics
    metrics = {}
    if key_metrics_ttm and isinstance(key_metrics_ttm, list) and key_metrics_ttm:
        k = key_metrics_ttm[0]
        metrics.update({
            "revenue_ttm": _to_float(k.get("revenuePerShareTTM")) * _to_float(k.get("sharesOutstanding"), 1.0),  # rough
            "net_income_ttm": _to_float(k.get("netIncomePerShareTTM")) * _to_float(k.get("sharesOutstanding"), 1.0),  # rough
            "eps_ttm": _to_float(k.get("epsTTM")),
            "fcf_ttm": _to_float(k.get("freeCashFlowTTM")),
            "debt_to_equity": _to_float(k.get("debtToEquityTTM")),
            "gross_margin": _to_float(k.get("grossProfitMarginTTM")),
            "operating_margin": _to_float(k.get("operatingMarginTTM")),
        })

    # YoY deltas from last two annual income statements (if available)
    revenue_yoy = 0.0
    net_income_yoy = 0.0
    if income_annual and isinstance(income_annual, list) and len(income_annual) >= 2:
        cur = _to_float(income_annual[0].get("revenue"))
        prev = _to_float(income_annual[1].get("revenue"))
        if prev:
            revenue_yoy = (cur - prev) / prev
        cur_n = _to_float(income_annual[0].get("netIncome"))
        prev_n = _to_float(income_annual[1].get("netIncome"))
        if prev_n:
            net_income_yoy = (cur_n - prev_n) / prev_n

    metrics.setdefault("revenue_ttm", 0.0)
    metrics.setdefault("net_income_ttm", 0.0)
    metrics["revenue_yoy"] = revenue_yoy
    metrics["net_income_yoy"] = net_income_yoy

    return {
        "company": company,
        "metrics": metrics,
    }


def fetch_fundamentals(ticker: str, out_root: str = "output") -> Path:
    api_key = os.getenv(FMP_KEY_ENV, "").strip()
    if not api_key:
        raise SystemExit(f"{FMP_KEY_ENV} is not set")

    sym = _map_symbol(ticker)
    client = FMPClient(api_key=api_key)

    profile = client._get(f"/profile/{sym}")
    key_metrics_ttm = client._get(f"/key-metrics-ttm/{sym}")
    income_annual = client._get(f"/income-statement/{sym}?period=annual&limit=2")

    data = _normalize(profile, key_metrics_ttm, income_annual)

    out_dir = Path(out_root) / "fundamentals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.json"

    payload = {
        "ticker": ticker,
        "as_of": time.strftime("%Y-%m-%d"),
        **data,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main(argv: Optional[list[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--out-root", default="output")
    args = ap.parse_args(argv)

    p = fetch_fundamentals(args.ticker, out_root=args.out_root)
    print(f"Wrote fundamentals → {p}")


if __name__ == "__main__":
    main()
