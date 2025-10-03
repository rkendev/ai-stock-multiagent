# src/agents/orchestrator.py
"""
Simple orchestrator that runs the full pipeline end-to-end.

Order:
  1) Ingest: prices, fundamentals, news
  2) Researcher -> Sentiment
  3) Fundamental agent -> Technical agent
  4) Reporter (LLM) to synthesize a markdown report

Notes:
  - This module shells out to each script with the current interpreter.
  - Keep arguments small and explicit; the called scripts do their own I/O.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import List


def mkdirp(path: str) -> None:
    """Create directories like `mkdir -p` (idempotent)."""
    os.makedirs(path, exist_ok=True)


def build_cmd(template: str, **kwargs) -> List[str]:
    """
    Expand a shell template into an argv list, substituting placeholders and
    splitting safely with shlex.split.

    Placeholders available:
      {py}      -> sys.executable
      {ticker}  -> ticker symbol
      {since}   -> ISO date
      {limit}   -> integer limit
      {model}   -> LLM model name
      {temp}    -> LLM temperature (float)

    Example:
        build_cmd("{py} src/ingest/prices.py --ticker {ticker} --since {since}",
                  ticker="AAPL", since="2024-06-01")
    """
    filled = template.format(py=shlex.quote(sys.executable), **kwargs)
    return shlex.split(filled)


def run(template: str, **kwargs) -> None:
    """Format and execute a command template, logging the exact command."""
    argv = build_cmd(template, **kwargs)
    print("Running:", " ".join(shlex.quote(a) for a in argv), flush=True)
    subprocess.run(argv, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline orchestrator")
    p.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    p.add_argument(
        "--since",
        default="2024-06-01",
        help="Start date for price/history (YYYY-MM-DD). Default: %(default)s",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max news items to fetch/analyze. Default: %(default)s",
    )
    p.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="Reporter LLM model. Default: %(default)s",
    )
    p.add_argument(
        "--temp",
        type=float,
        default=0.2,
        help="Reporter LLM temperature. Default: %(default)s",
    )
    args = p.parse_args()

    # Ensure output root exists (agents/scripts create their own subdirs)
    mkdirp("output")

    # Ingest
    run("{py} src/ingest/prices.py --ticker {ticker} --since {since}",
        ticker=args.ticker, since=args.since)
    run("{py} src/ingest/fundamentals.py --ticker {ticker}",
        ticker=args.ticker)
    run("{py} src/ingest/news.py --ticker {ticker} --limit {limit}",
        ticker=args.ticker, limit=args.limit)

    # Analysis
    run("{py} src/agents/researcher.py --ticker {ticker} --limit {limit}",
        ticker=args.ticker, limit=args.limit)
    run("{py} src/agents/sentiment.py --ticker {ticker} --limit {limit}",
        ticker=args.ticker, limit=args.limit)
    run("{py} src/agents/fundamental.py --ticker {ticker}",
        ticker=args.ticker)
    run("{py} src/agents/technical.py --ticker {ticker}",
        ticker=args.ticker)

    # Reporter (LLM)
    run("{py} src/agents/reporter.py --ticker {ticker} --model {model} --temp {temp}",
        ticker=args.ticker, model=args.model, temp=args.temp)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[Orchestrator] Completed for {args.ticker} at {ts}", flush=True)


if __name__ == "__main__":
    main()
