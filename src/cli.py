from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
import json, re

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyenv() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env

def _runpy(mod: str, *args: str) -> None:
    cmd = [sys.executable, "-m", mod, *args]
    subprocess.run(cmd, check=True, env=_pyenv(), text=True)
    

def _runpy_soft(mod: str, *args: str) -> None:
    cmd = [sys.executable, "-m", mod, *args]
    proc = subprocess.run(cmd, check=False, env=_pyenv(), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[warn] {mod} failed (non-fatal):\n{proc.stderr or proc.stdout}")


def _purge_legacy_sentiment(ticker: str, root: str = "output/sentiment") -> None:
    p = Path(root) / ticker
    if not p.exists():
        return
    for f in p.glob("*.json"):
        try:
            if f.read_text(encoding="utf-8").lstrip().startswith("["):
                f.unlink()
        except Exception:
            pass


def refresh(ticker: str, since: str, limit: int) -> None:
    _runpy("ingest.news", "--ticker", ticker, "--out-root", "output", "--limit", str(limit), "--since", since)
    _runpy("sentiment.score", "--ticker", ticker, "--in-root", "output", "--out-root", "output")

    # NEW: build price parquet before technical
    _runpy("ingest.prices", "--ticker", ticker, "--out-root", "output", "--since", since)

    # NEW: fundamentals & technical as soft steps (don’t kill batch on timeout/missing data)
    _runpy_soft("ingest.fundamentals_fmp", "--ticker", ticker, "--out-root", "outpu t")
    _runpy_soft("agents.technical", "--ticker", ticker, "--data-root", "data")

    _purge_legacy_sentiment(ticker)
    _runpy("reporter.report_generator", 
        "--ticker", ticker,
        "--out-dir", "output",
        "--data-dir", "data",
        "--sent-root", "output/sentiment")


def main() -> None:
    ap = argparse.ArgumentParser(prog="ai-stock-cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("refresh", help="prices -> news -> sentiment -> report")
    r.add_argument("--ticker", required=True)
    r.add_argument("--since", required=True, help="YYYY-MM-DD")
    r.add_argument("--limit", type=int, default=20)

    args = ap.parse_args()
    if args.cmd == "refresh":
        refresh(args.ticker, args.since, args.limit)


if __name__ == "__main__":
    main()
