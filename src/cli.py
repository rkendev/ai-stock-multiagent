from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _runpy(mod: str, *args: str) -> None:
    env = os.environ.copy()
    # Ensure src is importable
    src = str(Path(__file__).resolve().parent)
    repo_root = str(Path(src).parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", mod, *args]
    subprocess.run(cmd, check=True, env=env)


def refresh(ticker: str, since: str, limit: int) -> None:
    _runpy("ingest.prices", "--ticker", ticker, "--since", since, "--out-root", "data")
    # technical.json generation assumed elsewhere in your pipeline; keep existing one if present

    _runpy("ingest.news", "--ticker", ticker, "--out-root", "output", "--limit", str(limit), "--since", since)
    _runpy("agents.sentiment", "--ticker", ticker, "--news-root", "output/news", "--out-root", "output/sentiment", "--limit", str(limit))
    _runpy("sentiment.score", "--ticker", ticker, "--in-root", "output", "--out-root", "output")
    _runpy("reporter.report_generator", "--ticker", ticker, "--data-dir", "data", "--out-dir", "output")


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
