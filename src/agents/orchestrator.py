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
    Expand a shell template into argv list, substituting only the placeholders
    provided plus {py} for the current interpreter.

    Example:
        build_cmd("{py} src/ingest/prices.py --ticker {ticker} --since {since}",
                  ticker="AAPL", since="2024-06-01")
    """
    values = {"py": sys.executable}
    # Accept any subset of kwargs (tests may pass only some of them)
    for k, v in kwargs.items():
        values[k] = str(v)

    try:
        rendered = template.format(**values)
    except KeyError as e:
        # Give a clearer error if a required placeholder wasn't provided
        missing = e.args[0]
        raise KeyError(f"Missing placeholder value for {{{missing}}} in template: {template}") from e

    return shlex.split(rendered)


def run_cmd(template: str, **kwargs) -> None:
    """Render a command from template and run it, streaming output."""
    argv = build_cmd(template, **kwargs)
    print("Running:", " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Simple orchestrator to run the agents end-to-end.")
    p.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    p.add_argument("since", nargs="?", default=None, help="Optional start date, e.g., 2024-06-01")
    p.add_argument("--limit", type=int, default=20, help="News/sentiment limit")
    args = p.parse_args()

    ticker = args.ticker
    since = args.since
    limit = args.limit

    # Ensure output root exists (honor OUTPUT_ROOT if set)
    output_root = os.getenv("OUTPUT_ROOT", "output")
    mkdirp(output_root)

    # Ingest
    if since:
        run_cmd("{py} src/ingest/prices.py --ticker {ticker} --since {since}", ticker=ticker, since=since)
    else:
        # If no since is provided, just fetch 1 year back as a default example
        default_since = (datetime.now(timezone.utc).date().replace(year=datetime.now(timezone.utc).year - 1)).isoformat()
        run_cmd("{py} src/ingest/prices.py --ticker {ticker} --since {since}", ticker=ticker, since=default_since)

    run_cmd("{py} src/ingest/fundamentals.py --ticker {ticker}", ticker=ticker)
    run_cmd("{py} src/ingest/news.py --ticker {ticker} --limit {limit}", ticker=ticker, limit=limit)

    # Agents
    run_cmd("{py} -m agents.researcher --ticker {ticker} --limit {limit}", ticker=ticker, limit=limit)
    run_cmd("{py} -m agents.sentiment --ticker {ticker} --limit {limit}", ticker=ticker, limit=limit)
    run_cmd("{py} -m agents.fundamental --ticker {ticker}", ticker=ticker)
    run_cmd("{py} -m agents.technical --ticker {ticker}", ticker=ticker)

    # Reporter (OpenAI by default). ReporterAgent honors OUTPUT_ROOT automatically.
    run_cmd("{py} -m agents.reporter --ticker {ticker}", ticker=ticker)


if __name__ == "__main__":
    main()
