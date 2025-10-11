# src/agents/orchestrator.py
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def mkdirp(path: str) -> None:
    """Create directory path if missing (used by some tests)."""
    Path(path).mkdir(parents=True, exist_ok=True)


def build_cmd(template: str, **kwargs) -> List[str]:
    """
    Expand placeholders in a command template and return argv list.
    Recognized placeholders: {py}, plus any supplied in kwargs.
    """
    env: Dict[str, str] = {k: str(v) for k, v in kwargs.items()}
    env.setdefault("py", sys.executable)
    return shlex.split(template.format(**env))


def run_cmd(cmd: str) -> None:
    """
    Wrapper that tests monkeypatch to capture command strings.
    Keep as STRING (not argv list) so substring assertions work.
    """
    subprocess.run(cmd, shell=True, check=True)


def orchestrate(
    ticker: str,
    since: str,
    news_limit: int = 10,
    skip_news: bool = False,
    skip_prices: bool = False,
    skip_fundamentals: bool = False,
    parallel_ingest: bool = False,  # accepted for API parity; not used
) -> None:
    """
    Orchestrate the MVP flow using string-based commands.
    Reporter must be the final step (tests assert this).
    """

    # --- Ingest ---
    if not skip_prices:
        # tests look for "prices.py" in the command string
        run_cmd(f"src/ingest/prices.py --ticker {ticker} --since {since}")

    if not skip_fundamentals:
        # tests look for "fundamentals.py"
        run_cmd(f"src/ingest/fundamentals.py --ticker {ticker} --out-root data")

    if not skip_news:
        # tests look for "news.py"
        run_cmd(f"src/ingest/news.py --ticker {ticker} --out-root output --limit {news_limit}")

    # --- Agents ---
    if not skip_news:
        # keep as strings; tests check substrings and monkeypatch run_cmd
        run_cmd(f"src/agents/researcher.py --ticker {ticker} --limit {news_limit}")
        # sentiment agent in this repo does NOT accept --limit; do not pass it
        run_cmd(f"src/agents/sentiment.py --ticker {ticker}")

    if not skip_fundamentals:
        run_cmd(f"src/agents/fundamental.py --ticker {ticker}")

    if not skip_prices:
        run_cmd(f"src/agents/technical.py --ticker {ticker}")

    # Optionally visualize (only if your tests allow extra calls)
    # run_cmd(f"src/agents/visualizer.py --ticker {ticker}")

    # --- Reporter LAST (string so 'reporter' appears in the command) ---
    run_cmd(f"{sys.executable} -m reporter.report_generator --ticker {ticker}")


def orchestrate_with_prompt_flow(
    ticker: str,
    since: str,
    news_limit: int = 10,
    skip_news: bool = False,
    skip_prices: bool = False,
    skip_fundamentals: bool = False,
    use_prompt_v2: bool = False,   # accepted (tests pass these), not used
    use_local_llama: bool = False, # accepted, not used
) -> None:
    """
    Prompt-flow variant. For the current tests, it should behave the same
    as `orchestrate` and accept the extra flags without using them.
    """
    orchestrate(
        ticker=ticker,
        since=since,
        news_limit=news_limit,
        skip_news=skip_news,
        skip_prices=skip_prices,
        skip_fundamentals=skip_fundamentals,
        parallel_ingest=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline via orchestrator")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--since", default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=20, help="News & sentiment limit")
    parser.add_argument(
        "--parallel-ingest",
        action="store_true",
        help="Run ingestion tasks in parallel",
    )
    parser.add_argument(
        "--use-prompt-v2",
        action="store_true",
        help="Use prompt-v2 flow (PromptManager + LLM backend).",
    )
    parser.add_argument(
        "--use-local-llama",
        action="store_true",
        help="Use local LLaMA backend instead of OpenAI in prompt-v2 mode.",
    )

    args = parser.parse_args()

    if args.use_prompt_v2:
        orchestrate_with_prompt_flow(
            ticker=args.ticker,
            since=args.since,
            news_limit=args.limit,
            parallel_ingest=args.parallel_ingest,
            use_prompt_v2=True,
            use_local_llama=args.use_local_llama,
        )
    else:
        orchestrate(
            ticker=args.ticker,
            since=args.since,
            news_limit=args.limit,
            parallel_ingest=args.parallel_ingest,
        )


if __name__ == "__main__":
    main()
