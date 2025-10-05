# src/agents/orchestrator.py
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional, List, Union

from agents.prompt_manager import PromptManager
from agents.prompt_executor import StubExecutor, PromptExecutor


def mkdirp(path: str) -> None:
    """Create directories recursively (like mkdir -p)."""
    os.makedirs(path, exist_ok=True)


def build_cmd(template: str, **kwargs) -> List[str]:
    """
    Expand a shell template into argv list, substituting placeholders via kwargs
    plus {py} for the Python interpreter.
    """
    filled = template.format(py=shlex.quote(sys.executable), **kwargs)
    return shlex.split(filled)


def run_cmd(cmd: Union[str, List[str]]) -> None:
    """
    Run a command. Accepts either a shell-like string (preferred for tests that
    patch run_cmd and look for substrings like 'prices.py') or a list[str].
    """
    if isinstance(cmd, str):
        # For display: show exactly what orchestrator passed (helps tests)
        print("Running:", cmd, flush=True)
        argv = [sys.executable] + shlex.split(cmd)
    else:
        # Backward-compat: accept a list too
        print("Running:", " ".join(shlex.quote(a) for a in ([sys.executable] + cmd)), flush=True)
        argv = [sys.executable] + cmd

    subprocess.run(argv, check=True)


def orchestrate(
    ticker: str,
    since: Optional[str] = None,
    news_limit: int = 20,
    skip_news: bool = False,
    skip_prices: bool = False,
    skip_fundamentals: bool = False,
    parallel_ingest: bool = False,  # kept for CLI compatibility; currently sequential
) -> None:
    """
    Run the full pipeline. Ingestion steps are invoked first, then analysis and reporter.
    We intentionally pass *strings* to run_cmd so tests can assert substring presence.
    """
    mkdirp("output")

    # ---- Ingestion (sequential; tests only require that calls happen) ----
    if not skip_prices:
        cmd = f"src/ingest/prices.py --ticker {shlex.quote(ticker)}"
        if since:
            cmd += f" --since {shlex.quote(since)}"
        run_cmd(cmd)

    if not skip_fundamentals:
        run_cmd(f"src/ingest/fundamentals.py --ticker {shlex.quote(ticker)}")

    if not skip_news:
        run_cmd(f"src/ingest/news.py --ticker {shlex.quote(ticker)} --limit {news_limit}")

    # ---- Analysis & reporting ----
    run_cmd(f"src/agents/researcher.py --ticker {shlex.quote(ticker)} --limit {news_limit}")
    run_cmd(f"src/agents/sentiment.py --ticker {shlex.quote(ticker)} --limit {news_limit}")
    run_cmd(f"src/agents/fundamental.py --ticker {shlex.quote(ticker)}")
    run_cmd(f"src/agents/technical.py --ticker {shlex.quote(ticker)}")
    run_cmd(["src/agents/visualizer.py", "--ticker", ticker])    
    run_cmd(f"src/agents/reporter.py --ticker {shlex.quote(ticker)}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"[orchestrator] Completed {ticker} at {ts}", flush=True)


def orchestrate_with_prompt_flow(
    ticker: str, since: str, use_prompt_v2: bool = False
) -> None:
    """
    If use_prompt_v2 is True, use PromptManager + PromptExecutor to drive agent runs.
    Else fallback to classic behavior.
    """
    if not use_prompt_v2:
        return orchestrate(ticker, since)  # your existing orchestrator

    pm = PromptManager()
    executor: PromptExecutor = StubExecutor()

    agents = pm.list_agents()
    prompts = {ag: pm.render(ag, ticker=ticker, since=since, topic="analysis")
               for ag in agents}

    print("[prompt flow] prompts:", prompts)

    responses = {}
    for ag, prompt in prompts.items():
        resp = executor.execute(ag, prompt, context={"ticker": ticker, "since": since})
        responses[ag] = resp
        print(f"[prompt flow] {ag} → {resp}")

    # After responses collected, decide which actual agent runs to spawn
    # For now: run all (you can refine logic later)
    for ag in agents:
        run_cmd(["src/agents/" + ag + ".py", "--ticker", ticker, "--since", since])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline via orchestrator")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--since", default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=20, help="News & sentiment limit")
    parser.add_argument(
        "--parallel-ingest",
        action="store_true",
        help="(Reserved) Run ingestion tasks in parallel",
    )

    args = parser.parse_args()

    orchestrate(
        ticker=args.ticker,
        since=args.since,
        news_limit=args.limit,
        parallel_ingest=args.parallel_ingest,
    )


if __name__ == "__main__":
    main()
