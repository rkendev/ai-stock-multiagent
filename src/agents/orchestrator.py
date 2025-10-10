from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional, List, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed


def mkdirp(path: str) -> None:
    """Create directories recursively (like mkdir -p)."""
    os.makedirs(path, exist_ok=True)


def build_cmd(template: str, **kwargs) -> List[str]:
    """
    Expand a shell template into argv list, substituting placeholders via kwargs
    plus {py} for the Python interpreter.

    Tests assert that this returns a list[str] where the first element ends with
    'python' or 'python3', so we return an argv list with sys.executable as the
    {py} value (unquoted).
    """
    filled = template.format(py=sys.executable, **kwargs)
    return shlex.split(filled)


def _to_argv(cmd: Union[str, List[str]]) -> List[str]:
    """Normalize a command into argv list for subprocess."""
    if isinstance(cmd, list):
        return cmd
    return shlex.split(cmd)


def run_cmd(cmd: Union[str, List[str]]) -> None:
    """
    Run a command. Accepts either a string or a list.

    Tests monkey-patch this function and often pass/inspect a *string* so they can
    do substring checks like `"prices.py" in cmd`. We keep the signature flexible,
    but internally normalize to argv and prefix with the Python interpreter for
    real execution.
    """
    argv = [sys.executable] + _to_argv(cmd)
    print("Running:", " ".join(shlex.quote(a) for a in argv), flush=True)
    subprocess.run(argv, check=True)


def orchestrate(
    ticker: str,
    since: Optional[str] = None,
    news_limit: int = 20,
    skip_news: bool = False,
    skip_prices: bool = False,
    skip_fundamentals: bool = False,
    parallel_ingest: bool = False,
) -> None:
    """
    Classic pipeline: ingest (prices/fundamentals/news) -> agents -> reporter.
    """
    mkdirp("output")

    # Build ingest commands as STRINGS (important for tests that do substring assertions)
    ingest_tasks: List[tuple[str, str]] = []
    if not skip_prices:
        prices_cmd = f"src/ingest/prices.py --ticker {ticker}"
        if since:
            prices_cmd += f" --since {since}"
        ingest_tasks.append(("prices", prices_cmd))

    if not skip_fundamentals:
        ingest_tasks.append(("fundamentals", f"src/ingest/fundamentals.py --ticker {ticker}"))

    if not skip_news:
        ingest_tasks.append(("news", f"src/ingest/news.py --ticker {ticker} --limit {news_limit}"))

    # Run ingestion (parallel or sequential), always via run_cmd
    if ingest_tasks:
        if parallel_ingest and len(ingest_tasks) > 1:
            with ThreadPoolExecutor(max_workers=len(ingest_tasks)) as executor:
                futures = {executor.submit(run_cmd, task_str): name for name, task_str in ingest_tasks}
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"[orchestrator] ingest {name} failed: {e}", file=sys.stderr, flush=True)
                        raise
        else:
            for _, task_str in ingest_tasks:
                run_cmd(task_str)

    # Agents + reporter (also strings for consistency/tests)
    run_cmd(f"src/agents/researcher.py --ticker {ticker} --limit {news_limit}")
    run_cmd(f"src/agents/sentiment.py --ticker {ticker} --limit {news_limit}")
    run_cmd(f"src/agents/fundamental.py --ticker {ticker}")
    run_cmd(f"src/agents/technical.py --ticker {ticker}")
    run_cmd(f"python -m reporter.report_generator --ticker {ticker}")


    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"[orchestrator] Completed {ticker} at {ts}", flush=True)


def orchestrate_with_prompt_flow(
    ticker: str,
    since: Optional[str] = None,
    news_limit: int = 20,
    skip_news: bool = False,
    skip_prices: bool = False,
    skip_fundamentals: bool = False,
    parallel_ingest: bool = False,
    use_prompt_v2: bool = False,
    use_local_llama: bool = False,
) -> None:
    """
    Orchestrate using PromptManager + LLMBackendFactory when use_prompt_v2=True.
    Still dispatches the agent scripts via run_cmd so tests can observe calls.
    If use_prompt_v2=False, falls back to classic orchestrate().
    """
    if not use_prompt_v2:
        return orchestrate(
            ticker=ticker,
            since=since,
            news_limit=news_limit,
            skip_news=skip_news,
            skip_prices=skip_prices,
            skip_fundamentals=skip_fundamentals,
            parallel_ingest=parallel_ingest,
        )

    # Optional prompt construction: safe to no-op if modules not present.
    try:
        from agents.prompt_manager import PromptManager  # type: ignore
        from agents.prompt_executor import LLMBackendFactory  # type: ignore

        pm = PromptManager()
        backend_id = "local-llama" if use_local_llama else "openai"
        executor = LLMBackendFactory.create(backend=backend_id)

        agents = pm.list_agents()
        prompts: Dict[str, str] = {
            ag: pm.render(ag, ticker=ticker, since=since or "", topic="analysis")
            for ag in agents
        }

        print(f"[prompt flow] Using backend: {backend_id}", flush=True)
        print("[prompt flow] prompts:", prompts, flush=True)

        # Execute prompts, tolerating different method names (execute vs execute_prompt)
        exec_fn = getattr(executor, "execute", None) or getattr(executor, "execute_prompt", None)

        for ag, prompt in prompts.items():
            try:
                if exec_fn:
                    _ = exec_fn(prompt, context={"ticker": ticker, "since": since})
                else:
                    print(f"[prompt flow] No executor method found for {backend_id}; skipping.", flush=True)
                    break
            except Exception as e:
                print(f"[prompt flow] {ag} execution failed: {e}", file=sys.stderr, flush=True)
    except Exception as e:
        # If anything goes wrong with prompt modules, just continue to script calls.
        print(f"[prompt flow] Skipping prompt execution: {e}", file=sys.stderr, flush=True)

    # For compatibility with tests, still invoke agent scripts (as STRINGS).
    run_cmd(f"src/agents/researcher.py --ticker {ticker} --limit {news_limit}")
    run_cmd(f"src/agents/sentiment.py --ticker {ticker} --limit {news_limit}")
    # inside orchestrate(...) or your main flow, after prices ingest:
    run_cmd(f"{sys.executable} -m ingest.fundamentals --ticker {ticker} --out-root data")
    # run fundamental agent to produce JSON signals
    run_cmd(f"{sys.executable} -m agents.fundamental --ticker {ticker} --data-dir data --out-dir data")
    run_cmd(f"src/agents/technical.py --ticker {ticker}")
    run_cmd(f"{sys.executable} -m reporter.report_generator --ticker {ticker}")


    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"[orchestrator:prompt-v2] Completed {ticker} at {ts}", flush=True)


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
