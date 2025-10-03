from __future__ import annotations

import os
import sys
import subprocess
from typing import Optional

import typer

# -----------------------------
# .env loading (once, early)
# -----------------------------
def _load_env(env_file: Optional[str] = None) -> None:
    """
    Load .env into process environment.
    - If env_file is provided, load that.
    - Else try to find a .env starting from CWD upward.
    - If nothing is found, noop.
    """
    try:
        from dotenv import load_dotenv, find_dotenv
    except Exception:
        # python-dotenv not installed; just skip silently
        return

    if env_file:
        if os.path.exists(env_file):
            load_dotenv(env_file, override=False)
        return

    # Auto-discover .env (repo root or cwd)
    discovered = find_dotenv(usecwd=True)
    if discovered:
        # Only apply if not empty string
        if os.path.exists(discovered):
            load_dotenv(discovered, override=False)

# Load .env automatically at import time (safe no-op if missing)
_load_env()

app = typer.Typer(
    help="Multi-Agent Stock Analyst - unified CLI",
    no_args_is_help=True,
    add_completion=False,
)

# Global option (optional) allowing: python -m cli --env-file .env run AAPL ...
@app.callback()
def cli_root(
    env_file: Optional[str] = typer.Option(
        None,
        "--env-file",
        help="Path to a .env file to load before executing commands.",
    )
):
    if env_file:
        _load_env(env_file)


def env_or(value: Optional[str], env_key: str, fallback: Optional[str]) -> Optional[str]:
    """Return value if provided, else read env_key, else fallback."""
    if value is not None:
        return value
    return os.environ.get(env_key, fallback)


# =====================================================================================
# Ingest subcommands
# =====================================================================================
ingest_app = typer.Typer(help="Data ingestion commands", no_args_is_help=True)
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("prices")
def ingest_prices(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol (e.g., AAPL)"),
    since: str = typer.Option(..., "--since", "-s", help="Start date (YYYY-MM-DD)"),
):
    """Fetch historical prices and write Parquet."""
    # Lazy import keeps CLI fast and avoids heavy deps during --help
    from ingest.prices import main as prices_main  # type: ignore

    # Emulate argv to reuse existing main()
    sys.argv = ["prices.py", "--ticker", ticker, "--since", since]
    prices_main()


@ingest_app.command("fundamentals")
def ingest_fundamentals(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
):
    """Fetch fundamentals and write CSV."""
    from ingest.fundamentals import main as fundamentals_main  # type: ignore
    sys.argv = ["fundamentals.py", "--ticker", ticker]
    fundamentals_main()


@ingest_app.command("news")
def ingest_news(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="Number of items to fetch"),
):
    """Fetch RSS news and write JSON."""
    from ingest.news import main as news_main  # type: ignore
    sys.argv = ["news.py", "--ticker", ticker, "--limit", str(limit)]
    news_main()


# =====================================================================================
# Agent subcommands
# =====================================================================================
agent_app = typer.Typer(help="Agent commands", no_args_is_help=True)
app.add_typer(agent_app, name="agent")


@agent_app.command("researcher")
def agent_researcher(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="Number of items"),
):
    """Run the Researcher agent."""
    from agents.researcher import main as researcher_main  # type: ignore
    sys.argv = ["researcher.py", "--ticker", ticker, "--limit", str(limit)]
    researcher_main()


@agent_app.command("sentiment")
def agent_sentiment(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="Number of items"),
):
    """Run the Sentiment agent."""
    from agents.sentiment import main as sentiment_main  # type: ignore
    sys.argv = ["sentiment.py", "--ticker", ticker, "--limit", str(limit)]
    sentiment_main()


@agent_app.command("fundamental")
def agent_fundamental(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
):
    """Run the Fundamental agent."""
    from agents.fundamental import main as fundamental_agent_main  # type: ignore
    sys.argv = ["fundamental.py", "--ticker", ticker]
    fundamental_agent_main()


@agent_app.command("technical")
def agent_technical(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
):
    """Run the Technical agent."""
    from agents.technical import main as technical_agent_main  # type: ignore
    sys.argv = ["technical.py", "--ticker", ticker]
    technical_agent_main()


@agent_app.command("reporter")
def agent_reporter(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    model: Optional[str] = typer.Option(
        None, "--model", help="LLM model id (OpenAI or local). Defaults to env REPORTER_MODEL if unset."
    ),
    temp: Optional[float] = typer.Option(
        None, "--temp", min=0.0, max=1.0, help="Sampling temperature. Defaults to env REPORTER_TEMPERATURE if unset."
    ),
):
    """Run the Reporter agent."""
    from agents.reporter import main as reporter_main  # type: ignore

    model_val = env_or(model, "REPORTER_MODEL", None)
    temp_val = env_or(None if temp is None else str(temp), "REPORTER_TEMPERATURE", None)

    argv = ["reporter.py", "--ticker", ticker]
    if model_val:
        argv += ["--model", model_val]
    if temp_val:
        argv += ["--temp", str(temp_val)]

    sys.argv = argv
    reporter_main()


# =====================================================================================
# Orchestration
# =====================================================================================

def _call_orchestrator(
    ticker: str,
    since: Optional[str],
    news_limit: int,
    skip_news: bool,
    skip_prices: bool,
    skip_fundamentals: bool,
) -> None:
    """
    Prefer calling a Python function (agents.orchestrator.orchestrate),
    fall back to shelling out to the legacy script to preserve compatibility.
    Ensures current env (with .env vars) is inherited by children.
    """
    # Lazy import (avoid heavy imports for CLI help)
    try:
        import agents.orchestrator as orch  # type: ignore
    except Exception:
        orch = None

    if orch is not None:
        fn = getattr(orch, "orchestrate", None) or getattr(orch, "orchestrator", None)
        if callable(fn):
            fn(
                ticker=ticker,
                since=since,
                news_limit=news_limit,
                skip_news=skip_news,
                skip_prices=skip_prices,
                skip_fundamentals=skip_fundamentals,
            )
            return

    # Fallback to the script (backward compatibility)
    cmd = [sys.executable, "src/agents/orchestrator.py", ticker]
    if since:
        cmd.append(since)
    env = os.environ.copy()
    env["ORCH_NEWS_LIMIT"] = str(news_limit)
    env["ORCH_SKIP_NEWS"] = "1" if skip_news else "0"
    env["ORCH_SKIP_PRICES"] = "1" if skip_prices else "0"
    env["ORCH_SKIP_FUNDS"] = "1" if skip_fundamentals else "0"

    print(f"Running (fallback): {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


@app.command("run")
def run_pipeline(
    ticker: str = typer.Argument(..., help="Ticker symbol (e.g., AAPL)"),
    # Support BOTH positional [SINCE] and option --since/-s, choose whichever is provided
    since_arg: Optional[str] = typer.Argument(
        None, help="Optional start date (YYYY-MM-DD) for price ingestion"
    ),
    since_opt: Optional[str] = typer.Option(
        None, "--since", "-s", help="Optional start date (YYYY-MM-DD) for price ingestion"
    ),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="News/sentiment fetch size"),
    skip_news: bool = typer.Option(False, "--skip-news", help="Skip RSS/news ingestion"),
    skip_prices: bool = typer.Option(False, "--skip-prices", help="Skip price ingestion"),
    skip_fundamentals: bool = typer.Option(False, "--skip-fundamentals", help="Skip fundamentals ingestion"),
):
    """
    Orchestrate: ingest -> researcher/sentiment -> fundamental/technical -> reporter

    Examples:
      - Positional since:
          python -m cli run AAPL 2024-06-01
      - Option since:
          python -m cli run AAPL --since 2024-06-01
      - With explicit env-file:
          python -m cli --env-file .env run AAPL --since 2024-06-01
    """
    since = since_opt or since_arg
    _call_orchestrator(
        ticker=ticker,
        since=since,
        news_limit=limit,
        skip_news=skip_news,
        skip_prices=skip_prices,
        skip_fundamentals=skip_fundamentals,
    )


# Optional alias: `orchestrate` -> same as `run`
@app.command("orchestrate")
def orchestrate_alias(
    ticker: str = typer.Argument(..., help="Ticker symbol (e.g., AAPL)"),
    since_arg: Optional[str] = typer.Argument(
        None, help="Optional start date (YYYY-MM-DD) for price ingestion"
    ),
    since_opt: Optional[str] = typer.Option(
        None, "--since", "-s", help="Optional start date (YYYY-MM-DD) for price ingestion"
    ),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="News/sentiment fetch size"),
    skip_news: bool = typer.Option(False, "--skip-news", help="Skip RSS/news ingestion"),
    skip_prices: bool = typer.Option(False, "--skip-prices", help="Skip price ingestion"),
    skip_fundamentals: bool = typer.Option(False, "--skip-fundamentals", help="Skip fundamentals ingestion"),
):
    """Alias for `run`—kept for discoverability/backwards compatibility."""
    since = since_opt or since_arg
    _call_orchestrator(
        ticker=ticker,
        since=since,
        news_limit=limit,
        skip_news=skip_news,
        skip_prices=skip_prices,
        skip_fundamentals=skip_fundamentals,
    )


def get_app() -> typer.Typer:
    """For programmatic embedding or testing."""
    return app


if __name__ == "__main__":
    app()
