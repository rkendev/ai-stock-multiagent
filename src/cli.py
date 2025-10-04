from __future__ import annotations

import os
import sys
import subprocess
from typing import Optional

import typer

# Try to load .env automatically if python-dotenv is available
try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    load_dotenv = None
    find_dotenv = None

def _load_env(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from .env (or given env_file) into os.environ.
    """
    if load_dotenv is None:
        return
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        path = find_dotenv(usecwd=True)
        if path:
            load_dotenv(path, override=False)

# Load .env at import time (so CLI commands see OPENAI_API_KEY etc.)
_load_env()

app = typer.Typer(help="Multi-Agent Stock CLI", no_args_is_help=True)

@app.callback()
def _global_options(
    env_file: Optional[str] = typer.Option(
        None, "--env-file", help="Path to .env file to load before running commands"
    )
):
    if env_file:
        _load_env(env_file)

def env_or(value: Optional[str], env_key: str, fallback: Optional[str]) -> Optional[str]:
    if value is not None:
        return value
    return os.environ.get(env_key, fallback)

# Ingest commands
ingest_app = typer.Typer(help="Data ingestion commands")
app.add_typer(ingest_app, name="ingest")

@ingest_app.command("prices")
def ingest_prices(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    since: str = typer.Option(..., "--since", "-s", help="Start date YYYY-MM-DD"),
):
    from ingest.prices import main as prices_main  # type: ignore
    sys.argv = ["prices.py", "--ticker", ticker, "--since", since]
    prices_main()

@ingest_app.command("fundamentals")
def ingest_fundamentals(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
):
    from ingest.fundamentals import main as fundamentals_main  # type: ignore
    sys.argv = ["fundamentals.py", "--ticker", ticker]
    fundamentals_main()

@ingest_app.command("news")
def ingest_news(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="Number of items"),
):
    from ingest.news import main as news_main  # type: ignore
    sys.argv = ["news.py", "--ticker", ticker, "--limit", str(limit)]
    news_main()

# Agent commands
agent_app = typer.Typer(help="Agent commands")
app.add_typer(agent_app, name="agent")

@agent_app.command("researcher")
def agent_researcher(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="Number of items"),
):
    from agents.researcher import main as researcher_main  # type: ignore
    sys.argv = ["researcher.py", "--ticker", ticker, "--limit", str(limit)]
    researcher_main()

@agent_app.command("sentiment")
def agent_sentiment(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="Number of items"),
):
    from agents.sentiment import main as sentiment_main  # type: ignore
    sys.argv = ["sentiment.py", "--ticker", ticker, "--limit", str(limit)]
    sentiment_main()

@agent_app.command("fundamental")
def agent_fundamental(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
):
    from agents.fundamental import main as fundamental_main  # type: ignore
    sys.argv = ["fundamental.py", "--ticker", ticker]
    fundamental_main()

@agent_app.command("technical")
def agent_technical(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
):
    from agents.technical import main as technical_main  # type: ignore
    sys.argv = ["technical.py", "--ticker", ticker]
    technical_main()

@agent_app.command("reporter")
def agent_reporter(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model id (default from env REPORTER_MODEL)"
    ),
    temp: Optional[float] = typer.Option(
        None, "--temp", help="LLM temperature (default from env REPORTER_TEMPERATURE)"
    ),
):
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

@agent_app.command("visualizer")
def agent_visualizer(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol"),
):
    from agents.visualizer import main as visualizer_main  # type: ignore
    sys.argv = ["visualizer.py", "--ticker", ticker]
    visualizer_main()    

# Orchestrator integration
def _call_orchestrator(
    ticker: str,
    since: Optional[str],
    news_limit: int,
    skip_news: bool,
    skip_prices: bool,
    skip_fundamentals: bool,
    parallel_ingest: bool = False,
) -> None:
    # Try to import orchestrate function
    try:
        import agents.orchestrator as orch  # type: ignore
        fn = getattr(orch, "orchestrate", None)
        if fn and callable(fn):
            fn(
                ticker=ticker,
                since=since,
                news_limit=news_limit,
                skip_news=skip_news,
                skip_prices=skip_prices,
                skip_fundamentals=skip_fundamentals,
                parallel_ingest=parallel_ingest,
            )
            return
    except ImportError:
        pass

    # Fallback: subprocess invocation
    cmd = [sys.executable, "src/agents/orchestrator.py", ticker]
    if since:
        cmd += ["--since", since]
    if parallel_ingest:
        cmd += ["--parallel-ingest"]
    # pass other options if needed (skip flags)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env = _merge_dotenv(env)
    print(f"Fallback run: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

def _merge_dotenv(env: dict) -> dict:
    if find_dotenv:
        path = find_dotenv(usecwd=True)
        if path:
            from dotenv import dotenv_values  # type: ignore
            env.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    return env

@app.command("run")
def run_pipeline(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    since_arg: Optional[str] = typer.Argument(None, help="Optional since date"),
    since_opt: Optional[str] = typer.Option(None, "--since", "-s", help="Since (YYYY-MM-DD)"),
    limit: int = typer.Option(20, "--limit", "-n", help="Limit on news/sentiment"),
    skip_news: bool = typer.Option(False, "--skip-news", help="Skip news ingestion"),
    skip_prices: bool = typer.Option(False, "--skip-prices", help="Skip price ingestion"),
    skip_fundamentals: bool = typer.Option(False, "--skip-fundamentals", help="Skip fundamentals ingestion"),
    parallel: bool = typer.Option(False, "--parallel-ingest", "-p", help="Run ingestion in parallel"),
):
    """Orchestrate full pipeline: ingest → analyze → reporter."""
    since = since_opt or since_arg
    _call_orchestrator(
        ticker=ticker,
        since=since,
        news_limit=limit,
        skip_news=skip_news,
        skip_prices=skip_prices,
        skip_fundamentals=skip_fundamentals,
        parallel_ingest=parallel,
    )

@app.command("orchestrate")
def orchestrate_alias(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    since_arg: Optional[str] = typer.Argument(None, help="Optional since date"),
    since_opt: Optional[str] = typer.Option(None, "--since", "-s", help="Since date"),
    limit: int = typer.Option(20, "--limit", "-n", help="Limit news/sentiment"),
    skip_news: bool = typer.Option(False, "--skip-news", help="Skip news ingestion"),
    skip_prices: bool = typer.Option(False, "--skip-prices", help="Skip prices ingestion"),
    skip_fundamentals: bool = typer.Option(False, "--skip-fundamentals", help="Skip fundamentals ingestion"),
    parallel: bool = typer.Option(False, "--parallel-ingest", "-p", help="Run ingestion in parallel"),
):
    since = since_opt or since_arg
    _call_orchestrator(
        ticker=ticker,
        since=since,
        news_limit=limit,
        skip_news=skip_news,
        skip_prices=skip_prices,
        skip_fundamentals=skip_fundamentals,
        parallel_ingest=parallel,
    )

def get_app() -> typer.Typer:
    return app

if __name__ == "__main__":
    app()
