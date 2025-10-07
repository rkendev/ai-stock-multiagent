from __future__ import annotations

import os
import sys
import subprocess
from typing import Optional

import typer

# Try to load .env automatically if python-dotenv is available
try:
    from dotenv import load_dotenv, find_dotenv, dotenv_values
except ImportError:
    load_dotenv = find_dotenv = dotenv_values = None


def _load_env(env_file: Optional[str] = None) -> None:
    """Load environment variables from .env (or given env_file) into os.environ."""
    if not load_dotenv:
        return
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        path = find_dotenv(usecwd=True) if find_dotenv else None
        if path:
            load_dotenv(path, override=False)


# Load .env at import time
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


# =====================================================
# INGEST COMMANDS
# =====================================================

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
    limit: int = typer.Option(20, "--limit", "-n", min=1, help="Number of news items"),
):
    from ingest.news import main as news_main  # type: ignore
    sys.argv = ["news.py", "--ticker", ticker, "--limit", str(limit)]
    news_main()


# =====================================================
# AGENT COMMANDS
# =====================================================

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
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model id"),
    temp: Optional[float] = typer.Option(None, "--temp", help="LLM temperature"),
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


# =====================================================
# ORCHESTRATION + PROMPT-V2 + LOCAL-LLAMA SUPPORT
# =====================================================

def _merge_dotenv(env: dict) -> dict:
    """Merge .env variables into environment for subprocess calls."""
    if find_dotenv:
        path = find_dotenv(usecwd=True)
        if path and dotenv_values:
            env.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    return env


def _call_orchestrator(
    ticker: str,
    since: Optional[str],
    news_limit: int,
    skip_news: bool,
    skip_prices: bool,
    skip_fundamentals: bool,
    parallel_ingest: bool,
    use_prompt_v2: bool,
    use_local_llama: bool,
    model_name: Optional[str],
) -> None:
    """Invoke orchestrator (in-process or subprocess fallback)."""
    try:
        import agents.orchestrator as orch  # type: ignore
        if use_prompt_v2 and hasattr(orch, "orchestrate_with_prompt_flow"):
            fn = orch.orchestrate_with_prompt_flow
        else:
            fn = orch.orchestrate

        fn(
            ticker=ticker,
            since=since,
            news_limit=news_limit,
            skip_news=skip_news,
            skip_prices=skip_prices,
            skip_fundamentals=skip_fundamentals,
            parallel_ingest=parallel_ingest,
            use_local_llama=use_local_llama,
        )
        return
    except ImportError:
        pass

    # --- Fallback via subprocess
    cmd = [sys.executable, "src/agents/orchestrator.py", ticker]
    if since:
        cmd += ["--since", since]
    if parallel_ingest:
        cmd += ["--parallel-ingest"]
    if use_prompt_v2:
        cmd += ["--use-prompt-v2"]
    if use_local_llama:
        cmd += ["--use-local-llama"]
    if model_name:
        cmd += ["--model", model_name]

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env = _merge_dotenv(env)
    print(f"[cli] Fallback run: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


@app.command("run")
def run_pipeline(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    since_arg: Optional[str] = typer.Argument(None, help="Optional since date"),
    since_opt: Optional[str] = typer.Option(None, "--since", "-s", help="Since date"),
    limit: int = typer.Option(20, "--limit", "-n", help="News limit"),
    skip_news: bool = typer.Option(False, "--skip-news", help="Skip news ingestion"),
    skip_prices: bool = typer.Option(False, "--skip-prices", help="Skip price ingestion"),
    skip_fundamentals: bool = typer.Option(False, "--skip-fundamentals", help="Skip fundamentals ingestion"),
    parallel: bool = typer.Option(False, "--parallel-ingest", "-p", help="Parallel ingestion"),
    use_prompt_v2: bool = typer.Option(False, "--use-prompt-v2", help="Enable prompt-based workflow"),
    use_local_llama: bool = typer.Option(False, "--use-local-llama", help="Use local LLaMA backend for prompt execution"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name for LLM backend"),
):
    """Main entrypoint for full pipeline orchestration."""
    since = since_opt or since_arg
    _call_orchestrator(
        ticker=ticker,
        since=since,
        news_limit=limit,
        skip_news=skip_news,
        skip_prices=skip_prices,
        skip_fundamentals=skip_fundamentals,
        parallel_ingest=parallel,
        use_prompt_v2=use_prompt_v2,
        use_local_llama=use_local_llama,
        model_name=model,
    )


def get_app() -> typer.Typer:
    return app


if __name__ == "__main__":
    app()
