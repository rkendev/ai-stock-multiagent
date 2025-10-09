# src/cli.py
from __future__ import annotations

import os
import typer
import pandas as pd  # noqa: F401  # used when writing parquet

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _write_parquet(df, out_path: str) -> None:
    try:
        df.to_parquet(out_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to write Parquet to {out_path}. "
            f"Ensure 'pyarrow' (or 'fastparquet') is installed. Original error: {e}"
        ) from e


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    ticker: str = typer.Option(..., "--ticker", "-t", help="Ticker symbol, e.g. ASML.AS"),
    since: str = typer.Option(..., "--since", "-s", help="Start date (YYYY-MM-DD)"),
):
    """
    Fetch price data for `ticker` since `since` and save to data/<ticker>/prices.parquet.
    """
    if ctx.invoked_subcommand is not None:
        return

    from ingest.prices import fetch_prices  # local import for fast startup

    df = fetch_prices(ticker, since)

    out_dir = os.path.join("data", ticker)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prices.parquet")
    _write_parquet(df, out_path)

    typer.echo(f"Fetched {ticker} prices since {since}")
    typer.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    app()
