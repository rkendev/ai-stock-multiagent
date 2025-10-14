from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# FIX: repo root is two levels up (repo/src/ui/app.py -> parents[2] == repo)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "output"


def _pyenv() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


# -----------------------
# Stable, canonical loader
# -----------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _load_prices(ticker: str) -> pd.DataFrame:
    """
    Loads prices once and normalizes to a canonical schema so plotting
    logic is simple and deterministic.

    Canonical columns (lowercase): close, ma50, ma200
    Index: tz-naive DatetimeIndex, sorted.
    """
    p = DATA_ROOT / ticker / "prices.parquet"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_parquet(p)

    # 1) Flatten any MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2) Normalize column names → lowercase, snake-ish
    rename_map = {}
    for c in df.columns:
        k = str(c).strip().lower().replace(" ", "_")
        rename_map[c] = k
    df = df.rename(columns=rename_map)

    # 3) Promote common aliases to canonical names
    alias_map = {
        "close": ["close", "adj_close", "adjusted_close", "price", "close_price"],
        "ma50": ["ma50", "sma50", "moving_average_50"],
        "ma200": ["ma200", "sma200", "moving_average_200"],
    }

    def first_existing(candidates: list[str]) -> str | None:
        for n in candidates:
            if n in df.columns:
                return n
        return None

    col_promote = {}
    for target, candidates in alias_map.items():
        src = first_existing(candidates)
        if src and src != target:
            col_promote[src] = target
    if col_promote:
        df = df.rename(columns=col_promote)

    # 4) Clean index → tz-naive DatetimeIndex, sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
        except Exception:
            pass
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    # 5) Keep only the columns we plot (others can exist but are not required)
    cols = [c for c in ["close", "ma50", "ma200"] if c in df.columns]
    return df[cols].copy() if cols else pd.DataFrame(index=df.index)


# -----------------------
# Pure plotting (no I/O)
# -----------------------
def _price_chart(df: pd.DataFrame) -> go.Figure:
    """
    Build the price + moving averages chart from canonical columns.
    Assumes columns are lowercase: close, ma50, ma200
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", mode="lines"))
    if "ma50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ma50"], name="MA50", mode="lines"))
    if "ma200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ma200"], name="MA200", mode="lines"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), title="Price & Moving Averages")
    return fig


def _refresh(ticker: str, since: str, limit: int) -> str:
    # Run the consolidated CLI to keep all orchestration in one place
    import subprocess

    cmd = [sys.executable, "-m", "cli", "refresh", "--ticker", ticker, "--since", since, "--limit", str(limit)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_pyenv())
    if proc.returncode != 0:
        return f"Refresh failed:\n{proc.stderr or proc.stdout}"
    return "News fetched (since {since}, limit={limit}). Sentiment scored. Report regenerated.".format(
        since=since, limit=limit
    )


# -----------------------
# Sentiment helpers
# -----------------------
def _sentiment_files(ticker: str) -> list[Path]:
    """
    Collect sentiment files from common layouts:
      1) output/sentiment/<TICKER>/*.json
      2) output/<TICKER>/sentiment/*.json
      3) output/sentiment/<TICKER>.jsonl  (single JSONL file)
    """
    candidates: list[Path] = []
    p1 = OUT_ROOT / "sentiment" / ticker
    p2 = OUT_ROOT / ticker / "sentiment"
    if p1.exists():
        candidates.extend(sorted(p1.glob("*.json")))
    if p2.exists():
        candidates.extend(sorted(p2.glob("*.json")))
    # JSONL file (single file with many lines of JSON)
    jl1 = OUT_ROOT / "sentiment" / f"{ticker}.jsonl"
    if jl1.exists():
        candidates.append(jl1)
    return candidates


def _read_sentiment_records(paths: list[Path]) -> list[dict]:
    import json

    recs: list[dict] = []
    for p in paths:
        if p.suffix == ".jsonl":
            # one object per line
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
        else:
            try:
                recs.append(json.loads(p.read_text()))
            except Exception:
                pass
    return recs


def main() -> None:
    st.set_page_config(layout="wide", page_title="AI Stock Multi-Agent")

    st.sidebar.header("AI Stock Multi-Agent")
    ticker = st.sidebar.selectbox("Ticker", ["ASML.AS", "AAPL"], index=0)
    st.sidebar.caption("Tip: run  `./run_mvp.sh <TICKER> <SINCE>`  to refresh data.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Refresh data (news → sentiment → report)")
    since = st.sidebar.date_input("Since (for news fetch)", date(2023, 10, 1))
    limit = st.sidebar.slider("News limit", min_value=5, max_value=100, value=20, step=5)
    if st.sidebar.button("Refresh data now", use_container_width=True):
        msg = _refresh(ticker, since.strftime("%Y-%m-%d"), limit)
        st.sidebar.success(msg)

    st.title("AI Stock Multi-Agent")
    st.subheader(f"{ticker} — Dashboard")

    # Price chart
    df = _load_prices(ticker)
    st.markdown("### Price & Moving Averages")
    if df.empty or "close" not in df.columns:
        st.warning("No prices found. Use refresh or run the MVP pipeline.")
    else:
        try:
            st.plotly_chart(_price_chart(df), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render price chart: {e}")

    st.markdown("---")

    # Technical snapshot flags (simple reads from data/technical.json if present)
    tech_path = DATA_ROOT / ticker / "technical.json"
    sig = {}
    if tech_path.exists():
        import json

        try:
            sig = json.loads(tech_path.read_text()).get("signals", {})
        except Exception:
            sig = {}
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Above MA50", "✅" if sig.get("above_MA50") else "—")
    with col2:
        st.metric("Above MA200", "✅" if sig.get("above_MA200") else "—")
    with col3:
        st.metric("Overbought", "—" if not sig.get("overbought") else "⚠️")
    with col4:
        st.metric("Oversold", "—" if not sig.get("oversold") else "⚠️")
    with col5:
        st.metric("Volatility High", "⚠️" if sig.get("rally_volatility_high") else "—")

    st.markdown("---")

    # -----------------------
    # Sentiment Snapshot (patched)
    # -----------------------
    st.subheader("Sentiment Snapshot")
    st.caption(f"Reading output from: `{OUT_ROOT}`")  # small debug aid

    sent_paths = _sentiment_files(ticker)
    records = _read_sentiment_records(sent_paths)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Headlines", len(records))

    if records:
        # tolerate different key names
        def _score(r: dict) -> float:
            try:
                return float(r.get("score", r.get("polarity", 0.0)))
            except Exception:
                return 0.0

        def _label(r: dict) -> str:
            return str(r.get("sentiment", r.get("label", ""))).upper()

        scores = [_score(r) for r in records]
        pos = sum(1 for r in records if _label(r) == "POSITIVE")
        neg = sum(1 for r in records if _label(r) == "NEGATIVE")

        avg = (sum(scores) / len(scores)) if scores else 0.0
        with cols[1]:
            st.metric("Avg score", f"{avg:+.2f}")
        with cols[2]:
            st.metric("Positive", f"{(pos * 100 / len(records)):.0f}%")
        with cols[3]:
            st.metric("Negative", f"{(neg * 100 / len(records)):.0f}%")

        # tiny diagnostics
        from datetime import datetime

        latest_path = max(sent_paths, key=lambda p: p.stat().st_mtime)
        latest_time = latest_path.stat().st_mtime
        st.caption(
            f"Loaded {len(records)} records from {len(sent_paths)} file(s). "
            f"Latest file: {latest_path.name} @ {datetime.fromtimestamp(latest_time).isoformat(sep=' ', timespec='seconds')}"
        )
        ex = records[-1]
        st.caption(f'Example: {ex.get("title") or ex.get("headline") or "(no title)"}')

    else:
        with cols[1]:
            st.metric("Avg score", f"{0:+.2f}")
        with cols[2]:
            st.metric("Positive", "0%")
        with cols[3]:
            st.metric("Negative", "0%")

        # why zero? tell the truth:
        news_dir1 = OUT_ROOT / "news" / ticker
        news_dir2 = OUT_ROOT / ticker / "news"
        has_news = (news_dir1.exists() and any(news_dir1.glob("*"))) or (
            news_dir2.exists() and any(news_dir2.glob("*"))
        )
        if has_news:
            st.info("News exists but no sentiment files were found. Run the sentiment step or fix the writer path/extension.")
        else:
            st.info("No news found for this ticker/date window. Check your since filter, ticker mapping, or API key.")

    st.markdown("---")

    # Report viewer (download markdown)
    report_md = OUT_ROOT / ticker / "report.md"
    st.subheader("Report")
    if report_md.exists():
        st.download_button("Download report.md", report_md.read_bytes(), file_name="report.md")
        st.markdown(report_md.read_text())
    else:
        st.info("Report not found. Generate with the MVP pipeline or the refresh button.")


if __name__ == "__main__":
    main()
