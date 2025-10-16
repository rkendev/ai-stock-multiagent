# src/ui/app.py
from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.analyst_widget import render as render_analyst_widget


# repo paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
OUT_ROOT = REPO_ROOT / "output"
TICKERS_FILE = REPO_ROOT / "tickers_mvp.txt"


def _pyenv() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


# -----------------------
# Helpers: tickers & RSI
# -----------------------
def _read_tickers(path: Path) -> List[str]:
    if path.exists():
        raw = [ln.strip() for ln in path.read_text().splitlines()]
        return [t for t in raw if t and not t.startswith("#")]
    return ["ASML.AS", "AAPL"]


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


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
        p2 = OUT_ROOT / ticker / "prices.parquet"
        if p2.exists():
            p = p2
        else:
            return pd.DataFrame()

    df = pd.read_parquet(p)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename_map = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=rename_map)

    alias_map = {
        "close": ["close", "adj_close", "adjusted_close", "price", "close_price"],
        "ma50": ["ma50", "sma50", "moving_average_50"],
        "ma200": ["ma200", "sma200", "moving_average_200"],
    }

    def first_existing(cands: list[str]) -> str | None:
        for n in cands:
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

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
        except Exception:
            pass
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    cols = [c for c in ["close", "ma50", "ma200"] if c in df.columns]
    return df[cols].copy() if cols else pd.DataFrame(index=df.index)


def _sentiment_counts(ticker: str) -> Dict[str, float]:
    import json

    root1 = OUT_ROOT / "sentiment" / ticker
    root2 = OUT_ROOT / ticker / "sentiment"
    files: List[Path] = []
    if root1.exists():
        files += list(root1.glob("*.json"))
    if root2.exists():
        files += list(root2.glob("*.json"))

    pos = neg = neu = 0
    for f in files:
        try:
            obj = json.loads(f.read_text())
        except Exception:
            continue
        items = obj if isinstance(obj, list) else [obj]
        for it in items:
            if not isinstance(it, dict):
                continue
            label = str(it.get("sentiment", it.get("label", ""))).upper()
            if label == "POSITIVE":
                pos += 1
            elif label == "NEGATIVE":
                neg += 1
            else:
                neu += 1
    total = max(1, pos + neu + neg)
    return {
        "n": pos + neu + neg,
        "pos_pct": 100.0 * pos / total,
        "neg_pct": 100.0 * neg / total,
    }


# -----------------------
# Pure plotting (no I/O)
# -----------------------
def _price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", mode="lines"))
    if "ma50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ma50"], name="MA50", mode="lines"))
    if "ma200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ma200"], name="MA200", mode="lines"))
    fig.update_layout(
        title="Price & Moving Averages",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        autosize=True,
        width=None,
    )
    return fig


def _rsi_panel(df: pd.DataFrame, period: int = 14) -> go.Figure:
    if "close" not in df.columns or len(df) < 5:
        return go.Figure()
    rsi = _rsi(df["close"], period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi.values, name=f"RSI({period})", mode="lines"))
    fig.add_hline(y=70, line_dash="dot", annotation_text="70 overbought", annotation_position="top left")
    fig.add_hline(y=30, line_dash="dot", annotation_text="30 oversold", annotation_position="bottom left")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(
        title="RSI(14)",
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        autosize=True,
        width=None,
    )
    return fig


def _refresh(ticker: str, since: str, limit: int) -> str:
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
    candidates: list[Path] = []
    p1 = OUT_ROOT / "sentiment" / ticker
    p2 = OUT_ROOT / ticker / "sentiment"
    if p1.exists():
        candidates.extend(sorted(p1.glob("*.json")))
    if p2.exists():
        candidates.extend(sorted(p2.glob("*.json")))
    jl1 = OUT_ROOT / "sentiment" / f"{ticker}.jsonl"
    if jl1.exists():
        candidates.append(jl1)
    return candidates


def _read_sentiment_records(paths: list[Path]) -> list[dict]:
    import json

    recs: list[dict] = []
    for p in paths:
        if p.suffix == ".jsonl":
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


# -----------------------
# APP
# -----------------------
def main() -> None:
    st.set_page_config(layout="wide", page_title="AI Stock Multi-Agent")

    all_tickers = _read_tickers(TICKERS_FILE)

    st.sidebar.header("AI Stock Multi-Agent")
    ticker = st.sidebar.selectbox("Ticker (dashboard)", all_tickers, index=0)
    st.sidebar.caption("Tip: run  `./run_mvp.sh <TICKER> <SINCE>`  to refresh data.")
    st.sidebar.markdown("---")

    # Refresh control with "Last:" timestamp (per-ticker)
    st.sidebar.subheader("Refresh data (news → sentiment → report)")
    since = st.sidebar.date_input("Since (for news fetch)", date(2023, 10, 1))
    limit = st.sidebar.slider("News limit", min_value=5, max_value=100, value=20, step=5)

    # Show last refresh (persisted per ticker)
    last_key = f"last_refresh_{ticker}"
    last_val = st.session_state.get(last_key)
    col_btn, col_last = st.sidebar.columns([1.2, 1])
    with col_btn:
        do_refresh = st.button("Refresh data now", key="refresh_btn")
    with col_last:
        st.caption(f"Last: {last_val if last_val else '—'}")

    if do_refresh:
        msg = _refresh(ticker, since.strftime("%Y-%m-%d"), limit)
        st.sidebar.success(msg)
        # update the timestamp (local time)
        st.session_state[last_key] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Portfolio picker (persist selection) — no direct state writes + conditional default
    st.sidebar.markdown("---")
    st.sidebar.subheader("Portfolio (multi-select)")
    default_selection = all_tickers[:5] if "portfolio_selection" not in st.session_state else None
    portfolio = st.sidebar.multiselect(
        "Tickers (table below)",
        options=all_tickers,
        default=default_selection,       # only provided on first render
        key="portfolio_selection",       # Streamlit manages state after that
    )

    st.title("AI Stock Multi-Agent")

    # -----------------------
    # Portfolio snapshot
    # -----------------------
    if portfolio:
        rows = []
        for t in portfolio:
            dfp = _load_prices(t)
            if dfp.empty or "close" not in dfp.columns:
                continue
            last = float(dfp["close"].iloc[-1])
            lookback = min(len(dfp) - 1, 30) if len(dfp) > 1 else 1
            pct30 = (last / float(dfp["close"].iloc[-lookback]) - 1.0) * 100.0
            above_ma50 = "✅" if ("ma50" in dfp.columns and last > float(dfp["ma50"].iloc[-1])) else "—"
            above_ma200 = "✅" if ("ma200" in dfp.columns and last > float(dfp["ma200"].iloc[-1])) else "—"
            s = _sentiment_counts(t)
            rows.append({
                "Ticker": t,
                "Last": round(last, 2),
                "30d %": f"{pct30:+.1f}%",
                "Above MA50": above_ma50,
                "Above MA200": above_ma200,
                "Pos %": f"{s.get('pos_pct',0.0):.0f}%",
                "Neg %": f"{s.get('neg_pct',0.0):.0f}%",
                "Headlines": int(s.get("n", 0)),
            })
        if rows:
            st.markdown("### Portfolio")
            st.dataframe(pd.DataFrame(rows))

    st.markdown("---")

    # -----------------------
    # Single-ticker dashboard
    # -----------------------
    st.subheader(f"{ticker} — Dashboard")

    # Price chart
    df = _load_prices(ticker)
    st.markdown("### Price & Moving Averages")
    if df.empty or "close" not in df.columns:
        st.warning("No prices found. Use refresh or run the MVP pipeline.")
    else:
        try:
            st.plotly_chart(_price_chart(df), config={"displayModeBar": False, "responsive": True})
        except Exception as e:
            st.error(f"Failed to render price chart: {e}")

        # RSI panel
        try:
            st.plotly_chart(_rsi_panel(df), config={"displayModeBar": False, "responsive": True})
        except Exception as e:
            st.error(f"Failed to render RSI panel: {e}")

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

        from datetime import datetime as _dt
        latest_path = max(sent_paths, key=lambda p: p.stat().st_mtime)
        latest_time = latest_path.stat().st_mtime
        st.caption(
            f"Loaded {len(records)} records from {len(sent_paths)} file(s). "
            f"Latest file: {latest_path.name} @ {_dt.fromtimestamp(latest_time).isoformat(sep=' ', timespec='seconds')}"
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

        news_dir1 = OUT_ROOT / "news" / ticker
        news_dir2 = OUT_ROOT / ticker / "news"
        has_news = (news_dir1.exists() and any(news_dir1.glob("*"))) or (
            news_dir2.exists() and any(news_dir2.glob("*"))
        )
        if has_news:
            st.info("News exists but no sentiment files were found. Run the sentiment step or fix the writer path/extension.")
        else:
            st.info("No news found for this ticker/date window. Check your since filter, ticker mapping, or API key.")

    # -----------------------
    # AI Analyst (expander)
    # -----------------------
    st.markdown("---")
    with st.expander("AI Analyst", expanded=False):
        render_analyst_widget(
            ticker,
            data_dir=str(DATA_ROOT),
            out_dir=str(OUT_ROOT),
            sent_root=str(OUT_ROOT / "sentiment"),
        )

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
