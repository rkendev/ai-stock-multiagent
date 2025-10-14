from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Prefer shared scorer; fall back safely if missing
try:
    from sentiment.score import score_text as _score_text  # type: ignore
    from sentiment.score import score_articles_for_ticker as _score_for_ticker  # type: ignore
except Exception:  # pragma: no cover
    _score_text = None  # type: ignore[assignment]
    _score_for_ticker = None  # type: ignore[assignment]

# News fetcher used by tests/legacy agents (mock-friendly in CI)
try:
    from ingest.news import fetch_rss_news  # our back-compat shim
except Exception:  # pragma: no cover
    fetch_rss_news = None  # type: ignore[assignment]


def _ensure_scorer():
    """Return a callable(text)->(score,label) even if sentiment.score is missing."""
    if _score_text is not None:
        return _score_text

    # Tiny inline fallback (same heuristics used elsewhere)
    POS_WORDS = {
        "beats", "beat", "record", "surge", "rally", "upgraded",
        "strong", "growth", "bullish", "outperform"
    }
    NEG_WORDS = {
        "miss", "misses", "downgraded", "plunge", "fall", "weak",
        "bearish", "lawsuit", "recall", "layoff"
    }

    def _fallback(text: str) -> tuple[float, str]:
        t = text.lower()
        pos = sum(w in t for w in POS_WORDS)
        neg = sum(w in t for w in NEG_WORDS)
        raw = pos - neg
        if raw > 0:
            return (min(1.0, 0.1 * raw), "POSITIVE")
        if raw < 0:
            return (max(-1.0, -0.1 * abs(raw)), "NEGATIVE")
        return (0.0, "NEUTRAL")

    return _fallback


def _score_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add (score, sentiment) to each record; tolerate missing fields.
    Expects keys the tests use: title, summary (optional), link, published.
    """
    do_score = _ensure_scorer()
    out: List[Dict[str, Any]] = []
    for r in records:
        title = str(r.get("title", "")).strip()
        summary = str(r.get("summary", "")).strip()
        text = f"{title} {summary}".strip()
        sc, label = do_score(text)
        rr = dict(r)
        rr["score"] = float(sc)
        rr["sentiment"] = label
        out.append(rr)
    return out


def _read_scored_from_disk(sent_dir: Path) -> List[Dict[str, Any]]:
    """Read scored JSONs from <sent_dir>/*.json as a list of dicts."""
    items: List[Dict[str, Any]] = []
    if sent_dir.exists():
        import json
        for p in sorted(sent_dir.glob("*.json")):
            try:
                items.append(json.loads(p.read_text()))
            except Exception:
                continue
    return items


def _derive_in_root_from_news_root(news_root: str | Path) -> Path:
    """
    Our scorer expects: in_root / 'news' / <TICKER> / news.jsonl
    If caller provides news_root='.../output/news', then in_root must be the parent ('.../output').
    If caller already passes parent, we just return it unchanged.
    """
    nr = Path(news_root)
    return nr.parent if nr.name == "news" else nr


def _derive_written_sent_dir(out_root: str | Path, ticker: str) -> Path:
    """
    Tests sometimes pass out_root='<tmp>/output/sentiment' and expect files in
    '<tmp>/output/sentiment/<TICKER>/*.json'. Our scorer writes to base_out/'sentiment'/<TICKER>.
    So:
      - if out_root already ends with 'sentiment', the final dir is out_root/<TICKER>
      - else it's out_root/'sentiment'/<TICKER>
    """
    orp = Path(out_root)
    if orp.name == "sentiment":
        return orp / ticker
    return orp / "sentiment" / ticker


def _base_out_for_scorer(out_root: str | Path) -> Path:
    """
    Map the caller's out_root to the scorer's base_out:
      scorer writes to base_out/'sentiment'/<TICKER>.
      If caller passed '.../output/sentiment', base_out should be '.../output'.
    """
    orp = Path(out_root)
    return orp.parent if orp.name == "sentiment" else orp


def run_sentiment(
    ticker: str,
    *args: Any,
    in_root: Optional[str] = None,
    out_root: str = "output",
    news_file: str = "news.jsonl",
    news_root: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Backwards-compatible entrypoint used by tests and agents.

    Supported call styles:

    1) Legacy in-memory (returns a LIST of scored items):
         run_sentiment(ticker, limit:int)

    2) On-disk pipeline (returns a TUPLE (out_dir: Path, n_written: int)):
         run_sentiment(ticker,
                       news_root='<...>/output/news',  # or in_root='<...>/output'
                       out_root='<...>/output/sentiment',  # or '<...>/output'
                       news_file='news.jsonl')

       - Writes JSONs to:
            if out_root endswith 'sentiment': <out_root>/<ticker>/*.json
            else:                             <out_root>/sentiment/<ticker>/*.json
       - Returns (out_dir, n_written) to match test expectations.
    """

    # ---- Style (1): legacy second positional arg is integer limit ----
    if args and isinstance(args[0], int):
        limit = int(args[0])
        if fetch_rss_news is None:
            raise RuntimeError("ingest.news.fetch_rss_news is not available")
        news_items = fetch_rss_news(ticker, limit)  # our shim pads to exactly `limit`
        return _score_records(news_items)

    # ---- Style (2): on-disk pipeline ----
    if _score_for_ticker is None:
        raise RuntimeError("sentiment.score.score_articles_for_ticker not available")

    # Map news_root (legacy) to in_root expected by the scorer
    effective_in_root: Path
    if news_root:
        effective_in_root = _derive_in_root_from_news_root(news_root)
    elif in_root:
        effective_in_root = Path(in_root)
    else:
        # default to 'output', consistent with scorer defaults
        effective_in_root = Path("output")

    # The scorer writes to base_out/'sentiment'/<TICKER>
    base_out = _base_out_for_scorer(out_root)

    # Run the scorer; it returns count written
    n_written = _score_for_ticker(ticker, in_root=str(effective_in_root), out_root=str(base_out), news_file=news_file)

    # Compute actual directory where files were written
    out_dir = _derive_written_sent_dir(out_root, ticker)

    return out_dir, n_written


__all__ = ["run_sentiment"]
