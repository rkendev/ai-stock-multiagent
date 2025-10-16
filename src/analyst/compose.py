# src/analyst/compose.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List


# ---- Fact gathering ---------------------------------------------------------

def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _load_prices_snapshot(ticker: str, data_dir: Path, out_dir: Path) -> Dict[str, Any]:
    # only the last close & 30d change (report already handles plotting)
    import pandas as pd
    candidates = [
        data_dir / ticker / "prices.parquet",
        out_dir / ticker / "prices.parquet",
    ]
    for c in candidates:
        if c.exists():
            df = pd.read_parquet(c)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close_col = "Close" if "Close" in df.columns else next((c for c in df.columns if str(c).lower()=="close"), None)
            if close_col is None or len(df) < 2:
                break
            last = float(df[close_col].iloc[-1])
            lookback = min(len(df)-1, 30)
            pct30 = (last / float(df[close_col].iloc[-lookback]) - 1.0) * 100.0
            return {"last": last, "pct30": pct30}
    return {}

def _load_technical(ticker: str, data_dir: Path) -> Dict[str, Any]:
    p = data_dir / ticker / "technical.json"
    obj = _read_json(p) or {}
    sig = obj.get("signals", {}) or {}
    ind = obj.get("indicators", {}) or {}
    return {
        "above_ma50": bool(sig.get("above_MA50")),
        "above_ma200": bool(sig.get("above_MA200")),
        "overbought": bool(sig.get("overbought")),
        "oversold": bool(sig.get("oversold")),
        "volatility_high": bool(sig.get("rally_volatility_high")),
        "rsi14": float(ind.get("RSI14", 0.0))
    }

def _load_sentiment(ticker: str, sent_root: Path) -> Dict[str, Any]:
    d = sent_root / ticker
    if not d.exists():
        return {}
    pos = neu = neg = 0
    scores: List[float] = []
    for f in d.glob("*.json"):
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, list):
            objs = [o for o in obj if isinstance(o, dict)]
        elif isinstance(obj, dict):
            objs = [obj]
        else:
            continue
        for it in objs:
            label = str(it.get("sentiment", "")).upper()
            if label == "POSITIVE": pos += 1
            elif label == "NEGATIVE": neg += 1
            else: neu += 1
            try: scores.append(float(it.get("score", 0.0)))
            except Exception: pass
    n = max(1, pos+neu+neg)
    avg = (sum(scores)/len(scores)) if scores else 0.0
    return {"n": pos+neu+neg, "avg": avg, "pos_pct": 100*pos/n, "neg_pct": 100*neg/n}

def _load_fundamentals(t: str, data_dir: Path, out_dir: Path) -> Dict[str, Any]:
    cands = [
        out_dir.parent / "fundamentals" / f"{t}.json",
        data_dir / t / "fundamentals.json",
        data_dir / t / "fundamental.json",
    ]
    for p in cands:
        obj = _read_json(p)
        if obj:
            m = obj.get("metrics", {}) or {}
            return {
                "rev_yoy": m.get("revenue_yoy"),
                "ni_yoy": m.get("net_income_yoy"),
                "gm": m.get("gross_margin"),
                "om": m.get("operating_margin"),
                "company": (obj.get("company") or {}).get("name")
            }
    return {}

def build_facts(ticker: str, data_dir: str = "data", out_dir: str = "output", sent_root: str = "output/sentiment") -> Dict[str, Any]:
    data = Path(data_dir); out = Path(out_dir); sent = Path(sent_root)
    facts = {
        "ticker": ticker,
        "price": _load_prices_snapshot(ticker, data, out),
        "tech": _load_technical(ticker, data),
        "sent": _load_sentiment(ticker, sent),
        "fund": _load_fundamentals(ticker, data, out),
    }
    return facts

# ---- LLM integration ---------------------------------------------------------

def _openai_enabled() -> bool:
    return bool(os.getenv("ANALYST_ENABLED", "1")) and bool(os.getenv("OPENAI_API_KEY"))

def _model() -> str:
    return os.getenv("ANALYST_MODEL") or os.getenv("REPORTER_MODEL", "gpt-4o-mini")

def _temperature() -> float:
    try:
        return float(os.getenv("ANALYST_TEMPERATURE", "0.2"))
    except Exception:
        return 0.2

def _max_tokens() -> int:
    try:
        return int(os.getenv("ANALYST_MAX_TOKENS", "300"))
    except Exception:
        return 300

_SYSTEM = (
    "You are a cautious sell-side equity analyst. "
    "Only use the facts provided. Do not invent data or cite live sources. "
    "Keep the tone measured and include uncertainty where appropriate. "
    "Write 4–6 sentences max."
)

def render_analyst_take(facts: Dict[str, Any]) -> str:
    """
    Returns a short 'Analyst Take' paragraph. If LLM disabled or missing,
    returns a deterministic, template-based summary instead.
    """
    if not _openai_enabled():
        # fallback summary (deterministic)
        p = facts.get("price", {})
        t = facts.get("tech", {})
        s = facts.get("sent", {})
        f = facts.get("fund", {})
        parts = []
        if p: parts.append(f"Price ≈ {p.get('last', 'N/A'):.2f} ({p.get('pct30', 0.0):+.1f}% over 30 days).")
        if t:
            trend = "above MA200" if t.get("above_ma200") else ("above MA50" if t.get("above_ma50") else "below key MAs")
            mom = "overbought" if t.get("overbought") else ("oversold" if t.get("oversold") else "neutral momentum")
            parts.append(f"Trend: {trend}; Momentum: {mom}.")
        if f:
            def pct(v): 
                try: return f"{float(v)*100:+.1f}%"
                except: return "N/A"
            parts.append(f"Fundamentals: Rev YoY {pct(f.get('rev_yoy'))}, NI YoY {pct(f.get('ni_yoy'))}.")
        if s and s.get("n", 0)>0:
            parts.append(f"News mix: avg {s.get('avg',0.0):+.2f}, {s.get('pos_pct',0.0):.0f}% positive / {s.get('neg_pct',0.0):.0f}% negative.")
        parts.append("Overall: preliminary view only; consider broader context before making decisions.")
        return " ".join(parts)

    # LLM path
    from openai import OpenAI
    client = OpenAI()  # reads env vars

    user = {
        "ticker": facts.get("ticker"),
        "price": facts.get("price"),
        "technical": facts.get("tech"),
        "sentiment": facts.get("sent"),
        "fundamentals": facts.get("fund"),
        "instruction": "Summarize implications for short-term outlook without making recommendations."
    }

    prompt = (
        "Facts JSON follows. Summarize briefly and cautiously. "
        "Avoid absolutes; do not invent data.\n\n"
        f"{json.dumps(user, ensure_ascii=False)}"
    )

    resp = client.chat.completions.create(
        model=_model(),
        temperature=_temperature(),
        max_tokens=_max_tokens(),
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip() if resp and resp.choices else ""


def log_qa(ticker: str, question: str, answer: str, out_dir: str = "output") -> str:
    """
    Append a single Q/A record to output/<TICKER>/analyst_qa.jsonl and return the file path.
    Safe for repeated calls and missing directories.
    """
    safe = ticker.upper().strip()
    folder = Path(out_dir) / safe
    folder.mkdir(parents=True, exist_ok=True)
    fpath = folder / "analyst_qa.jsonl"

    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ticker": safe,
        "q": (question or "").strip(),
        "a": (answer or "").strip(),
        "source": "ui",        # or "reporter" later if you log from other places
        "version": 1
    }
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(fpath)
