# AI Stock Multi-Agent

## Getting Started (1-minute quick recipe)
# 1) Create and activate venv
python3 -m venv .venv && source .venv/bin/activate

# 2) Install
pip install -r requirements.txt

# 3) Configure env (create .env in repo root)
#   Required for fundamentals: FMP_API_KEY=...
#   Optional LLM (enable richer Analyst): OPENAI_API_KEY=...

# 4) Build artifacts for a ticker
PYTHONPATH=src python -m cli refresh --ticker AAPL --since 2023-10-01 --limit 20

# 5) Generate report (optional; refresh already does this)
PYTHONPATH=src python -m reporter.report_generator --ticker AAPL --out-dir output --data-dir data

# 6) Launch dashboard
streamlit run src/ui/app.py

</details>

A small, production-lean multi-agent pipeline that turns raw market data into a concise research report and a Streamlit dashboard.

Stable price plotting (Plotly) with MA50/MA200

News → sentiment scoring (FinBERT by default)

Fundamentals (FinancialModelingPrep, normalized)

Technical signals (RSI14, ATR14 percentile, 52-week proximity, simple flags)

Reporter generates report.md (facts + optional Analyst Take)

Streamlit UI with “Ask the Analyst” widget (retrieval-only; offline fallback)

Quick start
# 1) Create a virtual env
python3 -m venv .venv
source .venv/bin/activate

# 2) Install
pip install -r requirements.txt

# 3) Add minimal env (see sample below) to .env
#    Required: OPENAI_API_KEY for Analyst/Reporter (optional if ANALYST_ENABLED=0)
#    Required: FMP_API_KEY for fundamentals ingest

# 4) Pull data and build artifacts for a ticker
PYTHONPATH=src python -m cli refresh --ticker AAPL --since 2023-10-01 --limit 20

# 5) Generate (or regenerate) report explicitly (optional)
PYTHONPATH=src python -m reporter.report_generator --ticker AAPL --out-dir output --data-dir data

# 6) Run the dashboard
streamlit run src/ui/app.py


Open http://0.0.0.0:8501
 (or the URL Streamlit prints).

What you get after a refresh
data/
  AAPL/
    prices.parquet
    technical.json
output/
  AAPL/
    prices.parquet
    report.md
  fundamentals/
    AAPL.json             # normalized FMP metrics
  news/
    AAPL/news.jsonl
  sentiment/
    AAPL/0001.json ...    # per-article scored JSONs


The dashboard reads from these artifacts; the reporter combines them into report.md (Technical + Fundamentals + Sentiment + optional Analyst Take).

Environment (.env)

Create a file named .env in the repo root:

# --- Reporter / Analyst (OpenAI) ---
OPENAI_API_KEY=sk-............................................
REPORTER_MODEL=gpt-4o-mini
REPORTER_TEMPERATURE=0.3

# --- Fundamentals (FinancialModelingPrep) ---
FMP_API_KEY=your_fmp_key_here

# --- AI Analyst controls ---
ANALYST_ENABLED=1           # set 0 for offline fallback (no API calls)
ANALYST_MODEL=gpt-4o-mini
ANALYST_TEMPERATURE=0.2
ANALYST_MAX_TOKENS=300

# --- optional cache / perf ---
# HF_HUB_DISABLE_PROGRESS_BARS=1
# TRANSFORMERS_CACHE=/abs/path/.hf-cache
# HF_HOME=/abs/path/.hf-home


Tip: For CI we set ANALYST_ENABLED=0 to force offline, deterministic output.

Streamlit dashboard

Price & Moving Averages (Close, MA50, MA200)

Technical flags (above MA50, above MA200, overbought/oversold, volatility high)

Sentiment Snapshot (headline count, average score, pos/neg %; diagnostics)

AI Analyst (expander)

“Ask the Analyst” text box

Mode badge: LLM or offline (fallback uses only local facts)

Report (download + rendered Markdown)

CLI
# Help
PYTHONPATH=src python -m cli --help

# Full refresh for a ticker: news → sentiment → prices → fundamentals → technical → report
PYTHONPATH=src python -m cli refresh --ticker AAPL --since 2023-10-01 --limit 20


Use tickers_mvp.txt (repo root) to keep a small cross-sector watchlist for demos.

Tests
# unit tests (CI uses this target)
pytest -q -m "not integration" --disable-warnings --maxfail=1

# optional: a tiny analyst smoke test exists and runs offline by default

CI (GitHub Actions)

Workflow: .github/workflows/ci.yml

Sets PYTHONPATH=src, ANALYST_ENABLED=0 to keep tests deterministic/offline.

Unit job always runs; integration job is graceful when none are collected.

Technical details
Fundamentals (FMP)

Adapter: src/ingest/fundamentals_fmp.py

Normalized to output/fundamentals/<TICKER>.json, e.g.:

{
  "company": { "symbol": "AAPL", "name": "Apple Inc." },
  "metrics": {
    "revenue_yoy": 0.07,
    "net_income_yoy": 0.09,
    "gross_margin": 0.447,
    "operating_margin": 0.297
  }
}

Technical signals

Module: src/agents/technical.py

Computes: RSI(14), ATR(14) + 1-year percentile, distance to 52-week high/low, flags (above MA50/MA200, overbought/oversold, volatility high) → data/<TICKER>/technical.json.

Sentiment

Ingests RSS Google News, scores each headline with FinBERT (default)

Output: per-article JSON files in output/sentiment/<TICKER>/

Reporter is tolerant: handles dict files and legacy array files (but you should stick to per-article JSONs for consistency).

Reporter

Module: src/reporter/report_generator.py

Loads artifacts from data/ and output/, writes output/<TICKER>/report.md.

Adds Analyst Take if enabled.

AI Analyst

Module: src/analyst/compose.py

Builds a small facts bundle from local artifacts and generates a cautious, short summary.

If ANALYST_ENABLED=0 or no OPENAI_API_KEY, uses an offline fallback (deterministic template).

Streamlit widget: src/ui/analyst_widget.py.

Troubleshooting

No prices / empty chart
Run a refresh:
PYTHONPATH=src python -m cli refresh --ticker AAPL --since 2023-10-01 --limit 20

Sentiment shows zero
Check output/news/<TICKER>/news.jsonl exists; ensure scorer wrote per-article JSONs to output/sentiment/<TICKER>/.

Analyst widget says “No answer”

For LLM mode: verify OPENAI_API_KEY and ANALYST_ENABLED=1.

For fallback mode: ensure artifacts exist in data/ + output/.

CI failures on imports
Ensure PYTHONPATH=src is set (CI already does this).

Roadmap

Multi-ticker batch refresh & portfolio view

More fundamentals (TTM growth blends, quality/efficiency metrics)

Additional TA (MACD, Bollinger, 200/50 crossovers)

Richer Analyst prompts + source snippets in UI

Optional live sources (guarded; off by default)

License

MIT © 2025 rkendev