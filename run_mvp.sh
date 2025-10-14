#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
TICKER="${1:?Usage: ./run_mvp.sh <TICKER> <SINCE>}"
SINCE="${2:?Usage: ./run_mvp.sh <TICKER> <SINCE>}"

export PYTHONPATH=src

# 1) Prices + technical
python -m cli --ticker "$TICKER" --since "$SINCE"
python -m agents.technical --ticker "$TICKER" --data-dir data --out-dir data

# 2) Fundamentals (best-effort)
python -m ingest.fundamentals --ticker "$TICKER" --out-root data || true
python -m agents.fundamental --ticker "$TICKER" --data-dir data --out-dir data || true

# 3) News + sentiment  <<< THIS IS THE MISSING PART
python -m ingest.news --ticker "$TICKER" --out-root output --limit 10
python -m agents.sentiment --ticker "$TICKER" --news-root output/news --out-root output/sentiment --limit 10

# 4) Visuals (optional)
python -m agents.visualizer --ticker "$TICKER" --data-dir data --out-dir output || true

# 5) Reporter LAST (will now include sentiment)
python -m reporter.report_generator --ticker "$TICKER" --data-dir data --out-dir output

echo
echo "Artifacts:"
echo "  data/$TICKER/prices.parquet"
echo "  data/$TICKER/technical.json"
echo "  output/$TICKER/charts/price_ma50.png"
echo "  output/$TICKER/report.md"
echo "  output/news/$TICKER/news.jsonl"
echo "  output/sentiment/$TICKER/*.json"
